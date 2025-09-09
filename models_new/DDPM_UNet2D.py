import os, logging
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers
import sys
sys.path.append('../..')
from DDPM_base_model_2d import DDPMBaseModel2D
from networks import *
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler


DEFAULT_MODEL_CONFIG = {
    'features_root': 64,
    'layers': 4,
    'max_filters': 320,
    'batch_size': 32,
    'attention': False,
    'iters_per_epoch': 1000,  # each epoch runs <= 1000 iters
    'epoch': 1000,
    'save_period': 50,
    'norm_config': {'norm': False, 'norm_channels': 'all_channels',
                    'norm_mean': None, 'norm_std': None},
    'sampling_config': None,
    'initial_lr': 0.01,
    'end_lr': 1e-6,
    'lr_decay': 0.1,
    'optimizer_type': 'SGD', # currently only supporting 'SGD' and 'Adam'
    'sgd_momentum': 0.99,
    'fg_sampling_ratio': 0.5,
    # Diffusion model parameters
    'max_timesteps': 1000,  # Maximum number of timesteps T
    'dropout': 0.1,  # Dropout probability
    'beta_1': 1e-4,
    'beta_T': 0.02,
    'infer_T': 1000,
    'squeue': None
}

class DDPM_UNet2D(DDPMBaseModel2D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, im_size, num_threads, 
                 input_channels=1, output_channels=1, model_config=None, resume=True, **kwargs):
        super(DDPM_UNet2D, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        
        if model_config is None:
            model_config = DEFAULT_MODEL_CONFIG
        else:
            config = DEFAULT_MODEL_CONFIG
            for key in model_config.keys():
                if key not in config:
                    print('[Warning] Unknown configuration key-value:', key, '-', model_config[key])
                config[key] = model_config[key]
            model_config = config
        print('Model configuration:', model_config)
      
        self.sampling_config = model_config.get('sampling_config', None)

        self.batch_size = int(model_config['batch_size'])
        self.epoch = int(model_config['epoch'])
        self.norm_config = model_config['norm_config']

        self.optimizer_type = model_config['optimizer_type']
        self.initial_lr = float(model_config['initial_lr'])
        self.end_lr = float(model_config['end_lr'])
        self.lr_decay = float(model_config['lr_decay'])
        self.fg_sampling_ratio = model_config['fg_sampling_ratio']
        self.features_root = int(model_config['features_root'])
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.layer_number = int(model_config['layers'])
        self.max_filters = int(model_config['max_filters'])
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        self.save_period = model_config['save_period']
        self.im_size = im_size
        self.enlarged_im_size = (int(im_size[0] * 1.1875), int(im_size[1] * 1.1875))
        self.attention = model_config['attention']
        self.counter = 0
        self.resume = resume

        #### Diffusion
        self.max_timesteps = model_config['max_timesteps']
        self.dropout = model_config['dropout']
        self.beta_1 = model_config['beta_1']
        self.beta_T = model_config['beta_T']
        self.squeue = model_config['squeue']
        self.infer_T = model_config['infer_T']
        
        if resume==True:
            print("[INFO] Resume mode: attempting to load checkpoint...")
            _loaded, self.counter = self.load()
        else:
            print("[INFO] New training mode: starting from scratch...")
            _loaded = False
            self.counter = 0
        if not _loaded:
            self.unet = self.build_diffusion_model()
            self.unet.summary()

        print(f'T={self.max_timesteps}, infer_T={self.infer_T}')
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.unet,
            beta_1=self.beta_1,
            beta_T=self.beta_T,
            T=self.max_timesteps
        )
        self.diffusion_sampler = GaussianDiffusionSampler(
                model=self.unet,
                beta_1=self.beta_1,
                beta_T=self.beta_T,
                T=self.max_timesteps,
                infer_T=self.infer_T,
                squeue=self.squeue,
            )
        print("Diffusion trainer & Sampler created successfully!!")
        
        # Log variables
        vars_log_path = os.path.join(self.log_dir, self.model_dir, 'vars.txt')
        os.makedirs(os.path.dirname(vars_log_path), exist_ok=True)
        self.vars_log_path = vars_log_path
        self_vars = {k: vars(self).get(k) for k in dir(self)
                     if not k.startswith('_') and vars(self).get(k) is not None}
        logging.basicConfig(filename=vars_log_path, level=logging.INFO)
        logging.info(self_vars)
    
    def get_optimizer(self):
        initial_learning_rate = self.initial_lr
        end_learning_rate = self.end_lr
        power = 1 - self.lr_decay
        decay_steps = self.steps_per_epoch * (self.epoch - self.counter)
        current_learning_rate = ((initial_learning_rate - end_learning_rate)
                                 * (1 - self.counter / self.epoch) ** (power)) + end_learning_rate
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(     # 学习率随时间衰减,而不是根据loss变化
            initial_learning_rate=current_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=False, name=None
        )
        if self.optimizer_type == 'SGD':
            # momentum 0.99 is good for most situations
            # if the training/testing dice is 0, we could lower down the momentum to 0.95
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=self.sgd_momentum, clipnorm=12,
                                          nesterov=True)
        elif self.optimizer_type == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=12)
 
        return opt

    def compile_it(self):
        self.opt = self.get_optimizer()
        self.optimizer = self.opt
        def diffusion_loss(y_true, output_loss):
            """Placeholder loss function for diffusion model"""
            # The actual loss is computed in diffusion_trainer.forward
            return tf.reduce_mean(tf.square(output_loss))
        self.unet.compile(
            optimizer=self.optimizer,
            loss=diffusion_loss
        )
        print(f"Diffusion model compiled - Optimizer: {self.optimizer_type}, Initial learning rate: {self.initial_lr}")
    
    @tf.function
    def loss_fn(self, labels, probs, mask=None):
        # The actual loss is computed in diffusion_trainer.forward
        return tf.reduce_mean(probs)

    def build_diffusion_model(self):
        unet = build_diffusion_unet(
            im_size=self.im_size, 
            nclass=self.output_channels,
            input_channels=self.input_channels, 
            features_root=self.features_root,
            layer_number=self.layer_number, 
            max_filters=self.max_filters, 
            attention=self.attention,
            dropout=self.dropout,
        )

        input_shape = [None for _ in self.im_size] + [self.input_channels,]
        img_input = layers.Input(shape=input_shape, name='image_input')
        time_input = layers.Input(shape=[], dtype=tf.int32, name='time_input')
        x = unet([img_input, time_input])
        outputs = x
        model = Model(inputs=[img_input, time_input], outputs=outputs)
        return model

    @property
    def model_dir(self):
        return 'transform'
    
