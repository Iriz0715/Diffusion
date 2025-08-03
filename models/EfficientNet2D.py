import os, re, glob, logging
import numpy as np
import h5py, json
import tensorflow as tf
from tensorflow.keras import layers

from classification_2d import BaseModel
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from networks import modify_input_channels

DEFAULT_MODEL_CONFIG = {
    'epoch': 1000,  # total_iterations = iters_per_epoch * epoch
    'batch_size': 32,
    'iters_per_epoch': 3200,  # each epoch only runs 3200 iters
    'period': 50,  # save/validate period
    # mutli-class: using Categorical*; multi-label: using Binary*
    'loss_type': 'CategoricalCrossentropy',
    'norm_config': {'norm': True, 'norm_channels':  'rgb_channels'},
    'mirror_config': {'training_mirror': True, 'testing_mirror': False, 'mirror_axes': [1, 2]},
    'aug_config': {}, # additional augmentation config
    'policy': 'float32',  # ['float16', 'mixed_float16', 'float32']
    'pretrained': True,
    'initial_lr': 2e-4,
    'end_lr': 1e-7,
}

class EfficientNet2D(BaseModel):
    def __init__(self, checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads=6,
                 input_channels=3, channel_dropout=False, class_weights=None, left_right_swap_config=None, mirror_config=None, 
                 model_config=None, **kwargs):
        
        super(EfficientNet2D, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        
        self.mirror_config = mirror_config
        
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
        
        self.policy = model_config['policy']
        if self.policy != 'float32':
            ### TODO: currently only supporting float32
            get_policy = tf.keras.mixed_precision.Policy(self.policy)
            tf.keras.mixed_precision.set_global_policy(get_policy)

        self.aug_config = model_config.get('aug_config', {})
        self.group = group
        
        self.batch_size = int(model_config['batch_size'])
        self.epoch = int(model_config['epoch'])
        self.period = model_config['period']
        self.norm_config = model_config['norm_config']
        
        self.input_channels = input_channels

        self.rois = rois
        self.nclass = len(self.rois) + 1
        
        self.num_threads = num_threads
        
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        
        self.loss_type = model_config['loss_type']
        self.initial_lr = model_config['initial_lr']
        self.end_lr = model_config['end_lr']
        
        self.im_size = im_size

        self.pretrained = model_config.get('pretrained', False)
        
        self.counter = 0
        self._loaded, self.counter = self.load()
        if not self._loaded:
            print('Initializing...')
            self.networks = self.build_networks()
            print('Complete initialization.')

        # Log variables
        vars_log_path = os.path.join(self.log_dir, self.model_dir, 'vars.txt')
        os.makedirs(os.path.dirname(vars_log_path), exist_ok=True)
        self.vars_log_path = vars_log_path
        self_vars = {k: vars(self).get(k) for k in dir(self)
                     if not k.startswith('_') and vars(self).get(k) is not None}
        logging.basicConfig(filename=vars_log_path, level=logging.INFO)
        logging.info(self_vars)
    
    def compile_it(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)
        self.compile(optimizer=self.opt, loss=self.loss_fn, metrics=[tf.keras.metrics.AUC()])
        return

    def get_optimizer(self):
        starter_learning_rate = self.initial_lr
        end_learning_rate = self.end_lr
        power = 0.9
        decay_steps = self.steps_per_epoch * (self.epoch - self.counter)
        current_learning_rate = ((starter_learning_rate - end_learning_rate)
                                 * (1 - self.counter / self.epoch) ** (power)) + end_learning_rate
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=current_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=False, name=None
        )
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=12)
 
        return opt
    
    def build_networks(self):
        input_shape = self.im_size + [self.input_channels,]
        inputs = layers.Input(input_shape)
        
        pretrain_networks = EfficientNetV2S(include_top=False,
                                            weights='imagenet' if self.pretrained else None, 
                                            input_shape=[input_shape[0], input_shape[1], 3],
                                            pooling='avg',
                                            classes=self.nclass, 
                                            include_preprocessing=False)
        if self.input_channels != 3:
            pretrain_networks = modify_input_channels(pretrain_networks, self.input_channels)
        pretrain_networks.trainable = True

        x = pretrain_networks(inputs)
        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
        outputs = layers.Dense(self.nclass, activation=None, name='logits')(x)
        networks = tf.keras.Model(inputs, outputs, name='EfficientNet2D')
        return networks
    
    @property
    def model_dir(self):
        return 'EffNetV2S-Group' + str(self.group)