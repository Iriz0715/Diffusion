import os, re, logging
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers

from image_transform_base_model_2d import ImageTransformBaseModel2D
from networks import *
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

DEFAULT_MODEL_CONFIG = {
    'is_natural_image': False,
    'features_root': 64,
    'conv_size': 3,
    'use_upsampling': True,  # upsample+conv > convT in image transform. Then deconv_size can be 3
    'deconv_size': 3,  # without use_upsampling, deconv_size is better with 2
    'layers': 4,
    'max_filters': 320,
    'dilation': False,
    'loss_type': 'l1', # l1 or hybrid (l1 + dice)
    'batch_size': 32,
    'deep_supervision': False, # not applicable
    'attention': False,
    'iters_per_epoch': 1000,  # each epoch runs <= 1000 iters
    'epoch': 1000,
    'save_period': 50,
    'norm_config': {'norm': False, 'norm_channels': 'all_channels',
                    'norm_mean': None, 'norm_std': None},
    'sampling_config': None,
    'simulation_config': None,
    'mirror_config': {'training_mirror': False, 'testing_mirror': False,
                      'mirror_axes': [1, 2], 'rot90': True}, # axis 0 is batch
    'augmentation_params': {},
    'initial_lr': 0.01,
    'end_lr': 1e-6,
    'lr_decay': 0.1,
    'optimizer_type': 'SGD', # currently only supporting 'SGD' and 'Adam'
    # only apply to SGD
    # momentum 0.99 is good for most situations
    # if the training/testing dice is 0, we could lower down the momentum to 0.95
    # if the label is very heterogenous, probably lower the momentum to 0.90
    'sgd_momentum': 0.99,
    # increase this ratio when there are very small structures
    'fg_sampling_ratio': 0.5,
    ## currently only supporting float32
    ## float16 speed-up requires tensorflow>=2.4.0 and cudnn>=8
    'policy': 'float32',  # ['float16', 'mixed_float16', 'float32']
    
    # experimental
    'residual': True,
    'activation': 'sigmoid'
}

class ImageTransformUNet2D(ImageTransformBaseModel2D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, im_size, num_threads, 
                 input_channels=1, output_channels=1, model_config=None, **kwargs):
        super(ImageTransformUNet2D, self).__init__()
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
        self.simulation_config = model_config.get('simulation_config', None)
        self.mirror_config = model_config['mirror_config']
        self.augmentation_params = model_config.get('augmentation_params', {})
        self.tta = model_config.get('tta', {})
        
        self.add_identity_sample = model_config.get('add_identity_sample', False)
        self.identity_sampling_ratio = model_config.get('identity_sampling_ratio', 0)
        
        self.bin_weights = []
        if self.sampling_config is not None:
            self.bin_weights = self.sampling_config.get('bin_weights', [])
        
        self.policy = model_config['policy']
        if self.policy != 'float32':
            ### TODO: currently only supporting float32
            get_policy = tf.keras.mixed_precision.Policy(self.policy)
            tf.keras.mixed_precision.set_global_policy(get_policy)
        
        self.batch_size = int(model_config['batch_size'])
        self.epoch = int(model_config['epoch'])
        self.features_root = int(model_config['features_root'])
        self.loss_type = model_config['loss_type']
        self.norm_config = model_config['norm_config']

        self.optimizer_type = model_config['optimizer_type']
        self.initial_lr = float(model_config['initial_lr'])
        self.end_lr = float(model_config['end_lr'])
        self.lr_decay = float(model_config['lr_decay'])
        self.sgd_momentum = model_config['sgd_momentum']
        self.fg_sampling_ratio = model_config['fg_sampling_ratio']
        self.features_root = int(model_config['features_root'])
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_size = int(model_config['conv_size'])
        self.deconv_size = int(model_config['deconv_size'])
        self.use_upsampling = model_config['use_upsampling']
        self.layer_number = int(model_config['layers'])
        self.max_filters = int(model_config['max_filters'])
        self.dilation = model_config['dilation']
        self.deep_supervision = model_config['deep_supervision']
        self.num_threads = num_threads
        
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        
        self.save_period = model_config['save_period']
        
        self.im_size = im_size
        self.enlarged_im_size = (int(im_size[0] * 1.1875), int(im_size[1] * 1.1875))
        
        self.strides_list = get_strides_list(self.layer_number, self.im_size)
        
        self.counter = 0

        # Use natural image model config
        self.is_natural_image = bool(model_config['is_natural_image'])
        
        # experimental
        self.residual = model_config['residual']
        self.attention = model_config['attention']
        
        self.activation = model_config['activation']


        #### Diffusion
        self.use_time_embedding = model_config.get('use_time_embedding',False)  # 是否使用时间嵌入
        self.max_timesteps = model_config.get('max_timesteps', 1000)  # 最大时间步数 T
        self.dropout = model_config.get('dropout', 0.0)  # dropout概率
        
        # 添加 Diffusion 相关参数
        self.beta_1 = model_config.get('beta_1', 0.0001)
        self.beta_T = model_config.get('beta_T', 0.02)
        self.is_diffusion = model_config.get('is_diffusion', False)  # 是否使用扩散模型
        
        _loaded, self.counter = self.load()
        if not _loaded:
            self.unet = self.build_model()
            # 如果是扩散模型，创建 trainer
            if self.is_diffusion:
                self.diffusion_trainer = GaussianDiffusionTrainer(
                    model=self.unet,
                    beta_1=self.beta_1,
                    beta_T=self.beta_T,
                    T=self.max_timesteps
                )
            self.unet.summary()

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
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
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
        if hasattr(self, 'is_diffusion') and self.is_diffusion:
            # 对于扩散模型，不使用标准的 compile
            self.optimizer = self.opt
        else:
            self.compile(optimizer=self.opt, loss=self.loss_fn)
    
    @tf.function
    def loss_fn(self, labels, probs, mask=None):
        # 如果是扩散模型，使用不同的损失计算
        if hasattr(self, 'is_diffusion') and self.is_diffusion:
            # labels 是目标 CT 图像，probs 是条件 CBCT 图像（在这种情况下）
            # 实际的损失计算在 diffusion_trainer.forward 中进行
            return tf.reduce_mean(probs)  # 占位符，实际损失在训练步骤中计算
        
        computed_l1_loss = l1_loss(gt=labels[..., :1], pred=probs[..., :1], mask=mask)
        eps = 1e-5
        if self.loss_type == 'hybrid':
            # Assume mask in channel 1 and it's a binary mask
            seg_labels = tf.cast(tf.one_hot(tf.cast(labels[..., 1], tf.uint8), depth=2), tf.float32)
            seg_logits = tf.stack([tf.math.log((1.0 - probs[..., 1] + eps) / (probs[..., 1] + eps)), 
                                   tf.math.log((probs[..., 1] + eps) / (1.0 - probs[..., 1] + eps))], axis=-1)
            computed_seg_loss, _, dice = segmentation_loss(labels=seg_labels, logits=seg_logits)
            loss = 0.99 * computed_l1_loss + 0.01 * computed_seg_loss
        else:
            loss = computed_l1_loss
        return loss
    
    def build_model(self):
        unet = build_unet(
            im_size=self.im_size, 
            nclass=self.output_channels,
            strides_list=self.strides_list,
            input_channels=self.input_channels, 
            features_root=self.features_root,
            conv_size=self.conv_size, 
            deconv_size=self.deconv_size, 
            layer_number=self.layer_number, 
            max_filters=self.max_filters, 
            dilation=self.dilation, 
            attention=self.attention,
            deep_supervision=False,
            use_upsampling=self.use_upsampling,
            use_residual_encoder=self.residual,
            ######## Diffusion
            use_time_embedding=False,    # 是否使用时间嵌入
            max_timesteps=1000,          # 最大时间步数 T
            time_embedding_dim=None,     # 时间嵌入维度，默认为 features_root * 4
            dropout=0.0,                 # dropout概率
        )
        input_shape = [None for _ in self.im_size] + [self.input_channels,]
        inputs = layers.Input(shape=input_shape)
        x = unet(inputs)
        if self.activation == 'hybrid':
            outputs_0 = layers.Activation('relu')(x[..., 0 : 1])
            outputs_1 = layers.Activation('sigmoid')(x[..., 1 : 2])
            outputs = layers.Concatenate(axis=-1)([outputs_0, outputs_1])
        else:
            outputs = layers.Activation('sigmoid')(x)
        model = Model(inputs, outputs)
        return model

    @property
    def model_dir(self):
        return 'transform'
    





















    def diffusion_train_step(self, x, c):
        """自定义的扩散模型训练步骤"""
        with tf.GradientTape() as tape:
            # x: CBCT 图像 (condition), y: CT 图像 (target)
            # 在扩散训练中，我们用 CT 作为 x_0，CBCT 作为条件
            loss = self.diffusion_trainer.forward(x_0=x, context=c)
            loss = tf.reduce_mean(loss)
        
        # 计算梯度并更新模型
        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        
        return loss
    
    def diffusion_train(self):
        """专门为扩散模型定制的训练方法"""
        import h5py
        import numpy as np
        import tensorflow as tf
        
        print("开始扩散模型训练...")
        
        # 编译模型
        self.compile_it()
        
        for epoch in range(self.counter, self.epoch):
            print(f"\nEpoch {epoch + 1}/{self.epoch}")
            
            epoch_losses = []
            
            # 随机选择训练文件
            np.random.shuffle(self.training_paths)
            
            for i, file_path in enumerate(self.training_paths[:self.steps_per_epoch]):
                try:
                    # 加载数据
                    with h5py.File(file_path, 'r') as f_h5:
                        cbct = np.asarray(f_h5['input_images'], dtype=np.float32)  # CBCT
                        ct = np.asarray(f_h5['output_images'], dtype=np.float32)   # CT
                    
                    # 添加批次和通道维度
                    cbct = tf.expand_dims(tf.expand_dims(cbct, 0), -1)  # [1, H, W, D, 1]
                    ct = tf.expand_dims(tf.expand_dims(ct, 0), -1)      # [1, H, W, D, 1]
                    
                    # # 数据预处理 - 归一化到 [-1, 1]
                    # cbct = (cbct - tf.reduce_min(cbct)) / (tf.reduce_max(cbct) - tf.reduce_min(cbct))
                    # cbct = cbct * 2.0 - 1.0
                    
                    # ct = (ct - tf.reduce_min(ct)) / (tf.reduce_max(ct) - tf.reduce_min(ct))
                    # ct = ct * 2.0 - 1.0
                    
                    # 执行训练步骤
                    loss = self.diffusion_train_step(x=ct, c=cbct)
                    epoch_losses.append(float(loss))
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Step {i + 1}/{self.steps_per_epoch}, Loss: {loss:.4f}")
                
                except Exception as e:
                    print(f"  Error processing {file_path}: {str(e)}")
                    continue
            
            # 计算平均损失
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.save_period == 0:
                self.save(epoch + 1)
                print(f"Saved checkpoint at epoch {epoch + 1}")
        
        print("训练完成!")
