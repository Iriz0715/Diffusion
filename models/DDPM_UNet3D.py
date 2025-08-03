import os, re, logging
import time
import h5py
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')  # Adjust the path as needed to import from the parent directory
from DDPM_base_model_3d import DDPMBaseModel3D
from networks_copy import *
from networks import  *
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
import importlib


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
    'residual': False,    ## 改用time_embeddng_resnet
    'activation': 'sigmoid',

    # Diffusion model parameters
    'temb_residual': True,  # 是否使用时间嵌入
    'max_timesteps': 1000,  # 最大时间步数 T
    'dropout': 0.1,  # dropout概率
    'beta_1': 1e-4,   # 起始 beta 值
    'beta_T': 0.02,   # 结束 beta 值
    'infer_T': 1000,
    'squeue': None
}

class DDPM_UNet3D(DDPMBaseModel3D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, im_size, num_threads, 
                 input_channels=1, output_channels=1, model_config=None, resume=True, **kwargs):
        super(DDPM_UNet3D, self).__init__()
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
        self.enlarged_im_size = (int(im_size[0] * 1.1875), int(im_size[1] * 1.1875), int(im_size[2] * 1.1875))  # 加一个维度

        self.strides_list = get_strides_list(self.layer_number, self.im_size)
        
        self.counter = 0
        self.resume = resume  # 是否恢复训练

        # Use natural image model config
        self.is_natural_image = bool(model_config['is_natural_image'])
        
        # experimental
        self.residual = model_config['residual']
        self.attention = model_config['attention']
        
        self.activation = model_config['activation']


        #### Diffusion
        self.temb_residual = model_config['temb_residual']  # 是否使用时间嵌入
        self.max_timesteps = model_config['max_timesteps'] # 最大时间步数 T
        self.dropout = model_config['dropout']  # dropout概率
        self.beta_1 = model_config['beta_1']
        self.beta_T = model_config['beta_T']
        self.squeue = model_config['squeue']
        self.infer_T = model_config['infer_T']
        # self.is_diffusion = model_config.get('is_diffusion', False)  # 是否使用扩散模型
        
        # 根据resume参数决定是否加载检查点
        if resume==True:
            print("[INFO] Resume mode: attempting to load checkpoint...")
            _loaded, self.counter = self.load()
        else:
            print("[INFO] New training mode: starting from scratch...")
            _loaded = False
            self.counter = 0
        if not _loaded:
            # 如果没有加载成功，创建新的UNet
            self.unet = self.build_diffusion_model()
            self.unet.summary()
            # self.unet.summary(expand_nested=True)

        # 如果是DM，创建 trainer
        # if self.is_diffusion:
        print(f'T={self.max_timesteps}, infer_T={self.infer_T}')
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.unet,      # resume=True 时，unet 已经包含了训练好的权重；resume=False 时，unet 是新初始化的模型
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
    

    ######################## 编译模型 ########################
    def compile_it(self):
        """编译扩散模型"""
        self.opt = self.get_optimizer()
        self.optimizer = self.opt
    
        # 对于扩散模型，我们主要关心噪声预测的准确性
        def diffusion_loss(y_true, output_loss):      ## 这里不理解编译对象是unet(output:noise_pred)还是diffusion_trainer(output:loss)
            """扩散模型的占位符损失函数"""
            # 实际损失在 diffusion_trainer.forward 中计算
            return tf.reduce_mean(tf.square(output_loss))
        
        
        # 编译UNet模型
        self.unet.compile(
            optimizer=self.optimizer,
            loss=diffusion_loss
        )
        
        print(f"扩散模型编译完成 - 优化器: {self.optimizer_type}, 学习率: {self.initial_lr}")
    
    @tf.function
    def loss_fn(self, labels, probs, mask=None):
        # labels 是目标 CT 图像，probs 是条件 CBCT 图像（在这种情况下）
        # 实际的损失计算在 diffusion_trainer.forward 中进行，l2 loss
        return tf.reduce_mean(probs)  # 占位符，model输出即loss

    ####################### 构建模型 ########################
    def build_diffusion_model(self):
        unet = build_diffusion_unet(        ## networks_copy.py 简单unet
        # unet = build_unet(      # networks.py 原unet
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
            # num_res_blocks=2,  # 每层的残差块数量
            ######## Diffusion
            use_temb_encoder=self.temb_residual,    # 是否使用时间嵌入
            max_timesteps=self.max_timesteps,          # 最大时间步数 T
            dropout=self.dropout,                 # dropout概率
        )


        input_shape = [None for _ in self.im_size] + [self.input_channels,]
        # 扩散模型需要两个输入：[图像, 时间步]
        img_input = layers.Input(shape=input_shape, name='image_input')
        time_input = layers.Input(shape=[], dtype=tf.int32, name='time_input')
            
        # 将时间步传递给UNet（如果UNet支持）
        x = unet([img_input, time_input])
        outputs = x  # 无激活函数，直接输出噪声预测
        
        model = Model(inputs=[img_input, time_input], outputs=outputs)
        return model

    @property
    def model_dir(self):
        return 'transform'
    
