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
from DDPM_base_model_2d import DDPMBaseModel2D
from networks import *
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
import importlib
from models.dataset_tf import create_dataset


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
            self.unet = self.build_model()
            self.unet.summary()

        # 如果是DM，创建 trainer
        # if self.is_diffusion:
        print(f'T={self.max_timesteps}')
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.unet,      # resume=True 时，unet 已经包含了训练好的权重；resume=False 时，unet 是新初始化的模型
            beta_1=self.beta_1,
            beta_T=self.beta_T,
            T=self.max_timesteps
        )
        print("Diffusion trainer created successfully!!")
        
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
      
    ######################### 构建模型 ########################
    def build_model(self):
        unet = build_unet(        ## networks.py
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
    






    # ############ 20250702
    # def diffusion_train(self, run_validation=False, validation_paths=None, resume=None, **kwargs):
    #     """
    #     完善的扩散模型训练方法，参考 DDPM_base_model_2d.train()
        
    #     Args:
    #         run_validation: 是否运行验证
    #         validation_paths: 验证集文件路径列表
    #         resume: (已弃用) 现在通过初始化时的resume参数控制
    #     """
    #     if resume is not None:
    #         print("[WARNING] The 'resume' parameter in diffusion_train() is deprecated. "
    #               "Please use the 'resume' parameter in model initialization instead.")
        
    #     print("开始扩散模型训练...")
        
    #     # 1. 编译模型
    #     self.compile_it()
        
    #     # 2. 设置日志目录
    #     log_dir = os.path.join(self.log_dir, self.model_dir)
    #     os.makedirs(log_dir, exist_ok=True)
        
    #     # # 3. 创建 TensorBoard 回调
    #     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=0)
        
    #     # 清理旧的 TensorBoard 文件（可选）
    #     if not resume:  # 只在新训练时清理
    #         import glob
    #         old_events = glob.glob(os.path.join(log_dir, '**/events.out.tfevents.*'), recursive=True)
    #         for old_file in old_events:
    #             try:
    #                 os.remove(old_file)
    #                 print(f"Removed old TensorBoard file: {old_file}")
    #             except Exception as e:
    #                 print(f"Failed to remove {old_file}: {e}")
    
    #     # 4. 创建验证配置
    #     validation_config = None
    #     if run_validation and validation_paths:
    #         validation_config = {
    #             'validation_paths': validation_paths,
    #             'validation_fn': self.validate_diffusion,
    #             'log_dir': log_dir,
    #         }
        
    #     # 5. 创建保存配置
    #     saver_config = {
    #         'period': self.save_period,
    #         'save_fn': self.save,
    #         'log_dir': log_dir,
    #     }

    #     # 使用 DiffusionModelSaver（完全模拟 ModelSaver 的行为）
    #     saver_callback = self.DiffusionModelSaver(saver_config=saver_config, validation_config=validation_config)
        
    #     # 6. 创建数据集
    #     train_dataset = create_dataset(self.training_paths, batch_size=self.batch_size, shuffle=True, return_filenames=False)
    #     total_steps = self.steps_per_epoch
        
    #     # 7. 准备训练
    #     num_samples = len(self.training_paths)
    #     if num_samples == 0:
    #         print('No training data')
    #         return
        
    #     # 使用当前的计数器值作为训练起点
    #     print(f'Steps per epoch: {total_steps}')
    #     print(f'Training from epoch {self.counter} to {self.epoch}')

    #     idx_list = np.arange(num_samples)
    #     total_iters = self.steps_per_epoch * (self.epoch - self.counter)
    #     ## data_generator = self.data_generator(idx_list, total_iters)    # 自定义了dataloader，直接使用 create_dataset
        
    #     if total_iters > 0:
    #         print('Running on complete dataset with total training samples:', num_samples)
            
    #         # 创建回调列表
    #         # callbacks = [tensorboard_callback, saver_callback]
    #         callbacks = [saver_callback]  # 只使用保存回调
            
    #         # 模拟 fit() 的回调生命周期
    #         # 1. on_train_begin
    #         for callback in callbacks:
    #             callback.on_train_begin()
            
    #         # 训练循环
    #         for epoch in range(self.counter, self.epoch):
    #             print(f"\nEpoch {epoch + 1}/{self.epoch}")
                
    #             # 2. on_epoch_begin
    #             for callback in callbacks:
    #                 callback.on_epoch_begin(epoch)
                
    #             # 重置指标
    #             train_loss_metric = tf.keras.metrics.Mean()
                
    #             # 进度条
    #             progress_bar = tf.keras.utils.Progbar(
    #                 target=total_steps, 
    #                 stateful_metrics=['loss', 'lr']
    #             )
                
    #             # 训练一个 epoch
    #             step_count = 0
    #             for step, (cbct, ct) in enumerate(train_dataset.take(total_steps)):
                    
    #                 # 3. on_batch_begin
    #                 for callback in callbacks:
    #                     callback.on_batch_begin(step)
                    
    #                 # 3d to 2d
    #                 cbct = cbct[:,cbct.shape[1]//2,...]   # [bs, im_size, im_size, channels]
    #                 ct = ct[:,ct.shape[1]//2,...]

    #                 # 执行训练步骤
    #                 step_loss = self._diffusion_train_step(cbct, ct)
                    
    #                 # 更新指标
    #                 train_loss_metric.update_state(step_loss)
                    
    #                 # 获取当前学习率
    #                 if hasattr(self.optimizer.learning_rate, '__call__'):
    #                     current_step = epoch * total_steps + step
    #                     current_lr = float(self.optimizer.learning_rate(current_step))
    #                 else:
    #                     current_lr = float(self.optimizer.learning_rate)
                    
    #                 # 准备批次日志
    #                 batch_logs = {
    #                     'loss': float(step_loss),
    #                     'lr': current_lr
    #                 }
                    
    #                 # 4. on_batch_end
    #                 for callback in callbacks:
    #                     callback.on_batch_end(step, batch_logs)
                    
    #                 # 更新进度条
    #                 progress_bar.update(
    #                     step + 1, 
    #                     values=[
    #                         ('loss', train_loss_metric.result().numpy()),
    #                         ('lr', current_lr)
    #                     ]
    #                 )
                    
    #                 step_count += 1
                
    #             # Epoch 结束处理
    #             epoch_loss = train_loss_metric.result()
                
    #             # 准备 epoch 日志
    #             epoch_logs = {
    #                 'loss': float(epoch_loss),
    #                 'lr': current_lr
    #             }
                
    #             print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.6f}, LR: {current_lr:.8f}")
                
    #             # 5. on_epoch_end
    #             for callback in callbacks:
    #                 callback.on_epoch_end(epoch, epoch_logs)

    #             # 重置度量
    #             train_loss_metric.reset_states()
            
    #         # 6. on_train_end
    #         for callback in callbacks:
    #             callback.on_train_end()

    #         self.save(self.epoch)  # 保存最后一个 epoch 的检查点
        
    #     print("训练完成!")
    
    # @tf.function
    # def _diffusion_train_step(self, cbct, ct):
    #     """单个训练步骤"""
    #     with tf.GradientTape() as tape:
    #         loss = self.diffusion_trainer.forward(x_0=ct, context=cbct)
    #         loss = tf.reduce_mean(loss)
        
    #     # 计算梯度并更新模型
    #     gradients = tape.gradient(loss, self.unet.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        
    #     return loss
    

    # ############# 还没修改
    # def validate_diffusion(self, validation_paths):
    #     """
    #     扩散模型验证方法
    #     """
    #     if not validation_paths:
    #         return {'mae': 0.0, 'mse': 0.0}
        
    #     print("Running validation...")
    #     val_losses = []
        
    #     for val_path in validation_paths[:min(10, len(validation_paths))]:  # 限制验证样本数量
    #         try:
    #             # 生成验证样本
    #             with h5py.File(val_path, 'r') as f_h5:
    #                 cbct = np.asarray(f_h5['input_images'], dtype=np.float32)
    #                 ct = np.asarray(f_h5['output_images'], dtype=np.float32)
                
    #             # 添加批次和通道维度
    #             cbct = tf.expand_dims(tf.expand_dims(cbct, 0), -1)
    #             ct = tf.expand_dims(tf.expand_dims(ct, 0), -1)
                
    #             # 计算验证损失
    #             val_loss = self.diffusion_trainer.forward(x_0=ct, context=cbct)
    #             val_losses.append(float(tf.reduce_mean(val_loss)))
                
    #         except Exception as e:
    #             print(f"Validation error for {val_path}: {str(e)}")
    #             continue
        
    #     avg_val_loss = np.mean(val_losses) if val_losses else 0.0
    #     print(f"Validation loss: {avg_val_loss:.6f}")
        
    #     return {'loss': avg_val_loss, 'mse': avg_val_loss}
  

    # class SimplifiedModelSaver(tf.keras.callbacks.Callback):
    #     """
    #     简化的模型保存回调，只负责保存和验证，完全模拟 DDPM_base_model_2d.ModelSaver 的行为
    #     """
    #     def __init__(self, saver_config, validation_config=None):
    #         self.counter = 0
    #         self.period = saver_config['period']
    #         self.save = saver_config['save_fn']
    #         self.validation_config = validation_config
    #         self.logs = {}
            
    #         print(f"SimplifiedModelSaver initialized - save period: {self.period}")
        
    #     def on_train_begin(self, logs=None):
    #         """训练开始时的处理"""
    #         print(f"Starting training with save period of {self.period} epochs")
        
    #     def on_batch_end(self, batch, logs=None):
    #         """批次结束时记录日志（和标准 ModelSaver 一致）"""
    #         if logs is None:
    #             logs = {}
            
    #         if len(self.logs) == 0:
    #             for key in logs.keys():
    #                 self.logs[key] = []
            
    #         for key in logs.keys():
    #             if key in self.logs:
    #                 self.logs[key].append(logs[key])
        
    #     def on_epoch_end(self, epoch, logs=None):
    #         """Epoch 结束时的处理（完全模拟标准 ModelSaver）"""
    #         if logs is None:
    #             logs = {}
                
    #         self.counter += 1
            
    #         # 重置批次日志
    #         self.logs = {}
            
    #         # 保存和验证（和标准 ModelSaver 完全一致）
    #         if self.counter % self.period == 0:
    #             print(f"\nSaving checkpoint at epoch {epoch + 1}...")
    #             self.save(epoch + 1)
                
    #             # 运行验证（如果配置了）
    #             if self.validation_config is not None:
    #                 print("Running validation...")
    #                 try:
    #                     val_results = self.validation_config['validation_fn'](
    #                         self.validation_config['validation_paths']
    #                     )
    #                     print(f"Validation results: {val_results}")
    #                 except Exception as e:
    #                     print(f"Validation failed: {str(e)}")
                
    #             print(f"Checkpoint saved at epoch {epoch + 1}")
        
    #     def on_train_end(self, logs=None):
    #         """训练结束时的处理"""
    #         print("Training completed - SimplifiedModelSaver finished")

    # class DiffusionModelSaver(DDPMBaseModel2D.ModelSaver):
    #     """
    #     扩散模型专用的模型保存回调，继承基类 ModelSaver 并扩展手动训练循环所需的功能
    #     """
    #     def __init__(self, saver_config, validation_config=None, custom_log_file=None):
    #         # 调用父类初始化
    #         super().__init__(saver_config, validation_config, custom_log_file)
    #         print(f"DiffusionModelSaver initialized - save period: {self.period} epochs")
        
    #     def on_train_begin(self, logs=None):
    #         """训练开始时的处理"""
    #         print(f"Starting diffusion training with save period of {self.period} epochs")
        
    #     def on_epoch_begin(self, epoch, logs=None):
    #         """Epoch 开始时的处理"""
    #         pass
        
    #     def on_batch_begin(self, batch, logs=None):
    #         """批次开始时的处理"""
    #         pass
        
    #     def on_epoch_end(self, epoch, logs=None):
    #         """重写父类方法，增加更好的日志处理和资源管理"""
    #         if logs is None:
    #             logs = {}
                
    #         self.counter += 1
            
    #         # 计算 epoch 平均日志并记录到 TensorBoard
    #         with self.train_writer.as_default():
    #             record_logs = {}
    #             for key in self.logs.keys():
    #                 if 'loss' in key:
    #                     # 对于 loss 指标，过滤掉 0 值
    #                     non_zero_values = np.array(self.logs[key])[np.nonzero(self.logs[key])]
    #                     if len(non_zero_values) > 0:
    #                         epoch_avg = non_zero_values.mean()
    #                     else:
    #                         epoch_avg = np.nanmean(self.logs[key])
    #                 else:
    #                     epoch_avg = np.nanmean(self.logs[key])
                    
    #                 tf.summary.scalar(f'epoch_{key}', epoch_avg, step=epoch)
    #                 record_logs[key] = epoch_avg
                
    #             self.train_writer.flush()
                
    #             # 改进的自定义日志文件记录（兼容扩散模型可能没有 dice 的情况）
    #             if self.custom_log_file is not None:
    #                 import csv
    #                 import datetime
    #                 with open(self.custom_log_file, 'a') as f:
    #                     writer = csv.writer(f, delimiter=';')
    #                     loss_val = record_logs.get('loss', 0.0)
    #                     # 对于扩散模型，可能没有 dice 指标，使用可用指标或 loss
    #                     dice_val = record_logs.get('dice', record_logs.get('mse', loss_val))  
    #                     writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'),
    #                                    f'epoch: {epoch + 1}, metric: {dice_val:.4f}, loss: {loss_val:.4f}'])
            
    #         # 重置批次日志
    #         self.logs = {}
            
    #         # 保存和验证
    #         if self.counter % self.period == 0:
    #             print(f"\nSaving checkpoint at epoch {epoch + 1}...")
    #             self.save(epoch + 1, max_to_keep=3)  # 保留3个最新模型
                
    #             # 运行验证（如果配置了）
    #             if self.validation_config is not None:
    #                 print("Running validation...")
    #                 try:
    #                     val_scores = self.validation_fn(self.validation_paths)    
                        
    #                     # 记录验证结果到 TensorBoard
    #                     if hasattr(self, 'test_writer') and self.test_writer is not None:
    #                         with self.test_writer.as_default():
    #                             for metric in val_scores:
    #                                 tf.summary.scalar(metric, val_scores[metric], step=epoch + 1)
    #                             self.test_writer.flush()
                        
    #                     print(f"Validation results: {val_scores}")
    #                 except Exception as e:
    #                     print(f"Validation failed: {str(e)}")
                
    #             print(f"Checkpoint saved at epoch {epoch + 1}")
        
    #     def on_train_end(self, logs=None):
    #         """训练结束时的处理 - 增加资源清理"""
    #         print("Training completed - DiffusionModelSaver finished")
    #         # 关闭 TensorBoard writers
    #         if hasattr(self, 'train_writer'):
    #             self.train_writer.close()
    #         if hasattr(self, 'test_writer') and self.test_writer is not None:
    #             self.test_writer.close()

    # # 为了兼容性，保留原方法名
    # def diffusion_train_simple(self):
    #     """简化版训练方法（原版本）"""
    #     return self.diffusion_train(run_validation=False)



    # ################# 20250702 test: reverse sampling
    # def diffusion_test(self, testing_paths, output_path, squeue=None, **kwargs):
    #     # 创建测试数据集
    #     test_dataset = create_dataset(
    #         data_file=testing_paths,  # 替换为实际的测试数据路径
    #         batch_size=1,  # 测试时通常使用 batch_size 1
    #         shuffle=False,
    #         return_filenames=True  # 测试时需要文件名
    #         )
        
    #     # 加载模型
    #     _loaded, self.counter = self.load()
    #     if _loaded:
    #       self.diffusion_sampler = GaussianDiffusionSampler(
    #           model=self.unet,
    #           beta_1=self.beta_1,
    #           beta_T=self.beta_T,
    #           T=self.max_timesteps,
    #           squeue=squeue
    #       )
    #     else:
    #         raise ValueError("model load failed.")
        
        
    #     # reverse sampling
    #     for i,(cbct, ct, file_name) in enumerate(test_dataset):
    #         print(f"Processing sample {i + 1}/{len(test_dataset)}")
    #         # 3D to 2D
    #         cbct = cbct[:, cbct.shape[1] // 2, ...]
    #         ct = ct[:, ct.shape[1] // 2, ...]
    #         current_save_path = os.path.join(output_path, f'sample_{file_name[0].numpy().decode("utf-8")}')   # batch=1就是一个图像一个文件夹
    #         self._diffusion_sample_step(context=cbct, squeue=squeue, save_path=current_save_path)
    #         print(f"Sample {i + 1} saved !!")

    #     print(f"All samples processed and saved to {output_path}")


    # ############# 扩散采样步骤
    # def _diffusion_sample_step(self, context, squeue=None, save_path=None):
    #     """ 单个扩散采样步骤
    #     Args:
    #         context: 条件输入(CBCT图像)
    #         squeue: 保存中间结果的间隔步数
    #         save_dir: 保存目录
    #     """
    #     # 生成样本
    #     x_T = tf.random.normal(tf.shape(context))   # [batch_size, H, W, channels]
    #     generated = self.diffusion_sampler.reverse(x_T, context=context)
    #     print(f"Generated shape: {generated.shape}")
        
    #     # 创建保存目录
    #     os.makedirs(save_path, exist_ok=True)
        
    #     # 保存结果
    #     if squeue is not None:
    #         # 保存中间结果
    #         for step in range(generated.shape[-1]):   # [batch_size, H, W, D, save_steps]
    #             plt.figure(figsize=(6, 6))
    #             plt.imshow(generated[0, ..., step], cmap='gray')
    #             plt.title(f'x_{self.max_timesteps-(step+1)*squeue}')
    #             plt.axis('off')
    #             plt.savefig(os.path.join(save_path, f'step_{self.max_timesteps-(step+1)*squeue:04d}.png'))
    #             plt.close()
    #     else:
    #         # 只保存最终结果
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(generated[0, ..., -1], cmap='gray')
    #         plt.title(f'x_{generated.shape[-1]}')
    #         plt.axis('off')
    #         plt.savefig(os.path.join(save_path, f'final_x0.png'))
    #         plt.close()


      







if __name__ == "__main__":
  test_dataset = create_dataset(
      data_file='/home/jiayizhang/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/data_training/test', 
      batch_size=1,  # 测试时通常使用 batch_size 1
      shuffle=False,
      return_filenames=True
      )
  print('ok')