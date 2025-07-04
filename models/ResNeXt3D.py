import os, re, glob
import numpy as np
import h5py, json
import tensorflow as tf

from classification_3d import BaseModel
from utils_resnext3d import ResNeXt101

DEFAULT_MODEL_CONFIG = {
    'epoch': 1000,  # total_iterations = iters_per_epoch * epoch
    'batch_size': 8,
    'iters_per_epoch': 800,  # each epoch only runs 800 iters
    # mutli-class: using Categorical*; multi-label: using Binary*
    'loss_type': 'CategoricalCrossentropy',  # BinaryCrossentropy, BinaryFocalCrossentropy, CategoricalCrossentropy
    'norm_config': {'norm': False, 'norm_channels': 'all_channels'},
}

class ResNeXt3D(BaseModel):
    def __init__(self, checkpoint_dir, log_dir, training_paths, train_classes, im_size, num_threads=6, class_weights=None,
                 input_channels=1, apply_segmentation_mask=False, mirror_config=None, model_config=None, **kwargs):
        
        super(ResNeXt3D, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = self.oversample_paths(training_paths, class_weights=class_weights)
        
        self.train_classes = train_classes
        self.apply_segmentation_mask = apply_segmentation_mask
        self.mirror_config = mirror_config
        
        if model_config is None:
            model_config = DEFAULT_MODEL_CONFIG
        else:
            config = DEFAULT_MODEL_CONFIG
            for key in model_config.keys():
                config[key] = model_config[key]
            model_config = config
        
        self.batch_size = int(model_config['batch_size'])
        self.epoch = int(model_config['epoch'])
        self.norm_config = model_config['norm_config']
        
        self.input_channels = input_channels
        
        self.nclass = len(self.train_classes) + 1
        
        self.num_threads = num_threads
        
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        
        self.loss_type = model_config['loss_type']
        
        self.im_size = im_size
        
        self.counter = 0
        self._loaded, self.counter = self.load()
        if not self._loaded:
            print('Initializing...')
            self.networks = self.build_networks()
            print('Complete initialization.')
    
    def compile_it(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
        if self.loss_type == 'BinaryFocalCrossentropy':
            self.loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=False,
                alpha=0.25,
                gamma=2.0,
                from_logits=True,
            )
        elif self.loss_type == 'BinaryCrossentropy':
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif self.loss_type == 'CategoricalCrossentropy':
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.compile(optimizer=self.opt, loss=self.loss_fn)
        return
    
    def build_networks(self):
        input_shape = self.im_size + [self.input_channels,]
        networks = ResNeXt101(include_top=True, input_shape=input_shape, weights=None, classes=self.nclass)
        return networks
    
    @property
    def model_dir(self):
        return 'ResNeXt3D'