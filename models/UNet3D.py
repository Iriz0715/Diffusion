import os, re, glob, logging
import numpy as np
import h5py, json
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization

from base_model import BaseModel
from networks import *

DEFAULT_MODEL_CONFIG = {
    'features_root': 32,
    'conv_size': 3,
    'deconv_size': 2,
    'layers': 5,
    'max_filters': 320,
    'dilation': False,
    'test_nstride': [2, 2, 2],
    'loss_type': 'both',
    # loss head: list of dicts
    # Only rois are required.
    # Default head activation is softmax. Default head weight is 1.
    # e.g. [{'rois': [161, 162, 163], 'activation': 'softmax', 'weight': 2}, 
    #       {'rois': [417, 420, 421], 'activation': 'sigmoid', 'weight': 1}, ...]
    'loss_head': None,
    'batch_size': 1,
    'deep_supervision': True,
    'iters_per_epoch': 100,  #each epoch only runs 100 iters
    'epoch': 3000,
    'norm_config': {'norm': True, 'norm_channels':  'all_channels'},
    'aug_config': {}, # additional augmentation config
    'optimizer_type': 'SGD', # currently only supporting 'SGD' and 'Adam'
    # only apply to SGD
    # momentum 0.99 is good for most situations
    # if the training/testing dice is 0, we could lower down the momentum to 0.95
    # if the label is very heterogenous, probably lower the momentum to 0.90
    'sgd_momentum': 0.99,
    # increase this ratio when there are very small structures
    'fg_sampling_ratio': 0.5,
    # add the missing rois in training; method: union/intersection
    'enhanced_rois': [], 
    'enhanced_method': 'intersection',
    # in case some rois are small and hard to sample, only valid with enhanced_rois
    'enhanced_sampling': False,
    # roi specific augmentation
    'aug_rois_config': {'rois': [], 'contrast_scale': (0.5, 1.5), 'p_contrast': 0.5,},
    ## currently only supporting float32
    ## float16 speed-up requires tensorflow>=2.4.0 and cudnn>=8
    'policy': 'float32',  # ['float16', 'mixed_float16', 'float32']
    
    # debug config
    'debug_roi_dice': True,  # log dices for each roi in tensorboard
    'debug_loss': False,  # log dice_loss and ce_loss in tensorboard
    
    # experimental
    'attention': False,
    'residual': True,
}

class UNet3D(BaseModel):
    def __init__(self, checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads=6,
                 input_channels=1, channel_dropout=False, class_weights=None, left_right_swap_config=None, mirror_config=None, 
                 model_config=None):
        
        super(UNet3D, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self._training_paths_filtered = False
        
        self.group = group
        self.rois = rois
        
        self.mirror_config = mirror_config
        # overwrite left_right_swap when mirroring in all directions
        if self.mirror_config is not None and self.mirror_config['mirror_all_dimensions']:
            self.left_right_swap_config = None
        else:
            self.left_right_swap_config = left_right_swap_config
        
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
        
        self.batch_size = int(model_config['batch_size'])
        self.epoch = int(model_config['epoch'])
        self.features_root = int(model_config['features_root'])
        self.loss_type = model_config['loss_type']
        loss_head = model_config.get('loss_head')
        # Pre-process the input loss head config
        self.loss_head = self.parse_loss_head(loss_head, self.rois)
        print('(Internal) Loss head config:', self.loss_head)
        
        self.norm_config = model_config['norm_config']
        self.aug_config = model_config['aug_config']

        self.optimizer_type = model_config['optimizer_type']
        self.sgd_momentum = model_config['sgd_momentum']
        self.fg_sampling_ratio = model_config['fg_sampling_ratio']
        self.enhanced_rois = model_config['enhanced_rois']
        self.enhanced_method = model_config['enhanced_method']
        self.enhanced_sampling = model_config['enhanced_sampling']
        self.aug_rois_config = model_config['aug_rois_config']
        
        self.debug_loss = model_config['debug_loss']
        self.debug_roi_dice = model_config['debug_roi_dice']
        
        self.input_channels = input_channels
        self.channel_dropout = channel_dropout
        
        if self.loss_head:
            total_channels = 0
            for head in self.loss_head:
                total_channels += len(head['rois'])
            num_bg_channels = total_channels - len(self.rois)
            self.nclass = len(self.rois) + num_bg_channels
        else:
            self.nclass = len(self.rois) + 1
        
        self.conv_size = int(model_config['conv_size'])
        self.deconv_size = int(model_config['deconv_size'])
        self.layer_number = int(model_config['layers'])
        self.max_filters = int(model_config['max_filters'])
        self.dilation = model_config['dilation']
        self.test_nstride = model_config['test_nstride']
        self.deep_supervision = model_config['deep_supervision']
        self.num_threads = num_threads
        self.class_weights = class_weights
        
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        
        self.counter = 0
        self.need_retrain = False
        
        self.im_size = im_size

        # A bit arbitrary to enlarge by 1.1875
        self.enlarged_im_size = (int(self.im_size[0] * 1.1875), int(self.im_size[1] * 1.1875), int(self.im_size[2] * 1.1875))
        
        self.strides_list = self.get_strides_list(self.layer_number, self.im_size)

        if self.deep_supervision:
            self.ds_loss_weights = [0.] + [2**i/(2**(self.layer_number - 1) - 1) for i in range(self.layer_number - 1)]
            self.deep_supervision_scales = [
                np.prod(self.strides_list[:i], axis=0).tolist() for i in range(self.layer_number)
            ][::-1]
            self.downscaling_model = self.build_auxiliary_ds_model()
        
        # experimental
        self.residual = model_config['residual']
        self.attention = model_config['attention']

        # given points and segment patch, only used in validate/test
        self.guided = False

        self.inference_model = None
        self._loaded, self.counter = self.load()
        if not self._loaded:
            print('Initializing...')
            self.unet = self.build_unet()
            print('Complete initialization.')

        # Log variables
        vars_log_path = os.path.join(self.log_dir, self.model_dir, 'vars.txt')
        os.makedirs(os.path.dirname(vars_log_path), exist_ok=True)
        self.vars_log_path = vars_log_path
        self_vars = {k: vars(self).get(k) for k in dir(self)
                     if not k.startswith('_') and vars(self).get(k) is not None}
        logging.basicConfig(filename=vars_log_path, level=logging.INFO)
        logging.info(self_vars)

    
    def build_auxiliary_ds_model(self):
        downscaling_model = build_auxiliary_ds_model(
            im_size=self.im_size, 
            nclass=self.nclass, 
            layer_number=self.layer_number, 
            deep_supervision_scales=self.deep_supervision_scales
        )
        return downscaling_model
    
    def build_unet(self):
        unet = build_unet(
            im_size=self.im_size, 
            nclass=self.nclass,
            strides_list=self.strides_list,
            input_channels=self.input_channels, 
            features_root=self.features_root,
            conv_size=self.conv_size, 
            deconv_size=self.deconv_size, 
            layer_number=self.layer_number, 
            max_filters=self.max_filters, 
            dilation=self.dilation, 
            attention=self.attention,
            deep_supervision=self.deep_supervision,
            use_residual_encoder=self.residual,
            freeze_input_shape=False,
        )
        return unet
    
    def compile_it(self):
        self.opt = self.get_optimizer()
        # run_eagerly=True for debug mode, maybe it is related memory leakage.
        self.compile(optimizer=self.opt, loss=self.loss_fn, run_eagerly=False)
        return
    
    def get_optimizer(self):
        starter_learning_rate = 0.01
        end_learning_rate = 1e-6
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
        if self.optimizer_type == 'SGD':
            # momentum 0.99 is good for most situations
            # if the training/testing dice is 0, we could lower down the momentum to 0.95
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=self.sgd_momentum, clipnorm=12, nesterov=True)
        elif self.optimizer_type == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=12)
 
        return opt
    
    @tf.function        
    def loss_fn(self, input_logits, input_labels, input_loss_weights):
        if self.loss_head is None:
            loss_head = [self.loss_head]
        else:
            loss_head = self.loss_head

        final_loss = 0.0
        final_roi_dices = [-1.0 for _ in range(len(self.rois))]
        final_dice = 0.0
        final_cross_entropy_loss = 0.0
        final_dice_loss = 0.0
        for head in loss_head:
            if head is None:
                logits, labels, loss_weights = input_logits, input_labels, input_loss_weights
                head_weight = 1.0
                nclass = self.nclass
                class_weights = self.class_weights
                loss_type = self.loss_type
                ce_loss_fn = tf.nn.softmax_cross_entropy_with_logits
            else:
                logits = tf.gather(input_logits, indices=head['rois'], axis=-1)
                labels = tf.gather(input_labels, indices=head['rois'], axis=-1)
                loss_weights = tf.gather(input_loss_weights, indices=head['rois'], axis=-1)
                head_weight = head['weight']
                nclass = len(head['rois'])
                class_weights = None # TODO
                loss_type = 'both' # TODO
                if head['activation'] == 'softmax':
                    ce_loss_fn = tf.nn.softmax_cross_entropy_with_logits
                elif head['activation'] == 'sigmoid': # TODO
                    ce_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
           
            if self.policy != 'float32':
                logits = tf.cast(logits, tf.float32)
                labels = tf.cast(labels, tf.float32)
                loss_weights = tf.cast(loss_weights, tf.float32)
            
            flat_logits = tf.reshape(logits, [-1, nclass])
            flat_labels = tf.reshape(labels, [-1, nclass])
            flat_weights = tf.reshape(loss_weights, [-1, nclass])
    
            weights_logits = tf.multiply(flat_logits, flat_weights)
            weights_labels = tf.multiply(flat_labels, flat_weights)

            # Cross-entropy loss
            if class_weights is not None:
                class_weights = tf.constant(np.asarray(class_weights, dtype=np.float32))
                weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
                loss_map = ce_loss_fn(logits=weights_logits, labels=weights_labels)
                
                # First multiply by class weights, then by loss weights due to missing of rois
                weighted_loss = tf.multiply(loss_map, weight_map)
            else:
                loss_map = ce_loss_fn(logits=weights_logits, labels=weights_labels)
                weighted_loss = loss_map
            
            cross_entropy_loss = tf.reduce_mean(weighted_loss)
                
            if 'focal' in loss_type:
                # Focal Loss (categorical, multi-class)
                beta = 1.0 * nclass  # scaling factor for categorical focal loss
                gamma = 2.0  # focusing factor
    
                ce_loss = weighted_loss  # use class-weighted ce loss calculated as above (replace alpha)
                pt = tf.nn.softmax(weights_logits)
                focal_loss_map = beta * tf.pow(1.0 - pt, gamma) * (weights_labels * ce_loss[:, None])
                focal_loss = tf.reduce_mean(focal_loss_map)
    
            # Dice loss
            probs = tf.nn.softmax(logits)
            predictions = tf.argmax(probs, 4)
            smooth = 1.0
            if self.debug_roi_dice:
                dice_value = []
            else:
                dice_value = 0.
            dice_loss = 0.
            n_rois = 0
            weighted_n_rois = 0
            
            for i in list(range(1, nclass)):
                if class_weights is not None:
                    weights = class_weights[i]
                else:
                    weights = 1.0
                    
                slice_weights = loss_weights[:, :, :, :, i]
                slice_prob = probs[:, :, :, :, i]
                slice_pred = tf.cast(tf.equal(predictions, i), tf.float32)
                slice_label = labels[:, :, :, :, i]
                intersection_prob = smooth + tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2, 3])
                intersection_pred = smooth + tf.reduce_sum(tf.multiply(slice_pred, slice_label), axis=[1, 2, 3])
                union_prob = 2.0 * smooth + tf.reduce_sum(slice_prob, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
                union_pred = 2.0 * smooth + tf.reduce_sum(slice_pred, axis=[1, 2, 3]) + tf.reduce_sum(slice_label, axis=[1, 2, 3])
                
                # Multiply by loss weights
                roi_exist = tf.reduce_mean(slice_weights, axis=[1, 2, 3]) # Either 1 or 0
                if self.debug_roi_dice:
                    dice_value.append(tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist)))
                else:
                    dice_value += tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist))
                
                n_rois += tf.reduce_mean(roi_exist)
                weighted_n_rois += tf.reduce_mean(roi_exist) * weights
                weighted_dice_loss = tf.reduce_mean(tf.multiply(tf.truediv(intersection_prob, union_prob), roi_exist)) * weights
    
                # cl dice loss
                if loss_type == 'cl_dice':
                    iter_ = 15
                    skel_prob = soft_skel(slice_prob[:, :, :, :, None], iter_)[:, :, :, :, 0]
                    skel_true = soft_skel(slice_label[:, :, :, :, None], iter_)[:, :, :, :, 0]
                    precision = smooth + tf.reduce_sum(tf.math.multiply(skel_prob, slice_label), axis=[1, 2, 3])
                    precision /= smooth + tf.reduce_sum(skel_prob, axis=[1, 2, 3])
                    recall = smooth + tf.reduce_sum(tf.math.multiply(skel_true, slice_pred), axis=[1, 2, 3])
                    recall /= smooth + tf.reduce_sum(skel_true, axis=[1, 2, 3])
    
                    weighted_dice_loss = tf.reduce_mean(tf.multiply(tf.truediv(intersection_prob, union_prob), roi_exist)) * weights
                    cl_dice_loss = tf.reduce_mean(tf.multiply(tf.truediv((precision * recall), (precision + recall)), roi_exist)) * weights
                    weighted_dice_loss = 0.5 * weighted_dice_loss + 0.5 * cl_dice_loss
                dice_loss += weighted_dice_loss
                
            dice_loss = 1.0 - dice_loss * 2.0 / weighted_n_rois
            
            if self.debug_roi_dice:
                roi_dices = tf.multiply(dice_value, 2.0)
                dice = tf.reduce_sum(roi_dices) / n_rois
            else:
                roi_dices = None
                dice = dice_value * 2.0 / n_rois
            
            if loss_type == 'cross_entropy':
                loss = cross_entropy_loss
            elif loss_type == 'dice':
                loss = dice_loss
            elif loss_type == 'both':
                loss = cross_entropy_loss + dice_loss
            elif loss_type == 'both_focal':
                loss = focal_loss + dice_loss
            elif loss_type == 'cl_dice':
                loss = cross_entropy_loss + dice_loss
            else:
                raise ValueError("Unknown cost function: " + loss_type)

            if head is None:
                final_loss = loss * head_weight
                final_roi_dices = roi_dices
                final_dice = dice * head_weight
                final_cross_entropy_loss = cross_entropy_loss * head_weight
                final_dice_loss = dice_loss * head_weight
            else:
                final_loss += loss * head_weight
                if head['activation'] == 'softmax':
                    rois_indices = head['rois'][1:]
                elif head['activation'] == 'sigmoid':
                    rois_indices = head['rois']
                for i, idx in enumerate(rois_indices):
                    final_roi_dices[idx] = roi_dices[i]
                final_dice += dice * head_weight
                final_cross_entropy_loss += cross_entropy_loss * head_weight
                final_dice_loss += dice_loss * head_weight
        
        if self.debug_loss:
            return final_loss, final_roi_dices, final_dice, final_cross_entropy_loss, final_dice_loss
        else:
            return final_loss, final_roi_dices, final_dice
    
    @property
    def model_dir(self):
        return 'Group' + str(self.group)