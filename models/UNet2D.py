import os, re, glob, logging
import numpy as np
import h5py, json
import tensorflow as tf

from base_model_2d import BaseModel2D
from networks import *

DEFAULT_MODEL_CONFIG = {
    'pretrained': False,
    'features_root': 32,
    'conv_size': 3,
    'deconv_size': 2,
    'layers': 5,
    'max_filters': 320,
    'dilation': False,
    'loss_type': 'both',
    'batch_size': 8,
    'deep_supervision': True,
    'iters_per_epoch': 1000,  # each epoch runs <= 1000 iters
    'epoch': 1000,
    'save_period': 50,
    'norm_config': {'norm': True, 'norm_channels': 'all_channels',
                    'norm_mean': None, 'norm_std': None},
    'mirror_config': {'training_mirror': False, 'testing_mirror': False,
                      'mirror_axes': [1, 2], 'rot90': True}, # axis 0 is batch
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
    
    # debug config
    'debug_roi_dice': True,  # log dices for each roi in tensorboard
    'debug_loss': False,  # log dice_loss and ce_loss in tensorboard
    
    # experimental
    'attention': False,
    'residual': True,
}

class UNet2D(BaseModel2D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads=6,
                 input_channels=1, class_weights=None, model_config=None, **kwargs):
        
        super(UNet2D, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        self._training_paths_filtered = False
        
        self.group = group
        self.rois = rois
        
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
        
        self.mirror_config = model_config['mirror_config']
        self.augmentation_params = model_config.get('augmentation_params', {})
        self.tta = model_config.get('tta', {})
        
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
        self.debug_loss = model_config['debug_loss']
        self.debug_roi_dice = model_config['debug_roi_dice']
        
        self.input_channels = input_channels
        if self.norm_config.get('norm') and self.norm_config.get('norm_channels') == 'rgb_channels':
            if input_channels != 3:
                print('Stack input channels to 3')
                self.input_channels = 3
                self.original_input_channels = input_channels
        else:
            self.original_input_channels = input_channels
        
        self.nclass = len(self.rois) + 1
        
        self.conv_size = int(model_config['conv_size'])
        self.deconv_size = int(model_config['deconv_size'])
        self.layer_number = int(model_config['layers'])
        self.max_filters = int(model_config['max_filters'])
        self.dilation = model_config['dilation']
        self.deep_supervision = model_config['deep_supervision']
        self.num_threads = num_threads
        self.class_weights = class_weights
        
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        
        self.save_period = model_config['save_period']
        
        self.counter = 0
        
        self.im_size = im_size
        
        self.strides_list = get_strides_list(self.layer_number, self.im_size)

        # Use natural image model config
        self.pretrained = bool(model_config['pretrained'])
        if self.pretrained:
            self.deep_supervision = False
            # If not assign a new lr, use 0.0005~1e-7 by default
            if (self.initial_lr == float(DEFAULT_MODEL_CONFIG['initial_lr']) 
                and self.end_lr == float(DEFAULT_MODEL_CONFIG['end_lr'])):
                self.initial_lr = 0.0005
                self.end_lr = 1e-7
        
        if self.deep_supervision:
            self.ds_loss_weights = [0.] + [2**i/(2**(self.layer_number - 1) - 1) for i in range(self.layer_number - 1)]
            self.deep_supervision_scales = [
                np.prod(self.strides_list[:i], axis=0).tolist() for i in range(self.layer_number)
            ][::-1]
            self.downscaling_model = self.build_auxiliary_ds_model()
        
        # experimental
        self.residual = model_config['residual']
        self.attention = model_config['attention']
        
        self._loaded, self.counter = self.load()
        if not self._loaded:
            print('Initializing...')
            if self.pretrained:
                self.unet = build_natural_image_model(nclass=self.nclass, input_channels=self.input_channels)
            else:
                self.unet = self.build_unet()
            self.unet.summary()
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
        )
        return unet
    
    def compile_it(self):
        self.opt = self.get_optimizer()
        self.compile(optimizer=self.opt, loss=self.loss_fn)
        return
    
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
    
    @tf.function        
    def loss_fn(self, logits, labels, loss_weights):
        if self.policy != 'float32':
            logits = tf.cast(logits, tf.float32)
            labels = tf.cast(labels, tf.float32)
            loss_weights = tf.cast(loss_weights, tf.float32)
        
        flat_logits = tf.reshape(logits, [-1, self.nclass])
        flat_labels = tf.reshape(labels, [-1, self.nclass])
        flat_weights = tf.reshape(loss_weights, [-1, self.nclass])

        weights_logits = tf.multiply(flat_logits, flat_weights)
        weights_labels = tf.multiply(flat_labels, flat_weights)

        # Cross-entrpy loss
        if self.class_weights is not None:
            class_weights = tf.constant(np.asarray(self.class_weights, dtype=np.float32))
            weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=weights_logits, labels=weights_labels)
            
            # First multiply by class weights, then by loss weights due to missing of rois
            weighted_loss = tf.multiply(loss_map, weight_map)
        else:
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=weights_logits, labels=weights_labels)
            weighted_loss = loss_map
        cross_entropy_loss = tf.reduce_mean(weighted_loss)
            
        if 'focal' in self.loss_type:
            # Focal Loss (categorical, multi-class)
            beta = 1.0 * self.nclass  # scaling factor for categorical focal loss
            gamma = 2.0  # focusing factor

            ce_loss = weighted_loss  # use class-weighted ce loss calculated as above (replace alpha)
            pt = tf.nn.softmax(weights_logits)
            focal_loss_map = beta * tf.pow(1.0 - pt, gamma) * (weights_labels * ce_loss[:, None])
            focal_loss = tf.reduce_mean(focal_loss_map)

        # Dice loss
        probs = tf.nn.softmax(logits)
        predictions = tf.argmax(probs, 3)
        eps = 1e-5
        if self.debug_roi_dice:
            dice_value = []
        else:
            dice_value = 0.
        dice_loss = 0.
        n_rois = 0
        weighted_n_rois = 0
        
        for i in range(1, self.nclass):
            if self.class_weights is not None:
                weights = self.class_weights[i]
            else:
                weights = 1.0
                
            slice_weights = tf.squeeze(tf.slice(loss_weights, [0, 0, 0, i], [-1, -1, -1, 1]), axis=3)
            slice_prob = tf.squeeze(tf.slice(probs, [0, 0, 0, i], [-1, -1, -1, 1]), axis=3)
            slice_pred = tf.cast(tf.equal(predictions, i), tf.float32)
            slice_label = tf.squeeze(tf.slice(labels, [0, 0, 0, i], [-1, -1, -1, 1]), axis=3)
            intersection_prob = eps + tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2])
            intersection_pred = eps + tf.reduce_sum(tf.multiply(slice_pred, slice_label), axis=[1, 2])
            union_prob = 2.0 * eps + tf.reduce_sum(slice_prob, axis=[1, 2]) + tf.reduce_sum(slice_label, axis=[1, 2])
            union_pred = 2.0 * eps + tf.reduce_sum(slice_pred, axis=[1, 2]) + tf.reduce_sum(slice_label, axis=[1, 2])
            
            # Multiply by loss weights
            roi_exist = tf.reduce_mean(slice_weights, axis=[1, 2]) # Either 1 or 0
            if self.debug_roi_dice:
                dice_value.append(tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist)))
            else:
                dice_value += tf.reduce_mean(tf.multiply(tf.truediv(intersection_pred, union_pred), roi_exist))
            
            n_rois += tf.reduce_mean(roi_exist)
            weighted_n_rois += tf.reduce_mean(roi_exist) * weights
            dice_loss += tf.reduce_mean(tf.multiply(tf.truediv(intersection_prob, union_prob), roi_exist)) * weights
            
        dice_loss = 1.0 - dice_loss * 2.0 / weighted_n_rois
        
        if self.debug_roi_dice:
            roi_dices = tf.multiply(dice_value, 2.0)
            dice = tf.reduce_sum(roi_dices) / n_rois
        else:
            roi_dices = None
            dice = dice_value * 2.0 / n_rois
        
        if self.loss_type == 'cross_entropy':
            loss = cross_entropy_loss
        elif self.loss_type == 'dice':
            loss = dice_loss
        elif self.loss_type == 'both':
            loss = cross_entropy_loss + dice_loss
        elif self.loss_type == 'both_focal':
            loss = focal_loss + dice_loss
        else:
            raise ValueError('Unknown cost function: ' + self.loss_type)
        
        if self.debug_loss:
            return loss, roi_dices, dice, cross_entropy_loss, dice_loss
        else:
            return loss, roi_dices, dice
    
    @property
    def model_dir(self):
        return 'Group' + str(self.group)

    
class MobileUNet2D(UNet2D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads=6,
                 input_channels=1, class_weights=None, model_config=None, **kwargs):
        
        super(MobileUNet2D, self).__init__(checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads,
                                           input_channels, class_weights, model_config)
        # Overwrite debug loss, dice
        self.debug_loss = False
        self.debug_roi_dice = True
        
        # Check unsupported argument
        if self.layer_number != 5:
            raise Exception('Layers other than 5 is not supported')
        
        self.temperature = model_config['temperature']
        self.distillation_weight = model_config['distillation_weight']
        self.mode = model_config['mode']
        self.teacher_model_dir = model_config['teacher_model_dir']
        
        # Requires teacher model to be trained first
        if self.mode == 2:
            self.teacher_model = self.build_unet()
            ckpt_files = sorted(glob.glob(os.path.join(self.teacher_model_dir, self.model_dir, '*.h5')))
            if not ckpt_files:
                raise Exception('No teacher model found')
            self.teacher_model.load_weights(ckpt_files[-1])

        if not self._loaded:      
        # Build student model, reuse name unet so that save and load still work
            self.unet = build_mobileunet(im_size=self.im_size + [self.input_channels], nclass=self.nclass)
    
    def probability_aug(self, teacher_model, images, temperature):
        # Define the augmentations and their inverse functions
        augmentations = []

        # Origina
        augmentations.append((images, lambda x: x))

        # Rotated images
        images_rot90 = tf.image.rot90(images)
        augmentations.append((images_rot90, lambda x: tf.image.rot90(x, k=-1)))

        # Generate all combinations of mirror axes
        mirror_axes = [[1], [2], [1, 2]]  # Height, Width, Both

        for axes in mirror_axes:
            # Mirrored images
            images_mirror = tf.reverse(images, axis=axes)
            augmentations.append(
                (images_mirror, lambda x, axes=axes: tf.reverse(x, axis=axes))
            )

            # Mirrored and rotated images
            images_mirror_rot90 = tf.image.rot90(images_mirror)
            augmentations.append((
                images_mirror_rot90,
                lambda x, axes=axes: tf.reverse(tf.image.rot90(x, k=-1), axis=axes)
            ))

        # Stack all augmented images
        augmented_images = tf.concat([aug[0] for aug in augmentations], axis=0)

        # Pass all augmented images through the model in a single forward pass
        logits_list = teacher_model(augmented_images, training=False)

        # Initialize a list to store final probabilities for each output tensor
        final_probs_list = []

        # Process each output tensor from the model
        for logits in logits_list:
            # Split the logits back into individual augmentations
            split_logits = tf.split(logits, num_or_size_splits=len(augmentations), axis=0)

            # Process each augmentation
            processed_probs = []
            for logit, (_, inverse_fn) in zip(split_logits, augmentations):
                # Apply softmax with temperature scaling
                probs = tf.nn.softmax(logit / temperature)
                # Apply inverse transformation
                probs = inverse_fn(probs)
                processed_probs.append(probs)

            # Sum and average the probabilities
            sum_probs = tf.add_n(processed_probs)
            avg_probs = sum_probs / tf.cast(len(augmentations), dtype=sum_probs.dtype)

            # Append to the list of final probabilities
            final_probs_list.append(avg_probs)

        return final_probs_list
    
    @tf.function        
    def loss_fn(self, logits, labels, loss_weights, teacher_probs):
        loss, roi_dices, dice = super(MobileUNet2D, self).loss_fn(logits, labels, loss_weights)

        if self.mode == 2:
            # distilation loss
            log_teacher_probs = tf.math.log(teacher_probs + 1e-10)
            log_student_probs = tf.nn.log_softmax(logits / self.temperature)
            d_loss = tf.reduce_mean(tf.reduce_sum(teacher_probs * (log_teacher_probs - log_student_probs), axis=-1))
            loss = (1 - self.distillation_weight) * loss + self.distillation_weight * d_loss * (self.temperature ** 2)

        return loss, roi_dices, dice
    
    @tf.function
    def train_step(self, data):
        images, labels, loss_weights = data
        if self.mode == 2:
            # Calculate probs from teacher model
            teacher_probs = self.probability_aug(self.teacher_model, images, self.temperature)
            # teacher_logits = self.teacher_model(images, training=False)
            # teacher_probs = []
            # for tensor in teacher_logits:
            #     probs = tf.nn.softmax(tensor / self.temperature)
            #     teacher_probs.append(probs)
        else:
            teacher_probs = [None, None, None, None]

        with tf.GradientTape() as tape:
            logits = self.unet(images, training=True)
            if self.deep_supervision:
                down_labels = self.downscaling_model(labels)
                down_loss_weights = self.downscaling_model(loss_weights)
                ds_weighted_loss = 0.
                for layer in range(len(logits)):
                    logits_layer = logits[layer]  # logits: collected from top to bottom
                    teacher_probs_layer = teacher_probs[layer]
                    down_labels_layer = down_labels[self.layer_number - layer - 2] # collected from bottom to top
                    down_loss_weights_layer = down_loss_weights[self.layer_number - layer - 2]
                    
                    # Calculate loss
                    ls, roi_d, d = self.loss_fn(logits_layer, down_labels_layer, down_loss_weights_layer, teacher_probs_layer)
                    if layer == 0:
                        dice = d
                        roi_dices = roi_d
                    
                    ds_weighted_loss += self.ds_loss_weights[self.layer_number - layer - 1] * ls
                loss = ds_weighted_loss
            else:
                # Calculate loss
                loss, roi_dices, dice = self.loss_fn(logits[0], labels, loss_weights, teacher_probs_layer[0])
                
        # Get the gradients
        gradient = tape.gradient(loss, self.unet.trainable_variables)
        
        # Update the weights
        self.optimizer.apply_gradients(zip(gradient, self.unet.trainable_variables))
        
        # Training logs
        logs = {'loss': loss, 'dice': dice}
        for i in range(len(self.rois)):
            logs['roi_' + str(self.rois[i])] = roi_dices[i]
        return logs