import os, re, glob, logging
import numpy as np
import h5py, json
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization

from detection_3d import BaseModel3D, create_fpr_dataset
from utils_detection_3d import AnchorBox, LabelEncoder, DecodePredictions
from networks import build_retinaunet

DEFAULT_MODEL_CONFIG = {
    'features_root': 32,
    'conv_size': 3,
    'deconv_size': 2,
    'layers': 5,
    'max_filters': 320,
    'dilation': False,
    'test_nstride': [2, 2, 2],
    'loss_type': 'both',
    'batch_size': 4,
    'norm_config': {'norm': False, 'norm_channels': 'all_channels'},
    'iters_per_epoch': 100,  #each epoch only runs 100 iters
    'epoch': 2000,
    'save_period': 100,  #save/validate per 100 training epochs
    
    'optimizer_type': 'SGD', # currently only supporting 'SGD' and 'Adam'
    # only apply to SGD
    # momentum 0.99 is good for most situations
    # if the training/testing dice is 0, we could lower down the momentum to 0.95
    # if the label is very heterogenous, probably lower the momentum to 0.90
    'sgd_momentum': 0.99,
    # increase this ratio when there are very small structures
    'fg_sampling_ratio': 0.9,
    
    # Detection config
    'feature_pyramid': True,
    ## For anchor box
    'aspect_ratios': [0.5, 1.0, 2.0],
    # Set upsampling ratio for input images/bboxes (anchors will be upsampled as well)
    #   to avoid the network failure in too small bboxes in some axes.
    'detection_upsample': [1, 1, 1],
    # anchor_scales: A list of float values representing the base (smallest) scale of the anchor boxes 
    #   at each location on the first (start_level) feature map. 
    'anchor_scales': [[4, 4, 4], [5, 6, 6], [6, 8, 8]],
    # start from what feature level, 0: the top level with original resolution.
    #   e.g. start_level=1 and num_head_levels=4 -> [1, 2, 3, 4] these levels are used for detection head.
    'start_level': 1,
    'num_head_levels': 4,
    ## For label encoder
    # normalize the boxes [x, y, z, x_range, y_range, z_range] for stable training. 1.: without norm.
    'box_variance': [0.1, 0.1, 0.1, 0.2, 0.2, 0.2], 
    'match_iou': 0.4,
    'ignore_iou': 0.2,
    ## For predictions decoder
    'confidence_threshold': 0.1,
    'nms_iou_threshold': 0.3, 
    'max_detections_per_class': 50,
    'max_detections': 50,

    # experimental
    'residual': True,
}


class RetinaUNet3D(BaseModel3D):
    def __init__(self, checkpoint_dir, log_dir, training_paths, group, rois, im_size, num_threads, input_channels=1, 
                 class_weights=None, left_right_swap_config=None, model_config=None, mirror_config=None, **kwargs):
        super(RetinaUNet3D, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.training_paths = training_paths
        
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
        
        self.batch_size = int(model_config['batch_size'])
        self.steps_per_epoch = model_config['iters_per_epoch'] // self.batch_size
        self.epoch = int(model_config['epoch'])
        self.save_period = model_config['save_period']
        self.features_root = int(model_config['features_root'])
        self.loss_type = model_config['loss_type']
        self.norm_config = model_config['norm_config']
        
        self.optimizer_type = model_config['optimizer_type']
        self.sgd_momentum = model_config['sgd_momentum']
        self.fg_sampling_ratio = model_config['fg_sampling_ratio']
        
        self.nclass = len(self.rois)  # background is class -1
        
        self.conv_size = int(model_config['conv_size'])
        self.deconv_size = int(model_config['deconv_size'])
        self.layer_number = int(model_config['layers'])
        self.max_filters = int(model_config['max_filters'])
        self.dilation = model_config['dilation']
        self.test_nstride = model_config['test_nstride']
        self.feature_pyramid = model_config['feature_pyramid']
        self.num_threads = num_threads
        self.class_weights = class_weights
        self.input_channels = input_channels
        
        self.im_size = im_size
        # A bit arbitrary to enlarge by 1.1875
        self.enlarged_im_size = (int(im_size[0] * 1.1875), int(im_size[1] * 1.1875), int(im_size[2] * 1.1875))
        self.strides_list = self.get_strides_list(self.layer_number, self.im_size, mode='last')  # stride 1 last
        print(self.strides_list)
        
        # Config for detection
        self.detection_upsample = model_config['detection_upsample']
        # Anchor box
        self.start_level = model_config['start_level']
        self.num_head_levels = model_config['num_head_levels']
        self.aspect_ratios = model_config['aspect_ratios']
        # Calculate the accumulated scales for generating anchors
        self.anchor_strides = np.stack([np.prod(self.strides_list[:i+1], axis=0) 
                                        for i in range(len(self.strides_list))], axis=0)
        self.anchor_strides = self.anchor_strides[self.start_level : self.start_level + self.num_head_levels]
        self.anchor_scales = model_config['anchor_scales']
        self.anchor_box = AnchorBox(aspect_ratios=self.aspect_ratios, scales=self.anchor_scales, strides=self.anchor_strides)
        self.num_anchors = self.anchor_box._num_anchors
        print('num_anchors:', self.num_anchors)
        print('anchor_dims:', self.anchor_box._anchor_dims)
        
        # Label encoder
        self.box_variance = model_config['box_variance']
        self.match_iou = model_config['match_iou']
        self.ignore_iou = model_config['ignore_iou']
        self.label_encoder = LabelEncoder(anchor_box=self.anchor_box, box_variance=self.box_variance, 
                                          match_iou=self.match_iou, ignore_iou=self.ignore_iou)
        
        # Prediction decoder
        self.confidence_threshold = model_config['confidence_threshold']
        self.nms_iou_threshold = model_config['nms_iou_threshold']
        self.max_detections = model_config['max_detections']
        self.max_detections_per_class = model_config['max_detections_per_class']
        self.prediction_decoder = DecodePredictions(anchor_box=self.anchor_box,
                                                    num_classes=self.nclass,
                                                    confidence_threshold=self.confidence_threshold,
                                                    nms_iou_threshold=self.nms_iou_threshold,
                                                    max_detections_per_class=self.max_detections_per_class,
                                                    max_detections=self.max_detections, 
                                                    box_variance=self.box_variance)
        
        self.inference_model = None  # decode results to bbox

        self.residual = model_config['residual']
        
        self.counter = 0
        self._loaded, self.counter = self.load()
        if not self._loaded:
            print('Initializing...')
            self.networks = self.build_networks()
            print('Complete initialization.')
            self.networks.summary()

        # Log variables
        vars_log_path = os.path.join(self.log_dir, self.model_dir, 'vars.txt')
        os.makedirs(os.path.dirname(vars_log_path), exist_ok=True)
        self.vars_log_path = vars_log_path
        self_vars = {k: vars(self).get(k) for k in dir(self)
                     if not k.startswith('_') and vars(self).get(k) is not None}
        logging.basicConfig(filename=vars_log_path, level=logging.INFO)
        logging.info(self_vars)
    
    def compile_it(self):
        opt = self.get_optimizer()
        # use build-in compile, call fit by self.networks.fit()
        self.networks.compile(loss=self.loss_fn, optimizer=opt)
        return
    
    def get_optimizer(self):
        starter_learning_rate = 5e-4
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
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=self.sgd_momentum, clipnorm=12, 
                                          nesterov=True)
        elif self.optimizer_type == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=12)
 
        return opt
    
    def _box_loss(self, y_true, y_pred, delta=1.0):
        # Box regression loss - Smooth L1 loss
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)
    
    def _cls_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        # Classification loss - Focal loss
        # Focal loss automatically handles the class imbalance, hence weights are not required for the focal loss. 
        # The alpha and gamma factors handle the class imbalance in the focal loss equation. 
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha_map = tf.where(tf.equal(y_true, 1.0), alpha, (1.0 - alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha_map * tf.pow(1.0 - pt, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    
    @tf.function
    def loss_fn(self, labels, logits):  # the build-in call method is (y_true, y_pred)
        logits = tf.cast(logits, dtype=tf.float32)
        box_labels = labels[:, :, 1:]  # labels: channel[0] - cls; channel[1:] - box
        box_predictions = logits[:, :, self.nclass:]  # predictions: channel[:nclass] - cls; channel[nclass:] - box
        cls_labels = tf.one_hot(
            tf.cast(labels[:, :, 0], dtype=tf.int32),
            depth=self.nclass,
            dtype=tf.float32,
        )
        cls_predictions = logits[:, :, :self.nclass]
        positive_mask = tf.cast(tf.greater(labels[:, :, 0], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(labels[:, :, 0], -2.0), dtype=tf.float32)
        cls_loss = self._cls_loss(cls_labels, cls_predictions, alpha=0.25, gamma=2.0)  # set [alpha, gamma] here
        box_loss = self._box_loss(box_labels, box_predictions, delta=1.0)  # set [delta] here
        cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = cls_loss + box_loss
        return loss

    def build_networks(self):
        retinaunet = build_retinaunet(
            im_size=self.im_size, 
            nclass=self.nclass,
            strides_list=self.strides_list,
            input_channels=self.input_channels, 
            features_root=self.features_root,
            conv_size=self.conv_size, 
            deconv_size=self.deconv_size, 
            layer_number=self.layer_number, 
            start_level=self.start_level, 
            num_head_levels=self.num_head_levels,
            num_anchors=self.num_anchors,
            max_filters=self.max_filters, 
            dilation=self.dilation, 
            feature_pyramid=self.feature_pyramid,
            use_residual_encoder=self.residual,
            freeze_input_shape=True,
        )
        return retinaunet

    @property
    def model_dir(self):
        return 'Group' + str(self.group)
