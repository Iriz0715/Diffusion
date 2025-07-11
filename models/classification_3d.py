import os, glob, re, time, csv, datetime
import numpy as np
import h5py, json
import multiprocessing, threading
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from augmentation import *

AUGMENTATION_PARAMS = {
    'selected_seg_channels': [0],
    # elastic
    "do_elastic": False,
    'deformation_scale': (0, 0.25),
    'p_eldef': 0.2,
    # scale
    'do_scaling': True,
    'scale_range': (0.7, 1.4),
    'independent_scale_factor_for_each_axis': False,
    'p_independent_scale_per_axis': 1,
    'p_scale': 0.2,
    # rotate
    'do_rotation': True,
    'rotation_x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),  # axial
    'rotation_y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),  # sagittal
    'rotation_z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),  # coronal
    'rotation_p_per_axis': 1,
    'p_rot': 0.2,
    # crop
    'random_crop': False,
    'random_crop_dist_to_border': 10,
    # dummy 2D
    # if set to true, ignore axial augmentation and only do inplane rotation, scaling, random cropping (if available).
    'dummy_2D': False,
    # gamma
    'do_gamma': True,
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,
    # others
    'border_mode_data': 'constant',
}

class BaseModel(Model):
    # With this part, fit() can read our custom data generator
    def call(self, x):
        return self.networks(x)

    def save(self, step, max_to_keep=1):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if len(ckpt_files) >= max_to_keep:
            os.remove(ckpt_files[0])
        
        self.networks.save(os.path.join(checkpoint_dir, 'model_epoch%06d.h5' % step))
    
    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            print('No model is found, please train first')
            return False, 0
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if ckpt_files:
            ckpt_file = ckpt_files[-1]
            ckpt_name = os.path.basename(ckpt_file)
            self.counter = int(re.findall(r'epoch\d+', ckpt_name)[0][5:])  #e.g. model_epoch600.h5
            self.networks = self.build_networks()
            self.networks.load_weights(ckpt_file)
            print('Loaded model checkpoint:', ckpt_name)
            return True, self.counter
        else:
            print('Failed to find a checkpoint')
            return False, 0
    
    def perform_augmentation(self, images, patch_size):
        if len(images.shape) == 4:
            images_aug = np.expand_dims(np.transpose(images, (3, 0, 1, 2)), axis=0)
        else:
            images_aug = np.expand_dims(images, axis=(0, 1))
        
        # Convert to 2D tiles. Reshape axial and channel into one axis. Only applicable to 3D data.
        if AUGMENTATION_PARAMS.get('dummy_2D') and len(images_aug.shape) == 5:
            orig_img_shp = images_aug.shape
            images_aug = images_aug.reshape((orig_img_shp[0], orig_img_shp[1] * orig_img_shp[2],
                                             orig_img_shp[3], orig_img_shp[4]))
            
            assert len(patch_size) == 3
            patch_size = patch_size[1:]
            
            # With dummy_2D aug, it should be 2D like.
            assert len(images_aug.shape) == 4
            
        images_aug, _ = augment_spatial_2(images_aug, None, patch_size=patch_size,
                                          patch_center_dist_from_border=AUGMENTATION_PARAMS.get('random_crop_dist_to_border'),
                                          do_elastic_deform=AUGMENTATION_PARAMS.get('do_elastic'),
                                          deformation_scale=AUGMENTATION_PARAMS.get('deformation_scale'),
                                          do_rotation=AUGMENTATION_PARAMS.get('do_rotation'),
                                          angle_x=AUGMENTATION_PARAMS.get('rotation_x'),
                                          angle_y=AUGMENTATION_PARAMS.get('rotation_y'),
                                          angle_z=AUGMENTATION_PARAMS.get('rotation_z'),
                                          p_rot_per_axis=AUGMENTATION_PARAMS.get('rotation_p_per_axis'),
                                          do_scale=AUGMENTATION_PARAMS.get('do_scaling'),
                                          scale=AUGMENTATION_PARAMS.get('scale_range'),
                                          border_mode_data=AUGMENTATION_PARAMS.get('border_mode_data'),
                                          border_cval_data=0,
                                          order_data=3,
                                          border_mode_seg='constant', border_cval_seg=-1,
                                          order_seg=1, random_crop=AUGMENTATION_PARAMS.get('random_crop'),
                                          p_el_per_sample=AUGMENTATION_PARAMS.get('p_eldef'),
                                          p_scale_per_sample=AUGMENTATION_PARAMS.get('p_scale'),
                                          p_rot_per_sample=AUGMENTATION_PARAMS.get('p_rot'),
                                          independent_scale_for_each_axis=
                                          AUGMENTATION_PARAMS.get('independent_scale_factor_for_each_axis'))
        # Convert back to 3D.
        if AUGMENTATION_PARAMS.get('dummy_2D') and len(images_aug.shape) == 4:
            current_img_shape = images_aug.shape
            images_aug = images_aug.reshape((orig_img_shp[0], orig_img_shp[1], orig_img_shp[2],
                                             current_img_shape[-2], current_img_shape[-1]))
        
        images_aug = np.squeeze(images_aug, axis=0)

        images_aug = augment_gaussian_noise(images_aug, p_per_sample=0.1)
        images_aug = augment_gaussian_blur(images_aug, (0.5, 1.), per_channel=True, p_per_sample=0.2, p_per_channel=0.5)
        images_aug = augment_brightness_multiplicative(images_aug, multiplier_range=(0.75, 1.25), p_per_sample=0.15)
        images_aug = augment_contrast(images_aug, p_per_sample=0.15)
        images_aug = augment_linear_downsampling_scipy(images_aug, zoom_range=(0.5, 1), per_channel=True,
                                                       p_per_channel=0.5,
                                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                       ignore_axes=None)
        images_aug = augment_gamma(images_aug, AUGMENTATION_PARAMS.get('gamma_range'), invert_image=True, per_channel=True,
                                   retain_stats=AUGMENTATION_PARAMS.get('gamma_retain_stats'), p_per_sample=0.1)
        images_aug = augment_gamma(images_aug, AUGMENTATION_PARAMS.get('gamma_range'), invert_image=False, per_channel=True,
                                   retain_stats=AUGMENTATION_PARAMS.get('gamma_retain_stats'),
                                   p_per_sample=AUGMENTATION_PARAMS['p_gamma'])

        if self.mirror_config is not None and self.mirror_config['training_mirror']:
            if self.mirror_config.get('mirror_all_dimensions', False):
                mirror_axes = [1, 2, 3]
            else:
                mirror_axes = self.mirror_config.get('mirror_axes', [1, 2, 3])
            # Axes starts with batch from config
            # e.g. [1, 2] in mirror_config -> [0, 1] in augment_mirroring
            mirror_axes = [ax - 1 for ax in mirror_axes]
            images_aug, _ = augment_mirroring(images_aug, None, mirror_axes)

        if len(images.shape) == 4:
            images_aug = np.transpose(images_aug, (1, 2, 3, 0))
        else:
            images_aug = np.squeeze(images_aug, axis=0)
        return images_aug

    def oversample_paths(self, paths, class_weights=None):
        class_paths = {}
        for path in paths:
            with h5py.File(path, 'r') as f_h5:
                class_label = int(f_h5.attrs['class_label'])
            class_paths[str(class_label)] = class_paths.get(str(class_label), []) + [path]
        
        if class_weights is None:
            num_class_paths = np.array([len(class_paths[c]) for c in class_paths.keys()])
            class_weights = np.round(np.amax(num_class_paths) / num_class_paths, 0)
        
        class_weights = np.array(class_weights).astype(int)
        balanced_paths = []
        for i, c in enumerate(class_paths.keys()):
            balanced_paths += class_paths[c] * class_weights[i]

        # Shuffle balanced paths
        np.random.shuffle(balanced_paths)
        
        print('Classes:', list(class_paths.keys()), '. Number of cases:', num_class_paths)
        print('Oversampling with class_weights:', class_weights)
        print('Balanced number of cases:', list(num_class_paths * class_weights))
        return balanced_paths

    def read_training_inputs(self, file, train_classes, im_size, augmentation=True):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            # Apply segmented mask
            if getattr(self, 'apply_segmentation_mask', False):
                labels = np.asarray(f_h5['labels'], dtype=np.uint8)
                images = images * (labels > 0)
            class_label = int(f_h5.attrs['class_label'])
        
        train_classes = [0,] + train_classes  # Add negative class
        nclass = len(train_classes)
        if class_label in train_classes:
            one_hot_class_label = np.eye(nclass)[train_classes.index(class_label)]
        else:
            one_hot_class_label = np.zeros(nclass)
        
        # Normalize images
        if self.norm_config['norm']:
            if len(images.shape) == 3:
                if self.norm_config['norm_channels'] == 'rgb_channels':
                    rgb_mean = np.array(self.norm_config.get('norm_mean', [0.485, 0.456, 0.406]))
                    rgb_std = np.array(self.norm_config.get('norm_std', [0.229, 0.224, 0.225]))
                    images = (images - rgb_mean) / rgb_std
                elif self.norm_config['norm_channels'] == 'all_channels':
                    images = (images - np.mean(images, axis=(0, 1))) / np.std(images, axis=(0, 1))
                else:
                    for channel in self.norm_config['norm_channels']:
                        m = np.mean(images[..., channel])
                        s = np.std(images[..., channel])  
                        images[..., channel] = (images[..., channel] - m) / s
            else:
                images = (images - np.mean(images)) / np.std(images)

        # Random crop
        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)
        
        if augmentation:
            images = self.perform_augmentation(images, full_size)
        
        x_range = full_size[0] - im_size[0]
        y_range = full_size[1] - im_size[1]
        z_range = full_size[2] - im_size[2]

        if x_range * y_range * z_range > 0:
            p = np.ones((x_range, y_range, z_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range * z_range, p=p)
            x_start, y_start, z_start = np.unravel_index(o, (x_range, y_range, z_range))
            
            images_extracted = images[x_start : x_start + im_size[0], y_start : y_start + im_size[1],
                                      z_start : z_start + im_size[2], ...]  # Maybe 4 dims
        else:
            images_extracted = images
        return images_extracted, one_hot_class_label
    
    def read_testing_inputs(self, file, im_size):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            # Apply segmented mask
            if getattr(self, 'apply_segmentation_mask', False):
                labels = np.asarray(f_h5['labels'], dtype=np.uint8)
                images = images * (labels > 0)
        
        # Normalize images
        if self.norm_config['norm']:
            if len(images.shape) == 3:
                if self.norm_config['norm_channels'] == 'rgb_channels':
                    rgb_mean = np.array(self.norm_config.get('norm_mean', [0.485, 0.456, 0.406]))
                    rgb_std = np.array(self.norm_config.get('norm_std', [0.229, 0.224, 0.225]))
                    images = (images - rgb_mean) / rgb_std
                elif self.norm_config['norm_channels'] == 'all_channels':
                    images = (images - np.mean(images, axis=(0, 1))) / np.std(images, axis=(0, 1))
                else:
                    for channel in self.norm_config['norm_channels']:
                        m = np.mean(images[..., channel])
                        s = np.std(images[..., channel])  
                        images[..., channel] = (images[..., channel] - m) / s
            else:
                images = (images - np.mean(images)) / np.std(images)

        # Center crop
        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)
        if full_size != im_size:
            images_extracted = images[
                full_size[0] // 2 - im_size[0] // 2 : full_size[0] // 2 + im_size[0] // 2, 
                full_size[1] // 2 - im_size[1] // 2 : full_size[1] // 2 + im_size[1] // 2, 
                full_size[2] // 2 - im_size[2] // 2 : full_size[2] // 2 + im_size[2] // 2, ...
            ]
        else:
            images_extracted = images
        
        return images_extracted
            
    def train_data_mapper(self, i):
        images = np.empty((self.im_size[0], self.im_size[1], self.im_size[2], self.input_channels), dtype=np.float32)
        labels = np.empty((self.nclass,), dtype=np.float32)
        if self.input_channels == 1:
            images[..., 0], labels = self.read_training_inputs(self.training_paths[i], self.train_classes, self.im_size)
        else:
            images, labels = self.read_training_inputs(self.training_paths[i], self.train_classes, self.im_size)
        return images, labels

    def process_data_batch(self, q, idx_list):
        while True:
            shuffle_list = np.random.permutation(idx_list)
            images_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], 
                                     self.input_channels), dtype=np.float32)
            labels_batch = np.empty((self.batch_size, self.nclass), dtype=np.float32)
            ib = 0
            idx = 0
            while ib < self.batch_size and idx < len(idx_list):
                i = shuffle_list[idx]
                idx += 1
                images_batch[ib], labels_batch[ib] = self.train_data_mapper(i)
                ib += 1
            # Drop remainder
            if ib < self.batch_size:
                continue
            q.put((images_batch, labels_batch))

    def data_generator(self, idx_list, total_iters):
        q = multiprocessing.Queue(maxsize=self.num_threads * 8)
        pool = multiprocessing.pool.ThreadPool(self.num_threads, initializer=self.process_data_batch, 
                                               initargs=(q, idx_list))
        it = 0
        while it < total_iters:
            try:
                stuff = q.get()
                if stuff is None:
                    break
                images_batch, labels_batch = stuff
                it += 1
                yield images_batch, labels_batch
            except:
                break
        pool.close()
    
    def train(self, run_validation=False, validation_paths=None, custom_log_file=None, **kwargs):
        # Compile model for training
        self.compile_it()
        
        # Training config
        log_dir = os.path.join(self.log_dir, self.model_dir)
        if run_validation:
            profiler = 0  #shutdown by default
        else:
            profiler = 0
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=profiler)
        
        validation_config = None
        if run_validation:
            validation_config = {
                'validation_paths': validation_paths,
                'validation_fn': self.validate,
                'log_dir': log_dir,
                'train_classes': self.train_classes
            }
        saver_config = {
            'period': 50,  #save/validate per 50 training epochs
            'log_dir': log_dir,
            'save_fn': self.save, 
        }
        saver_callback = self.ModelSaver(saver_config=saver_config, validation_config=validation_config, 
                                         custom_log_file=custom_log_file)
        
        # Prepare data generator
        num_samples = len(self.training_paths)
        if num_samples == 0:
            print('No training data')
            return
        idx_list = np.arange(num_samples)
        total_iters = self.steps_per_epoch * (self.epoch - self.counter)
        data_generator = self.data_generator(idx_list, total_iters)
        
        print('Running on complete dataset with total training samples:', num_samples)
        if self.counter < self.epoch:
            self.fit(data_generator, validation_data=None, verbose=1,
                     steps_per_epoch=self.steps_per_epoch, initial_epoch=self.counter, epochs=self.epoch,
                     callbacks=[tensorboard_callback, saver_callback])

            self.save(self.epoch)
        print('--- Training scores ---')
        self.validate(self.training_paths)
        print('--- Validation scores ---')
        print(len(validation_paths))
        if len(validation_paths) > 0:
            self.validate(validation_paths)
        return

    def validate(self, validation_paths):
        from sklearn import metrics
        # Currently, only support binary classification validation, 0 or 1, nclass=2
        all_prob, all_gt = [], []
        
        for validation_path in validation_paths:
            prob = self.run_test(validation_path)  # (1, 2)
            with h5py.File(validation_path, 'r') as f_h5:
                gt = int(f_h5.attrs['class_label'])
            all_prob.append(prob[0, 1])
            all_gt.append(gt)
        all_prob = np.stack(all_prob)  # (N,)
        all_gt = np.stack(all_gt)  # (N,)
        # print(all_prob)
        
        fpr, tpr, thresh = metrics.roc_curve(np.where(all_gt > 0.5, 1, 0), np.where(all_prob > 0.5, 1, 0), pos_label=1)
        acc = metrics.accuracy_score(np.where(all_gt > 0.5, 1, 0), np.where(all_prob > 0.5, 1, 0))
        auc = metrics.auc(fpr, tpr)
        print(metrics.confusion_matrix(np.where(all_gt > 0.5, 1, 0), np.where(all_prob > 0.5, 1, 0)))
        print('Accuracy:', acc, ' AUC:', auc)               
        metrics = {}
        metrics['accuracy'] = acc
        metrics['auc'] = auc
        return metrics
    
    @tf.function
    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            logits = self.networks(images, training=True)
            loss = self.loss_fn(y_true=labels, y_pred=logits)
        # Get the gradients
        gradient = tape.gradient(loss, self.networks.trainable_variables)
        
        # Update the weights
        self.optimizer.apply_gradients(zip(gradient, self.networks.trainable_variables))
        
        # Training logs
        logs = {'loss': loss}
        return logs
    
    @tf.function
    def predict_step(self, data):
        # patch  ##NOTE: in tf>=2.3.0, images=data; in tf==2.2.0, images=data[0]
        if isinstance(data, tuple):
            images = data[0]
        else:
            images = data  
        logits = self.networks(images, training=False)
        probs = tf.nn.softmax(logits)
        probs_count = 1
        mirror_config = getattr(self, 'mirror_config', {})
        if mirror_config.get('testing_mirror', False):
            if mirror_config.get('mirror_all_dimensions', False):
                mirror_axes = [1, 2, 3]
            else:
                mirror_axes = mirror_config.get('mirror_axes', [1, 2, 3])
            
            # e.g. [[1], [2], [1, 2]], axis:0 is batch
            mirror_axes_comb = [[]]
            for ax in mirror_axes:  # get powerset
                mirror_axes_comb += [sub + [ax] for sub in mirror_axes_comb]

            for axis in mirror_axes_comb[1:]:
                images_mirror = tf.reverse(images, axis=axis)
                logits_mirror = self.networks(images_mirror, training=False)
                probs_mirror = tf.nn.softmax(logits_mirror)
                probs += probs_mirror
                probs_count += 1
        probs = probs / probs_count
        return probs
    
    def test(self, testing_paths, output_path, **kwargs):
        if not self._loaded:
            raise Exception('No model is found, please train first')
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for input_file in testing_paths:
            output_labels = self.run_test(input_file)
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file))
            with h5py.File(output_file, 'w') as f_h5:
                f_h5['predictions'] = output_labels.astype(np.float32)
                
    def run_test(self, input_file):
        images = np.empty((1, self.im_size[0], self.im_size[1], self.im_size[2], self.input_channels), dtype=np.float32)
        if self.input_channels == 1:
            images[0, ..., 0] = self.read_testing_inputs(input_file, self.im_size)
        else:
            images[0] = self.read_testing_inputs(input_file, self.im_size)
        
        probs = self.predict_on_batch(images)
        return probs
    
    class ModelSaver(tf.keras.callbacks.Callback):
        def __init__(self, saver_config, validation_config=None, custom_log_file=None):
            self.counter = 0
            self.period = saver_config['period']
            self.save = saver_config['save_fn']
            self.validation_config = validation_config
            self.logs = {}
            self.custom_log_file = custom_log_file
            self.train_writer = tf.summary.create_file_writer(os.path.join(saver_config['log_dir'], 'avg'))
                
            if validation_config is not None:
                self.validation_paths = validation_config['validation_paths']
                self.validation_fn = validation_config['validation_fn']
                self.train_classes = validation_config['train_classes']
                self.test_writer = tf.summary.create_file_writer(os.path.join(validation_config['log_dir'], 'test'))
        
        def on_batch_end(self, batch, logs):
            if len(self.logs) == 0:
                for key in logs.keys():
                    self.logs[key] = [logs[key]]
            else:
                for key in logs.keys():
                    self.logs[key].append(logs[key])
        
        def on_epoch_end(self, epoch, logs):
            self.counter += 1
            # Epoch average logs
            with self.train_writer.as_default():
                record_logs = {}
                for key in self.logs.keys():
                    if 'loss' in key:
                        epoch_avg = np.array(self.logs[key])[np.nonzero(self.logs[key])].mean()
                        tf.summary.scalar(f'epoch_{key}', epoch_avg, step=epoch)
                    else:
                        epoch_avg = np.nanmean(self.logs[key])
                        tf.summary.scalar(f'epoch_{key}', epoch_avg, step=epoch)
                    self.train_writer.flush()
                    record_logs[key] = epoch_avg
            self.logs = {}
            # Save / validate
            if self.counter % self.period == 0:
                self.save(epoch + 1)
                if self.validation_config is not None:
                    metrics = self.validation_fn(self.validation_paths)
                    # for ir, roi in enumerate(self.train_classes):
                    #     for k in metrics.keys():
                    #         tf.summary.scalar(f'{k}_{roi}', metrics[k][ir], step=epoch + 1)
                    #     self.test_writer.flush()