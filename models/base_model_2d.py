import os, glob, re, time, csv, datetime, shutil, subprocess
from tqdm import tqdm
import numpy as np
import h5py, json
import multiprocessing
from skimage import measure
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

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
    'random_crop_dist_to_border': 32,
    # gamma
    'do_gamma': True,
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,
    # others
    'border_mode_data': 'constant',
}

class BaseModel2D(tf.keras.models.Model):
    # With this part, fit() can read our custom data generator
    def call(self, x):
        return self.unet(x)

    def save(self, step, max_to_keep=1):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if len(ckpt_files) >= max_to_keep:
            os.remove(ckpt_files[0])
        
        self.unet.save(os.path.join(checkpoint_dir, 'model_epoch%06d.h5' % step))
    
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
            self.unet = tf.keras.models.load_model(ckpt_file, compile=False)
            print('Loaded model checkpoint:', os.path.basename(os.path.dirname(ckpt_file)), ckpt_name)
            return True, self.counter
        else:
            print('Failed to find a checkpoint')
            return False, 0

    @staticmethod
    def filter_training_cases(cases, rois):
        output_cases = []
        for case in cases:
            with h5py.File(case, 'r') as f_h5:
                labels = np.asarray(f_h5['labels'], dtype=np.float32)
            
            # roi labels only contain the labels within the model rois
            roi_labels = np.zeros(labels.shape, dtype=np.uint8)
            for ir, roi in enumerate(rois):
                roi_labels[labels == roi] = ir + 1
                
            if np.amax(roi_labels) > 0:
                output_cases.append(case)
        print('Data shape:', labels.shape)
        return output_cases
    
    def perform_augmentation(self, images, labels, patch_size):
        AUGMENTATION_PARAMS.update(self.augmentation_params)
        if self.mirror_config is not None and self.mirror_config['training_mirror'] and self.mirror_config['rot90']:
            # Random rotate 0, 90, 180, 270
            k_rot90 = np.random.randint(4)
            images = np.rot90(images, k_rot90)
            labels = np.rot90(labels, k_rot90)
        
        labels_aug = np.expand_dims(labels, axis=(0, 1))
        if len(images.shape) == 3:
            images_aug = np.expand_dims(np.transpose(images, (2, 0, 1)), axis=0)
        else:
            images_aug = np.expand_dims(images, axis=(0, 1))
        
        images_aug, labels_aug = augment_spatial_2(images_aug, labels_aug, patch_size=patch_size, 
                                                   patch_center_dist_from_border=
                                                   AUGMENTATION_PARAMS.get('random_crop_dist_to_border'),
                                                   do_elastic_deform=AUGMENTATION_PARAMS.get('do_elastic'),
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

        images_aug = np.squeeze(images_aug, axis=0)
        labels_aug = np.squeeze(labels_aug, axis=0)
        
        if self.mirror_config is not None and self.mirror_config.get('training_mirror', False):
            # Axes starts with batch from config
            # e.g. [1, 2] in mirror_config -> [0, 1] in augment_mirroring
            if self.mirror_config.get('mirror_all_dimensions'):
                mirror_axes = [1, 2]
            else:
                mirror_axes = self.mirror_config.get('mirror_axes', [1, 2])
            mirror_axes = [ax - 1 for ax in mirror_axes]
            images_aug, labels_aug = augment_mirroring(images_aug, labels_aug, mirror_axes)
        
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
        
        if len(images.shape) == 3:
            images_aug = np.transpose(images_aug, (1, 2, 0))
        else:
            images_aug = np.squeeze(images_aug, axis=0)
        labels_aug = np.squeeze(labels_aug, axis=0)
        return images_aug, labels_aug


    def read_training_inputs(self, file, rois, im_size, augmentation=True, class_balanced=True):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            labels = np.asarray(f_h5['labels'], dtype=np.float32)
        
        # Scale to 0 to 1 if need
        if self.pretrained and np.amax(images) > 1:
            images = images / 255.
        
        nclass = len(rois) + 1
        
        # Our 2D image dim standard is [batch(always=1 / maybe missing), h, w, channel(maybe missing)]
        # In training, we random choose one slice in batch, and drop batch dim
        if len(images.shape) == 4:
            full_size = list(images.shape)[1:-1]
            sli = np.random.choice(images.shape[0])
            images = images[sli]
            labels = labels[sli]
        elif len(images.shape) == 3:
            if self.original_input_channels == images.shape[-1]:
                full_size = list(images.shape)[:-1]
            else:
                full_size = list(images.shape[1:])
                sli = np.random.choice(images.shape[0])
                images = images[sli]
                labels = labels[sli]
        elif len(images.shape) == 2:
            full_size = list(images.shape)

        # input_channels: model input. original_input_channels: raw data
        if self.input_channels != self.original_input_channels:
            if len(images.shape) == 2:
                images = np.stack([images, images, images], axis=2)
            if len(images.shape) == 2:
                images = np.concatenate([images, images, images], axis=2)
        
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

        # Pad if full size is smaller than patch size
        # symmetric padding is always better than one-side padding
        pads = [max(0, im_size[ax] - full_size[ax]) for ax in range(len(im_size))]

        if np.any(np.array(pads) > 0):
            if len(images.shape) == 3:
                pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2), (0, 0))
            else:
                pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2))
            images = np.pad(images, pad_with, mode='constant')
            pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                        (pads[1] // 2, pads[1] - pads[1] // 2))
            labels = np.pad(labels, pad_with, mode='constant')

        full_size = list(images.shape)[:2]
        
        x_range = max(full_size[0] - im_size[0], 1)
        y_range = max(full_size[1] - im_size[1], 1)

        x_offset = int(im_size[0] / 2)
        y_offset = int(im_size[1] / 2)

        la = labels[x_offset : x_offset + x_range, y_offset : y_offset + y_range]

        # Get sampling prob
        # Random sampling ? of the time and always choose fg the rest of the time
        if np.random.random() > self.fg_sampling_ratio:
            # choose random
            o = np.random.choice(x_range * y_range)
        else:
            exist_classes = np.unique(la)[1:].astype(int)  # except bg:0
            # choose foreground of this class
            p = np.zeros((x_range, y_range), dtype=np.float32)
            if len(exist_classes) > 0:
                # choose class
                if class_balanced:
                    selected_class = np.random.choice(exist_classes)
                    p[la == selected_class] = 1
                else:
                    p[la > 0] = 1
            else:
                # if foreground is not present (gives NaN value for p)
                p = np.ones((x_range, y_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range, p=p)    
        x_start, y_start = np.unravel_index(o, (x_range, y_range))
        
        if augmentation:
            images, labels = self.perform_augmentation(images, labels, labels.shape)
        
        images_extracted = images[x_start : x_start + im_size[0], 
                                  y_start : y_start + im_size[1], ...]  # Maybe 3 dims
        labels_extracted = labels[x_start : x_start + im_size[0], 
                                  y_start : y_start + im_size[1]]
        
        final_labels = np.zeros(im_size + [nclass], dtype=np.float32)
        loss_weights = np.ones(im_size + [nclass], dtype=np.float32) # always used for anatomy, not lesion
        for z in range(1, nclass):
            final_labels[labels_extracted == rois[z - 1], z] = 1
        final_labels[..., 0] = np.amax(final_labels[..., 1:], axis=2) == 0

        # In case one voxel is assigned to 1 more than 1 times (very rare)
        final_labels = final_labels / np.expand_dims(np.sum(final_labels, axis=2), axis=-1)
        
        return images_extracted, final_labels, loss_weights
    
    def read_testing_inputs(self, file, im_size):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
        
        # Scale to 0 to 1 if need
        if self.pretrained and np.amax(images) > 1:
            images = images / 255.
        
        # Our 2D image dim standard is [batch(always=1 / maybe missing), h, w, channel(maybe missing)]
        # In testing, we keep the batch dim
        if len(images.shape) == 4:
            full_size = list(images.shape)[1:-1]
        elif len(images.shape) == 3:
            if self.original_input_channels == images.shape[-1]:
                full_size = list(images.shape)[:-1]
                images = np.expand_dims(images, axis=0)
            else:
                full_size = list(images.shape[1:])
                images = np.expand_dims(images, axis=-1)
        elif len(images.shape) == 2:
            full_size = list(images.shape)
            images = np.expand_dims(np.expand_dims(images, axis=0), axis=-1)
        
        info = {
            'full_size': full_size,
        }

        # input_channels: model input. original_input_channels: raw data
        if self.input_channels != self.original_input_channels:
            if len(images.shape) == 3:
                images = np.stack([images, images, images], axis=3)
            if len(images.shape) == 4:
                images = np.concatenate([images, images, images], axis=3)
        
        all_images = images.copy()
        for sli in range(all_images.shape[0]):
            images = all_images[sli]
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
            all_images[sli] = images
        return all_images, info

    def train_data_mapper(self, i):
        images = np.empty((self.im_size[0], self.im_size[1], self.input_channels), dtype=np.float32)
        labels = np.empty((self.im_size[0], self.im_size[1], self.nclass), dtype=np.float32)
        weights = np.empty((self.im_size[0], self.im_size[1], self.nclass), dtype=np.float32)
        if self.input_channels == 1:
            images[..., 0], labels, weights = self.read_training_inputs(self.training_paths[i], self.rois, self.im_size)
        else:
            images, labels, weights = self.read_training_inputs(self.training_paths[i], self.rois, self.im_size)
        return images, labels, weights

    def process_data_batch(self, q, idx_list):
        while True:
            shuffle_list = np.random.permutation(idx_list)
            images_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.input_channels),
                                    dtype=np.float32)
            labels_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.nclass),
                                    dtype=np.float32)
            weights_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.nclass),
                                     dtype=np.float32)
            ib = 0
            idx = 0
            while ib < self.batch_size and idx < len(idx_list):
                i = shuffle_list[idx]
                idx += 1
                images_batch[ib], labels_batch[ib], weights_batch[ib] = self.train_data_mapper(i)
                ib += 1
            # Drop remainder
            if ib < self.batch_size:
                continue
            q.put((images_batch, labels_batch, weights_batch))

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
                images_batch, labels_batch, weights_batch = stuff
                it += 1
                yield images_batch, labels_batch, weights_batch
            except:
                break
        pool.close()
    
    def train(self, run_validation=False, validation_paths=None, custom_log_file=None, **kwargs):
        if not self._training_paths_filtered:
            self.training_paths = self.filter_training_cases(self.training_paths, self.rois)
            self._training_paths_filtered = True
        
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
                'rois': self.rois
            }       
        saver_config = {
            'period': self.save_period,  #save/validate per 50 training epochs
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
        
        if total_iters > 0:
            print('Running on complete dataset with total training samples:', num_samples)
            self.fit(data_generator, validation_data=None, verbose=2,
                     steps_per_epoch=self.steps_per_epoch, initial_epoch=self.counter, epochs=self.epoch,
                     callbacks=[tensorboard_callback, saver_callback])
        
            self.save(self.epoch)
        
        if run_validation:
            print(len(validation_paths))
            print('validation score:', self.validate(validation_paths))
        return
    
    def validate(self, validation_paths):
        eps = 1e-5
        dice_scores = []
        
        pbar = tqdm(validation_paths)
        for validation_path in pbar:
            predictions = self.run_test(validation_path)
            with h5py.File(validation_path, 'r') as f_h5:
                labels = np.asarray(f_h5['labels'], dtype=np.float32)

            dice = []
            for ir, roi in enumerate(self.rois):
                gt = (labels == roi).astype(np.float32)
                model = (predictions == ir + 1).astype(np.float32)
                d = 2.0 * (np.sum(gt * model) + eps) / (np.sum(gt) + np.sum(model) + 2.0 * eps)
                dice.append(d)
            dice_scores.append(dice)
        dice_scores = np.asarray(dice_scores, dtype=np.float32)
        test_dice_mean = np.nanmean(dice_scores, axis=0)
        
        # Keep the same shape (n,) when there is no validation paths
        if len(test_dice_mean.shape) == 0:
            test_dice_mean = np.array([np.nan for _ in self.rois])
        
        return test_dice_mean
    
    @tf.function
    def train_step(self, data):
        images, labels, loss_weights = data

        with tf.GradientTape() as tape:
            logits = self.unet(images, training=True)
            if self.deep_supervision:
                down_labels = self.downscaling_model(labels)
                down_loss_weights = self.downscaling_model(loss_weights)
                ds_weighted_loss = 0.
                if self.debug_loss:
                    ds_weighted_ce_loss = 0.
                    ds_weighted_dice_loss = 0.
                for layer in range(len(logits)):
                    logits_layer = logits[layer]  #logits: collected from top to bottom
                    down_labels_layer = down_labels[self.layer_number - layer - 2] #collected from bottom to top
                    down_loss_weights_layer = down_loss_weights[self.layer_number - layer - 2]
                    
                    # Calculate loss
                    if self.debug_loss:
                        ls, roi_d, d, ce_ls, dc_ls = self.loss_fn(logits_layer, down_labels_layer, down_loss_weights_layer)
                    else:
                        ls, roi_d, d = self.loss_fn(logits_layer, down_labels_layer, down_loss_weights_layer)
                    if layer == 0:
                        dice = d
                        if self.debug_roi_dice:
                            roi_dices = roi_d
                    ds_weighted_loss += self.ds_loss_weights[self.layer_number - layer - 1] * ls
                    if self.debug_loss:
                        ds_weighted_ce_loss += self.ds_loss_weights[self.layer_number - layer - 1] * ce_ls
                        ds_weighted_dice_loss += self.ds_loss_weights[self.layer_number - layer - 1] * dc_ls
                loss = ds_weighted_loss
                if self.debug_loss:
                    ce_loss = ds_weighted_ce_loss
                    dice_loss = ds_weighted_dice_loss
            else:
                # Calculate loss
                if self.debug_loss:
                    loss, roi_dices, dice, ce_loss, dice_loss = self.loss_fn(logits, labels, loss_weights)
                else:
                    loss, roi_dices, dice = self.loss_fn(logits, labels, loss_weights)
        # Get the gradients
        gradient = tape.gradient(loss, self.unet.trainable_variables)
        
        # Update the weights
        self.optimizer.apply_gradients(zip(gradient, self.unet.trainable_variables))
        
        # Training logs
        logs = {'loss': loss, 'dice': dice}
        if self.debug_roi_dice:
            for i in range(len(self.rois)):
                logs['roi_' + str(self.rois[i])] = roi_dices[i]
        if self.debug_loss:
            logs['ce_loss'] = ce_loss
            logs['dice_loss'] = dice_loss
            return logs
        else:
            return logs
    
    @tf.function
    def predict_step(self, data):
        deep_supervision = getattr(self, 'deep_supervision', False)
        # patch  ##NOTE: in tf>=2.3.0, images=data; in tf==2.2.0, images=data[0]
        if isinstance(data, tuple):
            images = data[0]
        else:
            images = data

        logits = self(images, training=False)
        if deep_supervision:
            probs = tf.nn.softmax(logits[0])
        else:
            probs = tf.nn.softmax(logits)
        probs_count = 1

        tta = getattr(self, 'tta', {})
        nrotation = tta.get('nrotation', 0)
        if nrotation > 0:
            for rot in range(nrotation):
                radian = rot * 3.14159265359 * 2 / nrotation
                images_rot = tfa.image.rotate(images, radian)
                logits_rot = self(images_rot, training=False)
                if deep_supervision:
                    probs_rot = tf.nn.softmax(logits_rot[0])
                else:
                    probs_rot = tf.nn.softmax(logits_rot)
                probs_rot = tfa.image.rotate(probs_rot, -radian)
                probs += probs_rot
                probs_count += 1

        mirror_config = getattr(self, 'mirror_config', {})
        if mirror_config.get('testing_mirror', False):
            if mirror_config.get('rot90', False):
                images_rot90 = tf.image.rot90(images)
                logits_rot90 = self(images_rot90, training=False)
                if deep_supervision:
                    probs_rot90 = tf.nn.softmax(logits_rot90[0])
                else:
                    probs_rot90 = tf.nn.softmax(logits_rot90)
                probs_rot90 = tf.image.rot90(probs_rot90, -1)
                probs += probs_rot90
                probs_count += 1
            
            if self.mirror_config.get('mirror_all_dimensions'):
                mirror_axes = [1, 2]
            else:
                mirror_axes = self.mirror_config.get('mirror_axes', [1, 2])
            mirror_axes_comb = [[]]
            for ax in mirror_axes:  # get powerset
                mirror_axes_comb += [sub + [ax] for sub in mirror_axes_comb]
            for axis in mirror_axes_comb[1:]:
                images_mirror = tf.reverse(images, axis=axis)
                logits_mirror = self(images_mirror, training=False)
                if deep_supervision:
                    probs_mirror = tf.nn.softmax(logits_mirror[0])
                else:
                    probs_mirror = tf.nn.softmax(logits_mirror)
                probs_mirror = tf.reverse(probs_mirror, axis=axis)
                probs += probs_mirror
                probs_count += 1
                if mirror_config.get('rot90', False):
                    images_mirror_rot90 = tf.image.rot90(images_mirror)
                    logits_mirror_rot90 = self(images_mirror_rot90, training=False)
                    if deep_supervision:
                        probs_mirror_rot90 = tf.nn.softmax(logits_mirror_rot90[0])
                    else:
                        probs_mirror_rot90 = tf.nn.softmax(logits_mirror_rot90)
                    probs_mirror_rot90 = tf.image.rot90(probs_mirror_rot90, -1)
                    probs_mirror_rot90 = tf.reverse(probs_mirror_rot90, axis=axis)
                    probs += probs_mirror_rot90
                    probs_count += 1

        probs = probs / probs_count
        return probs
    
    def test(self, testing_paths, output_path, **kwargs):
        if not self._loaded:
            raise Exception('No model is found, please train first')
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        track_time = False
        if 'track_time' in kwargs:
            track_time = True
            total_time = 0
        
        for input_file in testing_paths:
            if track_time:
                output_labels, time_per_file = self.run_test(input_file, track_time=True)
                total_time += time_per_file
            else:
                output_labels = self.run_test(input_file)
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file))
            with h5py.File(output_file, 'w') as f_h5:
                f_h5['predictions'] = output_labels.astype(np.uint8)
                
        if track_time:
            print('Total time: ' + str(total_time))
                
    def run_test(self, input_file, track_time=False):
        all_images, info = self.read_testing_inputs(input_file, self.im_size)
        all_probs = np.zeros(list(all_images.shape[:-1]) + [self.nclass], np.float32)
        if track_time:
            lapsed_time = 0
        for n in range(all_images.shape[0]):
            # Avoid memory leakage. 
            # Converting the numpy array to a tensor maintains the same signature and avoids creating new graphs.
            tensor_im = tf.convert_to_tensor(all_images[n : n + 1, ...], dtype=tf.float32)
            if track_time:
                start_time = time.time()
            all_probs[n : n + 1] = self.predict_on_batch(tensor_im)
            if track_time:
                lapsed_time += time.time() - start_time
        output_labels = np.argmax(all_probs, axis=-1)
        if output_labels.shape[0] == 1:
            output_labels = np.squeeze(output_labels, axis=0)
            
        if track_time:
            return output_labels, lapsed_time
        else:
            return output_labels
    
    def test_tflite(self, model_path, testing_paths, output_path, **kwargs):
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        quant = (input_details['dtype'] == np.uint8)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        track_time = False
        if 'track_time' in kwargs:
            track_time = True
            total_time = 0

        for input_file in testing_paths:
            all_images, info = self.read_testing_inputs(input_file, self.im_size)
            all_probs = np.zeros(list(all_images.shape[:-1]) + [self.nclass], np.float32)
            
            for n in range(all_images.shape[0]):
                if quant:
                    input_scale, input_zero_point = input_details['quantization']
                    tensor_im = all_images[n : n + 1, ...] / input_scale + input_zero_point
                    tensor_im = tensor_im.astype(input_details['dtype'])
                else:
                    tensor_im = all_images[n : n + 1, ...]
                    
                if track_time:
                    start_time = time.time()
                interpreter.set_tensor(input_details['index'], tensor_im)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])[0]
                if quant:
                    output = tf.cast(output, tf.float32)
                output = tf.nn.softmax(output, axis=-1)
                if track_time:
                    total_time += time.time() - start_time
                all_probs[n : n + 1] = output
                
            output_labels = np.argmax(all_probs, axis=-1)
            if output_labels.shape[0] == 1:
                output_labels = np.squeeze(output_labels, axis=0)
                
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file))
            if quant:
                output_file = output_file.replace('.hdf5', '_uint8_tflite.hdf5')
            else:
                output_file = output_file.replace('.hdf5', '_float32_tflite.hdf5')
            with h5py.File(output_file, 'w') as f_h5:
                f_h5['predictions'] = output_labels.astype(np.uint8)
                
        if track_time:
            print('Total time: ' + str(total_time))
            
    def test_tflite_cplusplus(self, model_path, exec_path, testing_paths, output_path, quant, **kwargs):
        # "exec <tflite model> <data file> <outputfile> <timelog.txt>");
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        track_time = False
        if 'track_time' in kwargs:
            track_time = True
            total_time = 0

        for input_file in testing_paths:
            all_images, info = self.read_testing_inputs(input_file, self.im_size)  
            output_labels = np.empty(all_images.shape[:-1], dtype=np.uint8)
            for n in range(all_images.shape[0]):
                slice_data = all_images[n : n + 1]
                with open('input.dat', 'wb') as f:
                    slice_data.tofile(f)
                    
                command = exec_path + ' ' + model_path + ' input.dat output.dat timelog.txt 0'
                subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
                
                if quant:
                    out = np.fromfile('output.dat', dtype=np.uint8).reshape(self.im_size)
                else:
                    out = np.fromfile('output.dat', dtype=np.float32).reshape(self.im_size)
                    
                output_labels[n] = out
                with open('timelog.txt', 'r') as f:
                    line = f.readline()
                    total_time += int(line[20:].replace('ms', '')) / 1000.0
                
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file))
            if quant:
                output_file = output_file.replace('.hdf5', '_uint8_c_tflite.hdf5')
            else:
                output_file = output_file.replace('.hdf5', '_float32_c_tflite.hdf5')
            with h5py.File(output_file, 'w') as f_h5:
                f_h5['predictions'] = output_labels.astype(np.uint8)
        os.remove('input.dat')
        os.remove('output.dat')
        os.remove('timelog.txt')
                
        if track_time:
            print('Total time: ' + str(total_time))
        

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
                self.rois = validation_config['rois']
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
                if self.custom_log_file is not None:
                    with open(self.custom_log_file, 'a') as f:
                        writer = csv.writer(f, delimiter=';')
                        writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'),
                             'epoch: %d, dice: %.4f, loss: %.4f' % (epoch + 1, record_logs['dice'], record_logs['loss'])])
            self.logs = {}
            # Save / validate
            if self.counter % self.period == 0:
                self.save(epoch + 1)
                if self.validation_config is not None:
                    test_dice_mean = self.validation_fn(self.validation_paths)
                    with self.test_writer.as_default():
                        for ir, roi in enumerate(self.rois):
                            tf.summary.scalar(f'dice_{roi}', test_dice_mean[ir], step=epoch + 1)
                            self.test_writer.flush()