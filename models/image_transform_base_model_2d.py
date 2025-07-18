import os, h5py, glob, re, shutil, multiprocessing
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

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


class ImageTransformBaseModel2D(tf.keras.models.Model):
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

    def perform_augmentation(self, images, patch_size):
        AUGMENTATION_PARAMS.update(self.augmentation_params)
        if self.mirror_config is not None and self.mirror_config.get('training_mirror', False) and self.mirror_config.get('rot90', False):
            # Random rotate 0, 90, 180, 270
            k_rot90 = np.random.randint(4)
            images = np.rot90(images, k_rot90)
        
        if len(images.shape) == 3:
            images_aug = np.expand_dims(np.transpose(images, (2, 0, 1)), axis=0)
        else:
            images_aug = np.expand_dims(images, axis=(0, 1))
        
        images_aug, _ = augment_spatial_2(images_aug, None, patch_size=patch_size, 
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
        
        if self.mirror_config is not None and self.mirror_config.get('training_mirror', False):
            # Axes starts with batch from config
            # e.g. [1, 2] in mirror_config -> [0, 1] in augment_mirroring
            if self.mirror_config.get('mirror_all_dimensions'):
                mirror_axes = [1, 2]
            else:
                mirror_axes = self.mirror_config.get('mirror_axes', [1, 2])
            mirror_axes = [ax - 1 for ax in mirror_axes]
            images_aug, _ = augment_mirroring(images_aug, None, mirror_axes)
        
        # images_aug = augment_gaussian_noise(images_aug, p_per_sample=0.1)
        # images_aug = augment_gaussian_blur(images_aug, (0.5, 1.), per_channel=True, p_per_sample=0.2, p_per_channel=0.5)
        # images_aug = augment_brightness_multiplicative(images_aug, multiplier_range=(0.75, 1.25), p_per_sample=0.15)
        # images_aug = augment_contrast(images_aug, p_per_sample=0.15)
        # images_aug = augment_linear_downsampling_scipy(images_aug, zoom_range=(0.5, 1), per_channel=True,
        #                                                p_per_channel=0.5,
        #                                                order_downsample=0, order_upsample=3, p_per_sample=0.25,
        #                                                ignore_axes=None)
        # images_aug = augment_gamma(images_aug, AUGMENTATION_PARAMS.get('gamma_range'), invert_image=True, per_channel=True,
        #                            retain_stats=AUGMENTATION_PARAMS.get('gamma_retain_stats'), p_per_sample=0.1)

        # images_aug = augment_gamma(images_aug, AUGMENTATION_PARAMS.get('gamma_range'), invert_image=False, per_channel=True, 
        #                            retain_stats=AUGMENTATION_PARAMS.get('gamma_retain_stats'),
        #                            p_per_sample=AUGMENTATION_PARAMS['p_gamma'])
        
        if len(images.shape) == 3:
            images_aug = np.transpose(images_aug, (1, 2, 0))
        else:
            images_aug = np.squeeze(images_aug, axis=0)
        return images_aug
    
    def get_training_patch(self, images, full_size, im_size, enlarged_im_size, augmentation=True):
        full_size = list(full_size)
        # Pad
        pad = np.array(enlarged_im_size) + 1 - np.array(full_size)
        pad = np.clip(pad, 0, None)
        pad_with = tuple(zip(pad // 2, pad - pad // 2)) + ((0, 0),)
        if pad_with != ((0, 0), (0, 0), (0, 0)):
            images = np.pad(images, pad_with, mode='constant')
        
        full_size = list(np.maximum(full_size, enlarged_im_size))
        
        x_range = max(full_size[0] - enlarged_im_size[0], 1)
        y_range = max(full_size[1] - enlarged_im_size[1], 1)

        x_offset = int(enlarged_im_size[0] / 2)
        y_offset = int(enlarged_im_size[1] / 2)
        
        if self.sampling_config is not None and 'fg_range' in self.sampling_config:
            fg_low, fg_high = self.sampling_config['fg_range']
            labels = np.logical_and(images[..., 0] >= fg_low, images[..., 0] <= fg_high)
        else:
            labels = np.ones(full_size)
        la = labels[x_offset : x_offset + x_range, y_offset : y_offset + y_range]
        
        # Normalize images
        if self.norm_config['norm']:
            eps = 1e-7
            if len(images.shape) == 3:
                if self.norm_config['norm_channels'] == 'rgb_channels':
                    rgb_mean = np.array(self.norm_config.get('norm_mean', [0.485, 0.456, 0.406]))
                    rgb_std = np.array(self.norm_config.get('norm_std', [0.229, 0.224, 0.225]))
                    images = (images - rgb_mean) / rgb_std
                elif self.norm_config['norm_channels'] == 'all_channels':
                    images = (images - np.mean(images, axis=(0, 1))) / np.clip(np.std(images, axis=(0, 1)), eps, None)
                else:
                    for channel in self.norm_config['norm_channels']:
                        m = np.mean(images[..., channel])
                        s = np.clip(np.std(images[..., channel]), eps, None)
                        images[..., channel] = (images[..., channel] - m) / s
            else:
                images = (images - np.mean(images)) / np.clip(np.std(images), eps, None)
        
        # Get sampling prob
        # Random sampling ? of the time and always choose fg the rest of the time
        if np.random.random() > self.fg_sampling_ratio:
            # choose random
            o = np.random.choice(x_range * y_range)
        else:
            p = np.zeros((x_range, y_range), dtype=np.float32)
            if np.amax(la) > 0:
                p[la > 0] = 1
            else:
                # if foreground is not present (gives NaN value for p)
                p = np.ones((x_range, y_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range, p=p)    
        x_start, y_start = np.unravel_index(o, (x_range, y_range))
        
        images_extracted = images[x_start : x_start + enlarged_im_size[0], y_start : y_start + enlarged_im_size[1], ...]
        
        if augmentation:
            images_extracted = self.perform_augmentation(images_extracted, enlarged_im_size)

        x_border_width = int((enlarged_im_size[0] - im_size[0]) / 2)
        y_border_width = int((enlarged_im_size[1] - im_size[1]) / 2)

        images_extracted = images_extracted[x_border_width : x_border_width + im_size[0], 
                                            y_border_width : y_border_width + im_size[1], ...]
        
        return images_extracted
    
    def mixup_two_arrays(self, array1, array2, num_holes_range=(1, 4), hole_height_range=(64, 128), hole_width_range=(64, 128)):
        range1 = [np.amin(array1), np.amax(array1)]
        range2 = [np.amin(array2), np.amax(array2)]
        normed_array1 = (array1 - range1[0]) / (range1[1] - range1[0] + 1e-7)
        normed_array2 = (array2 - range2[0]) / (range2[1] - range2[0] + 1e-7)
        mixup_array = normed_array1
        min_holes, max_holes = num_holes_range
        min_height, max_height = hole_height_range
        min_width, max_width = hole_width_range
        holes = np.random.randint(min_holes, max_holes + 1)
        height, width = mixup_array.shape[:2]

        for _ in range(holes):
            hole_height = np.random.randint(min_height, max_height + 1)
            hole_width = np.random.randint(min_width, max_width + 1)

            y = np.random.randint(0, height - hole_height)
            x = np.random.randint(0, width - hole_width)

            mixup_array[y:y + hole_height, x:x + hole_width] = normed_array2[y:y + hole_height, x:x + hole_width]
        
        mixup_array = mixup_array * (range1[1] - range1[0] + 1e-7) + range1[0]
        return mixup_array
    
    def read_training_inputs(self, file, im_size, enlarged_im_size, augmentation=True):
        to_simulate = False
        input_images = None
        input_images_mask = None
        output_images = None
        output_images_mask = None
        
        with h5py.File(file, 'r') as f_h5:
            if 'input_images' in f_h5.keys():
                input_images = np.asarray(f_h5['input_images'], dtype=np.float32)
                if 'input_images_mask' in f_h5.keys():
                    input_images_mask = np.asarray(f_h5['input_images_mask'], dtype=np.uint8)
            
            if 'output_images' in f_h5.keys():
                output_images = np.asarray(f_h5['output_images'], dtype=np.float32)
                if 'output_images_mask' in f_h5.keys():
                    output_images_mask = np.asarray(f_h5['output_images_mask'], dtype=np.uint8)
            
            # Only have one, simulate input_images later
            if input_images is None and output_images is not None:
                input_images = np.copy(output_images)
                to_simulate = True
            elif output_images is None and input_images is not None:
                output_images = np.copy(input_images)
                to_simulate = True
        
        mask = None
        if input_images_mask is not None and output_images_mask is not None:
            # Use the pixels that the two are the same as the mask
            mask = input_images_mask == output_images_mask
        
        # Our 2D image dim standard is [batch(always=1 / maybe missing), h, w, channel(maybe missing)]
        # In training, we random choose one slice in batch, and drop batch dim
        if len(input_images.shape) == 4:
            full_size = list(input_images.shape)[1:-1]
            sli = np.random.choice(input_images.shape[0])
            input_images = input_images[sli]
            output_images = output_images[sli]
            if mask is not None:
                mask = mask[sli]
        elif len(input_images.shape) == 3:
            if self.input_channels == input_images.shape[-1]:
                full_size = list(input_images.shape)[:-1]
            else:
                full_size = list(input_images.shape[1:])
                sli = np.random.choice(input_images.shape[0])
                input_images = input_images[sli]
                output_images = output_images[sli]
                if mask is not None:
                    mask = mask[sli]
        elif len(input_images.shape) == 2:
            full_size = list(input_images.shape)
        
        # Use output_images as input_images to avoid over-transform
        if self.add_identity_sample:
            if np.random.uniform() < self.identity_sampling_ratio:
                input_images = np.copy(output_images)
        
        if self.augmentation_params.get('do_mixup', False):
            p_mixup = self.augmentation_params.get('p_mixup', 0.0)
            if np.random.uniform() < p_mixup:
                input_images = self.mixup_two_arrays(
                    input_images, output_images, 
                    num_holes_range=(1, 4), 
                    hole_height_range=(int(full_size[0] * 0.2), int(full_size[0] * 0.5)), 
                    hole_width_range=(int(full_size[1] * 0.2), int(full_size[1] * 0.5))
                )
        
        # Simulate input images if not exist
        if to_simulate:
            # mean: 0.0; std: 0.0~0.2
            simulation_fn = lambda x: np.clip(x + np.random.normal(0.0, np.random.random() * 0.2, x.shape), 0, 1)
            if self.simulation_config is not None:
                simulation_fn = self.simulation_config.get('simulation_fn', simulation_fn)
            input_images = simulation_fn(input_images)
            # Noise2Noise
            # output_images = simulation_fn(output_images)
        
        if mask is None:
            # if pad or rotate out, the outside area will not be counted in
            mask = np.ones(full_size, dtype=np.float32)
        
        # Adjust shape to uniform format (h, w, channel)
        if len(input_images.shape) == 2:
            input_images = input_images[..., None]
        if len(output_images.shape) == 2:
            output_images = output_images[..., None]
        if len(mask.shape) == 2:
            mask = mask[..., None]
        images = np.concatenate([input_images, output_images, mask], axis=-1)
        
        images_extracted = self.get_training_patch(images, full_size, im_size, enlarged_im_size, augmentation=True)
        input_images_extracted = images_extracted[..., 0 : self.input_channels]
        output_images_extracted = images_extracted[..., self.input_channels : self.input_channels + self.output_channels]
        mask = (images_extracted[..., self.input_channels + self.output_channels : ] > 0.5).astype(np.float32)
        
        bin_weights = list(self.bin_weights)
        for i in range(len(bin_weights)):
            mask[(output_images_extracted >= i / len(bin_weights)) &
                 (output_images_extracted < (i + 1) / len(bin_weights))] *= bin_weights[i]
        if self.sampling_config is not None and 'threshold' in self.sampling_config:
            mask[np.abs(output_images_extracted - input_images_extracted) > self.sampling_config['threshold']] = 0
        
        return input_images_extracted, output_images_extracted, mask.astype(np.float32)
    
    def read_testing_inputs(self, file):
        images = None
        to_simulate = False
        
        with h5py.File(file, 'r') as f_h5:
            if 'input_images' in f_h5.keys():
                images = np.asarray(f_h5['input_images'], dtype=np.float32)
            
            if images is None and 'output_images' in f_h5.keys():
                images = np.asarray(f_h5['output_images'], dtype=np.float32)
                to_simulate = True
        
        if to_simulate:
            simulation_fn = lambda x: np.clip(x + np.random.normal(0.0, np.random.random() * 0.2, x.shape), 0, 1)
            if self.simulation_config is not None:
                simulation_fn = self.simulation_config.get('simulation_fn', simulation_fn)
            images = simulation_fn(images)
        
        # Our 2D image dim standard is [batch(always=1 / maybe missing), h, w, channel(maybe missing)]
        # In testing, we keep the batch dim
        if len(images.shape) == 4:
            full_size = list(images.shape)[1:-1]
        elif len(images.shape) == 3:
            if self.input_channels == images.shape[-1]:
                full_size = list(images.shape)[:-1]
                images = np.expand_dims(images, axis=0)
            else:
                full_size = list(images.shape[1:])
                images = np.expand_dims(images, axis=-1)
        elif len(images.shape) == 2:
            full_size = list(images.shape)
            images = np.expand_dims(np.expand_dims(images, axis=0), axis=-1)
        
        dividable_by = 2 ** self.layer_number
        pad_size = (int(np.ceil(full_size[0] / dividable_by) * dividable_by),
                    int(np.ceil(full_size[1] / dividable_by) * dividable_by))
        
        # Pad if full size is smaller than patch size
        # symetric padding is always better than one-side padding
        pads = [max(0, pad_size[ax] - full_size[ax]) for ax in range(len(full_size))]
        
        if np.any(np.array(pads) > 0):
            if len(images.shape) == 4:
                pad_with = ((0, 0), (pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2), (0, 0))
            else:
                pad_with = ((0, 0), (pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2))
            images = np.pad(images, pad_with, mode='constant')
        
        info = {
            'full_size': full_size,
            'pad_size': pad_size
        }
        
        all_images = images.copy()
        for sli in range(all_images.shape[0]):
            images = all_images[sli]
            # Normalize images
            if self.norm_config['norm']:
                eps = 1e-7
                if len(images.shape) == 3:
                    if self.norm_config['norm_channels'] == 'rgb_channels':
                        rgb_mean = np.array(self.norm_config.get('norm_mean', [0.485, 0.456, 0.406]))
                        rgb_std = np.array(self.norm_config.get('norm_std', [0.229, 0.224, 0.225]))
                        images = (images - rgb_mean) / rgb_std
                    elif self.norm_config['norm_channels'] == 'all_channels':
                        images = (images - np.mean(images, axis=(0, 1))) / np.clip(np.std(images, axis=(0, 1)), eps, None)
                    else:
                        for channel in self.norm_config['norm_channels']:
                            m = np.mean(images[..., channel])
                            s = np.clip(np.std(images[..., channel]), eps, None)
                            images[..., channel] = (images[..., channel] - m) / s
                else:
                    images = (images - np.mean(images)) / np.clip(np.std(images), eps, None)
            all_images[sli] = images
        return all_images, info
    
    def train_data_mapper(self, i):
        input_patch, target_patch, mask = self.read_training_inputs(
            self.training_paths[i], self.im_size, self.enlarged_im_size)
        return input_patch, target_patch, mask
    
    def process_data_batch(self, q, idx_list):
        while True:
            shuffle_list = np.random.permutation(idx_list)
            input_batch = [None for _ in range(self.batch_size)]
            target_batch = [None for _ in range(self.batch_size)]
            mask_batch = [None for _ in range(self.batch_size)]
            ib = 0
            idx = 0
            while ib < self.batch_size and idx < len(idx_list):
                i = shuffle_list[idx]
                idx += 1
                input_batch[ib], target_batch[ib], mask_batch[ib] = self.train_data_mapper(i)
                ib += 1
            # Drop remainder
            if ib < self.batch_size:
                continue
            input_batch = np.stack(input_batch, axis=0)
            target_batch = np.stack(target_batch, axis=0)
            mask_batch = np.stack(mask_batch, axis=0)
            q.put((input_batch, target_batch, mask_batch))

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
                input_batch, target_batch, mask_batch = stuff
                it += 1
                yield input_batch, target_batch, mask_batch
            except:
                break
        pool.close()
    
    def train(self, run_validation=False, validation_paths=None, **kwargs):
        # Compile model for training
        self.compile_it()
        
        log_dir = os.path.join(self.log_dir, self.model_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=0)
        
        validation_config = None
        if run_validation:
            validation_config = {
                'validation_paths': validation_paths,
                'validation_fn': self.validate,
                'log_dir': log_dir,
            }
        saver_config = {
            'period': self.save_period,
            'save_fn': self.save,
            'log_dir': log_dir,
        }
        saver_callback = self.ModelSaver(saver_config=saver_config, validation_config=validation_config)
        
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
        return
    
    def validate(self, validation_paths):
        def keep_sli_helper(input_images):
            keep_sli = []
            for sli in range(input_images.shape[0]):
                if len(np.unique(input_images[sli])) == 1:
                    keep_sli.append(False)
                else:
                    keep_sli.append(True)
            return keep_sli
        def calculate_metric(a, b, metric, mask=None):
            diff = a - b
            if mask is None:
                mask = np.ones(a.shape, dtype=np.bool)
            if metric == 'sos':
                sum_of_squares = (a - b) * (a - b)
                res = np.sqrt(np.mean(sum_of_squares[mask]))
            elif metric == 'mae':
                res = np.mean(np.abs(diff)[mask])
            elif metric == 'ssim':
                from skimage.metrics import structural_similarity as ssim
                res = ssim(a, b)
            return res
        mae_before = []
        mae_after = []
        ssim_before = []
        ssim_after = []
        pbar = tqdm(validation_paths)
        for case in pbar:
            output_images = self.run_test(case)
            with h5py.File(case, 'r') as f_h5:
                #input_images = np.asarray(f_h5['input_images'], dtype=np.float32)
                output_images = np.asarray(f_h5['output_images'], dtype=np.float32)

            # input_images = (input_images - input_images.min()) / (input_images.max() - input_images.min())
            # output_images = np.clip(input_images + output_images, 0., 1.)
            # keep_sli = keep_sli_helper(input_images)
            # input_images = input_images[keep_sli]
            # output_images = output_images[keep_sli]
            mask = None
            if self.sampling_config is not None and 'threshold' in self.sampling_config:
                mask = np.abs(input_images - output_images) < self.sampling_config['threshold']
            # mae_before.append(calculate_metric(input_images, output_images, 'mae', mask))
            mae_after.append(calculate_metric(output_images, output_images, 'mae', mask))
            # ssim_before.append(calculate_metric(input_images, output_images, 'ssim'))
            ssim_after.append(calculate_metric(output_images, output_images, 'ssim'))
            pbar.set_postfix({'val_scores': [np.mean(mae_after), np.mean(ssim_after)]})
        return {'mae': np.mean(mae_after), 'ssim': np.mean(ssim_after)}
    
    # can be overwritten by inherited model
    @tf.function
    def train_step(self, data):
        input_patch, target_patch, mask = data
        
        with tf.GradientTape() as tape:
            output = self.unet(input_patch, training=True)
            loss = self.loss_fn(target_patch, output, mask)
            
        # Get the gradients
        gradient = tape.gradient(loss, self.unet.trainable_variables)
        # Update the weights
        self.optimizer.apply_gradients(zip(gradient, self.unet.trainable_variables))

        return {'loss': loss}
    
    @tf.function
    def predict_step(self, data):
        # patch  ##NOTE: in tf>=2.3.0, images=data; in tf==2.2.0, images=data[0]
        if isinstance(data, tuple):
            images = data[0]
        else:
            images = data

        outputs = self(images, training=False)
        outputs_count = 1

        tta = getattr(self,'tta', {})
        nrotation = getattr(tta, 'nrotation', 0)
        if nrotation > 0:
            for rot in range(nrotation):
                radian = rot * 3.14159265359 * 2 / nrotation
                images_rot = tfa.image.rotate(images, radian)
                outputs_rot = self(images_rot, training=False)
                outputs_rot = tfa.image.rotate(outputs_rot, -radian)
                outputs += outputs_rot
                outputs_count += 1

        mirror_config = getattr(self, 'mirror_config', {})
        if mirror_config.get('testing_mirror', False):
            if mirror_config.get('rot90', False):
                images_rot90 = tf.image.rot90(images)
                outputs_rot90 = self(images_rot90, training=False)
                outputs_rot90 = tf.image.rot90(outputs_rot90, -1)
                outputs += outputs_rot90
                outputs_count += 1
            if self.mirror_config.get('mirror_all_dimensions'):
                mirror_axes = [1, 2]
            else:
                mirror_axes = self.mirror_config.get('mirror_axes', [1, 2])
            mirror_axes_comb = [[]]
            for ax in mirror_axes:  # get powerset
                mirror_axes_comb += [sub + [ax] for sub in mirror_axes_comb]
            for axis in mirror_axes_comb[1:]:
                images_mirror = tf.reverse(images, axis=axis)
                outputs_mirror = self(images_mirror, training=False)
                outputs_mirror = tf.reverse(outputs_mirror, axis=axis)
                outputs += outputs_mirror
                outputs_count += 1
                if mirror_config.get('rot90', False):
                    images_mirror_rot90 = tf.image.rot90(images_mirror)
                    outputs_mirror_rot90 = self(images_mirror_rot90, training=False)
                    outputs_mirror_rot90 = tf.image.rot90(outputs_mirror_rot90, -1)
                    outputs_mirror_rot90 = tf.reverse(outputs_mirror_rot90, axis=axis)
                    outputs += outputs_mirror_rot90
                    outputs_count += 1

        outputs = outputs / outputs_count
        #outputs = tf.clip_by_value(images + outputs, 0., 1.)
        return outputs

    def test(self, testing_paths, output_path, **kwargs):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        for input_file in testing_paths:
            with h5py.File(input_file, 'r') as f_h5:
                if 'input_images' in f_h5.keys():
                    input_images = np.asarray(f_h5['input_images'], dtype=np.float32)
                else:
                    continue
            output_images = self.run_test(input_file)
                
            if output_path is not None:
                with h5py.File(os.path.join(output_path, os.path.basename(input_file)), 'w') as f_h5:
                    f_h5['output_images'] = output_images
    
    def run_test(self, input_file):
        all_images, info = self.read_testing_inputs(input_file)
        all_probs = []
        for n in range(all_images.shape[0]):
            # Avoid memory leakage. 
            # Converting the numpy array to a tensor maintains the same signature and avoids creating new graphs.
            tensor_im = tf.convert_to_tensor(all_images[n : n + 1, ...], dtype=tf.float32)
            all_probs.append(self.predict_on_batch(tensor_im))
        output_labels = np.concatenate(all_probs, axis=0)
        
        full_size = info['full_size']
        pad_size = info['pad_size']
        pads = [pad_size[ax] - full_size[ax] for ax in range(len(full_size))]
        output_labels = output_labels[:, pads[0] // 2 : pads[0] // 2 + full_size[0],
                                      pads[1] // 2 : pads[1] // 2 + full_size[1], ...]
        
        if output_labels.shape[0] == 1:
            output_labels = np.squeeze(output_labels, axis=0)
        if output_labels.shape[-1] == 1:
            output_labels = np.squeeze(output_labels, axis=-1)
        return output_labels
    
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
                    val_scores = self.validation_fn(self.validation_paths)
                    with self.test_writer.as_default():
                        for metric in val_scores:
                            tf.summary.scalar(metric, val_scores[metric], step=epoch + 1)
                        self.test_writer.flush()