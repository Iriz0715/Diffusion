import os, h5py, glob, re, shutil, multiprocessing
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import csv, datetime
import matplotlib.pyplot as plt 

from augmentation import *
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from skimage.transform import resize
import time
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

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


class DDPMBaseModel2D(tf.keras.models.Model):

    # With this part, fit() can read our custom data generator
    def call(self, x):
        return self.unet(x)

    def save(self, step, max_to_keep=1):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if len(ckpt_files) >= max_to_keep:    # 自动删除多余旧文件
            os.remove(ckpt_files[0])


        self.unet.save(os.path.join(checkpoint_dir, 'model_epoch%06d.h5' % step))


    def load(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            print('No model is found, please train first')
            return False, 0
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch*.h5')))
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
    
    def get_training_patch(self, images, full_size, im_size, enlarged_im_size, augmentation=False):
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
                input_images = input_images * 2. - 1.  ############ Scale to [-1, 1]
                if 'input_images_mask' in f_h5.keys():
                    input_images_mask = np.asarray(f_h5['input_images_mask'], dtype=np.uint8)
            
            if 'output_images' in f_h5.keys():
                output_images = np.asarray(f_h5['output_images'], dtype=np.float32)
                output_images = output_images * 2. - 1.  ############### Scale to [-1, 1]
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
        input_images_extracted = images_extracted[..., 0 : self.input_channels//2]      # input_channels =2
        output_images_extracted = images_extracted[..., self.input_channels//2 : self.input_channels//2 + self.output_channels]
        mask = (images_extracted[..., self.input_channels//2 + self.output_channels : ] > 0.5).astype(np.float32)
        
        bin_weights = list(self.bin_weights)
        for i in range(len(bin_weights)):
            mask[(output_images_extracted >= i / len(bin_weights)) &
                 (output_images_extracted < (i + 1) / len(bin_weights))] *= bin_weights[i]
        if self.sampling_config is not None and 'threshold' in self.sampling_config:
            mask[np.abs(output_images_extracted - input_images_extracted) > self.sampling_config['threshold']] = 0
        
        return input_images_extracted, output_images_extracted, mask.astype(np.float32)
    
    def read_testing_inputs(self, file):
        images = None
        targets = None
        to_simulate1 = False
        to_simulate2 = False
        
        with h5py.File(file, 'r') as f_h5:
            if 'input_images' in f_h5.keys():
                images = np.asarray(f_h5['input_images'], dtype=np.float32)
            if images is None and 'output_images' in f_h5.keys():
                images = np.asarray(f_h5['output_images'], dtype=np.float32)
                to_simulate1 = True

            if 'output_images' in f_h5.keys():
                targets = np.asarray(f_h5['output_images'], dtype=np.float32)
            if targets is None and 'input_images' in f_h5.keys():
                targets = np.asarray(f_h5['input_images'], dtype=np.float32)
                to_simulate2 = True
        
        if to_simulate1:   # simulate input images if not exist -> 对input images加噪并裁剪至[0,1]
            simulation_fn = lambda x: np.clip(x + np.random.normal(0.0, np.random.random() * 0.2, x.shape), 0, 1)
            if self.simulation_config is not None:
                simulation_fn = self.simulation_config.get('simulation_fn', simulation_fn)
            images = simulation_fn(images)

        if to_simulate2:   # simulate output images if not exist -> 对output images加噪并裁剪至[0,1]
            simulation_fn = lambda x: np.clip(x + np.random.normal(0.0, np.random.random() * 0.2, x.shape), 0, 1)
            if self.simulation_config is not None:
                simulation_fn = self.simulation_config.get('simulation_fn', simulation_fn)
            targets = simulation_fn(targets)

        images = images * 2. - 1.  ############## Scale to [-1, 1]
        targets = targets * 2. - 1. ############# Scale to [-1, 1]

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
        all_targets = targets.copy()
        for sli in range(all_images.shape[0]):
            images = all_images[sli]
            targets = all_targets[sli]
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
            all_targets[sli] = targets
        return all_images, all_targets, info    # [d, h, w, c], [d, h, w]

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
    






    


    # @staticmethod
    # # resize
    # def pad_and_resize(image, target_size=(256, 256)):
    #   h, w = image.shape
    #   pad_h = max(target_size[0] - h, 0)
    #   pad_w = max(target_size[1] - w, 0)
    #   pad_with = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    #   image_padded = np.pad(image, pad_with, mode='constant')
    #   image_resized = resize(image_padded, target_size, preserve_range=True, anti_aliasing=True).astype(np.float32)
    #   return image_resized

    @staticmethod
    def crop_pad2D(x, target_size, shift=[0, 0]):
        'crop or zero-pad the 2D volume to the target size'
        x = np.asarray(x)
        small = 0
        y = np.ones(target_size, dtype=np.float32) * small
        current_size = x.shape
        pad_size = [0, 0]
        # print('current_size:',current_size)
        # print('pad_size:',target_size)
        for dim in range(2):
            if current_size[dim] > target_size[dim]:
                pad_size[dim] = 0
            else:
                pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))
        # pad first
        x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]]], 'constant', constant_values=small)
        # crop on x1
        start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)
        start_pos = start_pos.astype(int)
        y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
              (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1])]
        return y


    ## 一次性读取所有数据，容易OOM，暂时不用
    def build_all_slice_dataset(self, folder_path, target_size=(256, 256), batch_size=8, shuffle=True, slice_mode='all', data_mode='train'):
        """
        一次性读取所有hdf5文件的所有slice，返回tf.data.Dataset
        Args:
            folder_path: hdf5文件夹路径（支持str或list）
            target_size: resize目标尺寸
            shuffle: 是否打乱
            batch_size: 批大小
        Returns:
            tf.data.Dataset
        """
        # 收集所有slice
        input_images = []
        output_images = []
        all_files = []
        if isinstance(folder_path, (list, tuple)):
            for single_folder in folder_path:
                if isinstance(single_folder, (list, tuple)):
                    # 防止嵌套list
                    for sub_folder in single_folder:
                        all_files.extend(sorted(glob.glob(os.path.join(str(sub_folder)))))
                else:
                    all_files.extend(sorted(glob.glob(os.path.join(str(single_folder)))))
        else:
            all_files = sorted(glob.glob(os.path.join(str(folder_path))))
        file_list = all_files
        if data_mode == 'valid':
            file_list = file_list[:10]
        for file_path in file_list:
            with h5py.File(file_path, 'r') as f_h5:
                input_imgs = np.asarray(f_h5['input_images'], dtype=np.float32)
                output_imgs = np.asarray(f_h5['output_images'], dtype=np.float32)
                # input_images = input_images * 2. - 1.  ############ Scale to [-1, 1]
                # output_images = output_images * 2. - 1. ############ Scale to [-1, 1]
                if slice_mode == 'all':
                    idxs = range(input_imgs.shape[0])
                elif slice_mode == 'random':
                    idxs = [random.randint(0, input_imgs.shape[0] - 1)]
                elif slice_mode == 'center':
                    idxs = [input_imgs.shape[0] // 2]
                for i in idxs:
                    input_images.append(self.pad_and_resize(input_imgs[i], target_size))
                    output_images.append(self.pad_and_resize(output_imgs[i], target_size))
        input_images = np.array(input_images)
        output_images = np.array(output_images)
        input_images = np.expand_dims(input_images, axis=-1)
        output_images = np.expand_dims(output_images, axis=-1)

        # 构建 tf.data.Dataset
        ds = tf.data.Dataset.from_tensor_slices((input_images, output_images))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(input_images))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    ## 每次 batch 从磁盘读取数据，边读边处理
    def hdf5_slice_generator(self, folder_path, target_size, slice_mode='all', mode='train'):
        # 支持 folder_path 为 str 或 list/tuple
        all_files = []
        # hdf5 路径list
        if isinstance(folder_path, (list, tuple)):
            for single_folder in folder_path:
                if isinstance(single_folder, (list, tuple)):
                    # 防止嵌套list
                    for sub_folder in single_folder:
                        all_files.extend(sorted(glob.glob(os.path.join(str(sub_folder)))))
                else:
                    all_files.extend(sorted(glob.glob(os.path.join(str(single_folder)))))
        else:
            if os.path.isdir(folder_path):
                # 文件夹，收集所有hdf5文件
                all_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.hdf5"))))
            elif os.path.isfile(folder_path) and folder_path.endswith('.hdf5'):
                # 单个hdf5文件
                all_files.append(folder_path)

        for file_idx, file_path in enumerate(all_files):
            with h5py.File(file_path, 'r') as f_h5:
                input_img = np.asarray(f_h5['input_images'], dtype=np.float32)
                output_img = np.asarray(f_h5['output_images'], dtype=np.float32)
                full_size = input_img.shape[1:]  # 原始大小 [h,w]
                # input_img = input_img * 2. - 1.  ############ Scale to [-1, 1]
                # output_img = output_img * 2. - 1. ############ Scale to [-1, 1]
                if slice_mode == 'all':
                    idxs = range(input_img.shape[0])
                elif slice_mode == 'random':
                    idxs = [np.random.randint(0, input_img.shape[0])]
                elif slice_mode == 'center':
                    idxs = [input_img.shape[0] // 2]

                for i in idxs:
                    if mode == 'train': # sample patch
                      input_img_ch = np.expand_dims(input_img[i], axis=-1)
                      output_img_ch = np.expand_dims(output_img[i], axis=-1)
                      images = np.concatenate([input_img_ch, output_img_ch], axis=-1)
                      images_extracted = self.get_training_patch(images, full_size, im_size=target_size, enlarged_im_size=self.enlarged_im_size)
                      input_images_extracted = images_extracted[..., 0 : self.input_channels//2]      # input_channels =2
                      output_images_extracted = images_extracted[..., self.input_channels//2 : self.input_channels//2 + self.output_channels]
                      yield input_images_extracted, output_images_extracted, file_idx
                    if mode=='test' or mode=='valid': # center crop
                      input_img_crop = self.crop_pad2D(x=input_img[i], target_size=target_size)
                      output_img_crop = self.crop_pad2D(x=output_img[i], target_size=target_size)
                      input_img_crop = np.expand_dims(input_img_crop, axis=-1)
                      output_img_crop = np.expand_dims(output_img_crop, axis=-1)
                      yield input_img_crop, output_img_crop, file_idx



    def get_tf_dataset(self, folder_path, target_size, slice_mode='all', mode='train', batch_size=8, shuffle=True):
        output_types = (tf.float32, tf.float32, tf.int32)
        output_shapes = ((target_size[0], target_size[1], 1), (target_size[0], target_size[1], 1), ())
        ds = tf.data.Dataset.from_generator(
            lambda: self.hdf5_slice_generator(folder_path, target_size, slice_mode, mode),
            output_types=output_types,
            output_shapes=output_shapes
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    ############ 20250724
    def diffusion_train(self, run_validation=False, validation_paths=None, **kwargs):
        """
        完善的扩散模型训练方法，参考 DDPM_base_model_2d.train()
        
        Args:
            run_validation: 是否运行验证
            validation_paths: 验证集文件路径列表
        """
        print("开始扩散模型训练...")
        # Compile model for training
        self.compile_it()
        log_dir = os.path.join(self.log_dir, self.model_dir)
        
        validation_config = None
        if run_validation:
            validation_config = {
                'validation_paths': validation_paths,
                'validation_fn': self.validate_diffusion,
                'log_dir': log_dir,
            }
        saver_config = {
            'period': self.save_period,
            'save_fn': self.save,
            'log_dir': log_dir,
        }
        saver_callback = self.DiffusionModelSaver(saver_config=saver_config, validation_config=validation_config)
        
        # Prepare data generator
        num_samples = len(self.training_paths)
        if num_samples == 0:
            print('No training data')
            return
        
        # 使用当前的计数器值作为训练起点
        print(f'Steps per epoch: {self.steps_per_epoch}')
        print(f'Training from epoch {self.counter} to {self.epoch}')

        idx_list = np.arange(num_samples)
        total_iters = self.steps_per_epoch * (self.epoch - self.counter)
        # data_generator = self.data_generator(idx_list, total_iters)
        print('Loading training data...')
        train_data = self.get_tf_dataset(self.training_paths, target_size=self.im_size,
                                           batch_size=self.batch_size, shuffle=True,
                                           slice_mode='all', mode='train')
        # train_data = self.build_all_slice_dataset(self.training_paths, target_size=self.im_size,
        #                                      batch_size=self.batch_size, shuffle=True, per_slice=True, mode='train')
        print('Training data loaded successfully.')
        tr_ls = []  # training loss history
        early_stop_patience = 50
        early_stop_delta = 0.01  # 1%
        min_epoch = 150
        early_stop_flag = False
        ## load loss history
        if self.resume == True:        
            readmat = sio.loadmat('/mnt/newdisk/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/Loss/ddpm_unet.mat')
            load_tr_ls = readmat['loss']
            for i in range(self.counter):
                tr_ls.append(load_tr_ls[0][i])
            print('Finish loading loss!')
        
        ####################### train
        if total_iters > 0:
            print('Running on complete dataset with total training samples:', num_samples)
            # 创建回调列表（移除 TensorBoard 回调，使用手动日志写入）
            callbacks = [saver_callback]
            # 模拟 fit() 的回调生命周期
            # 1. on_train_begin
            for callback in callbacks:
                callback.on_train_begin()

            print(f">>>>>>>>>>> 训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")
            start_time_all = time.time()  # 记录总训练时间

            # 训练循环
            for epoch in range(self.counter, self.epoch):
                # 2. on_epoch_begin
                for callback in callbacks:
                    callback.on_epoch_begin(epoch)
                # 重置指标
                train_loss_metric = tf.keras.metrics.Mean()

                iter_times = [] # 记录每个iteration的耗时
                # 用tqdm进度条包裹batch循环
                with tqdm(total=self.steps_per_epoch, ncols=100, desc=f"Epoch {epoch+1}", leave=False, disable=True) as pbar:
                    for step_count, (input_batch, target_batch, _) in enumerate(train_data):
                        start_time = time.time()  # 记录开始时间
                        if step_count >= self.steps_per_epoch:
                            break
                        # 3. on_batch_begin
                        for callback in callbacks:
                            callback.on_batch_begin(step_count)
                        
                        # 执行训练步骤
                        # print('input_batch shape:', input_batch.shape,'target_batch shape:', target_batch.shape, 'mask_batch shape:', mask_batch.shape)
                        # (bs,256,256,1) (bs,256,256,1) (bs,256,256,1)
                        step_loss = self._diffusion_train_step(input_batch, target_batch)

                        # 更新指标
                        train_loss_metric.update_state(step_loss)        
                        # 获取当前学习率
                        if hasattr(self.optimizer.learning_rate, '__call__'):
                            current_step = epoch * self.steps_per_epoch + step_count
                            current_lr = float(self.optimizer.learning_rate(current_step))
                        else:
                            current_lr = float(self.optimizer.learning_rate)
                        # 准备批次日志
                        batch_logs = {
                            'loss': float(step_loss),
                            'lr': current_lr
                        }
                        # 4. on_batch_end
                        for callback in callbacks:
                            callback.on_batch_end(step_count, batch_logs)
                        # tqdm动态显示loss/lr
                        pbar.set_postfix({
                            'loss': f"{train_loss_metric.result().numpy():.4f}",
                            'lr': f"{current_lr:.2e}"
                        })
                        pbar.update(1)

                        iter_time = time.time() - start_time  # 计算本次iteration耗时
                        iter_times.append(iter_time)  # 记录
                # epoch 总耗时
                if iter_times:
                    avg_iter_time = sum(iter_times) / len(iter_times)
                # Epoch 结束处理
                epoch_loss = train_loss_metric.result()              
                # 准备 epoch 日志
                epoch_logs = {
                    'loss': float(epoch_loss),
                    'lr': current_lr
                }
                # TensorBoard 日志由 DiffusionModelSaver 回调统一管理
                print(f"Epoch {epoch + 1}/{self.epoch} - Loss: {epoch_loss:.6f}, LR: {current_lr:.8f}, avg_iter_time: {avg_iter_time:.4f}s")

                # 5. on_epoch_end：log & valid & save
                for callback in callbacks:
                    callback.on_epoch_end(epoch, epoch_logs)

                # save loss history
                tr_ls.append(epoch_loss)
                sio.savemat('/mnt/newdisk/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/Loss/' + 'ddpm_unet.mat', {'loss': tr_ls})

                # 保存最新模型（每个epoch覆盖一次）
                self.unet.save(os.path.join(self.checkpoint_dir, self.model_dir, 'model_latest.h5'))

                # Early stopping
                if epoch+1 >= min_epoch and len(tr_ls) >= early_stop_patience+1:
                    avg_recent = sum(tr_ls[-early_stop_patience-1:-1]) / early_stop_patience
                    if tr_ls[-1] >= avg_recent * (1 - early_stop_delta):
                        print(f"Early stopping at epoch {epoch+1}: loss did not decrease by 1% relative to previous 50 epochs average.")
                        early_stop_flag = True

                if early_stop_flag == True:
                    break
                
                # 重置度量
                train_loss_metric.reset_states()

            # 6. on_train_end
            for callback in callbacks:
                callback.on_train_end()

            # TensorBoard writer 由回调管理，无需手动关闭

            self.save(epoch+1, max_to_keep=5)  # 保存最后一个 epoch 的检查点，=model_latest，但id包含epoch易读取

            # 记录训练结束时间
            end_time_all = time.time()
            print(f">>>>>>>>>>> 训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_all))} <<<<<<<<<<<<")
            print(f"训练总时长: {(end_time_all - start_time_all)/60:.2f} 分钟")
  
        print("训练完成!")
    
    @tf.function
    def _diffusion_train_step(self, input_batch, target_batch):    # 有的loss用mask计算，这里不确定要不要
        """单个训练步骤"""
        with tf.GradientTape() as tape:
            loss = self.diffusion_trainer.forward(x_0=target_batch, context=input_batch)
            loss = tf.reduce_mean(loss)
        
        # 计算梯度并更新模型
        gradients = tape.gradient(loss, self.unet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        
        return loss
    

    # ############# 0725
    def validate_diffusion(self, validation_paths):
        val_dataset = self.get_tf_dataset(validation_paths, target_size=self.im_size, 
                                          batch_size=self.batch_size, shuffle=False, 
                                          slice_mode='center', mode='valid')
        train_dataset = self.get_tf_dataset('/mnt/newdisk/mri2ct/data_training/train_2', target_size=self.im_size, 
                                             batch_size=self.batch_size, shuffle=False, 
                                             slice_mode='center', mode='valid')
        
        # 按 file_idx 分组收集所有 slice 的 pred 和 gt
        pred_dict_train = {}
        gt_dict_train = {}
        for input_batch, target_batch, file_idx_batch in train_dataset:
            # input_batch, target_batch: [bs, h, w, ch], file_idx_batch: [bs]
            x_T = tf.random.normal(tf.shape(input_batch))
            x_0 = self.diffusion_sampler.ddim_reverse(x_T, context=input_batch)  # [bs, h, w, ch]
            pred_batch = x_0.numpy() if hasattr(x_0, 'numpy') else x_0
            gt_batch = target_batch.numpy() if hasattr(target_batch, 'numpy') else target_batch
            for i in range(input_batch.shape[0]):
                idx = int(file_idx_batch[i].numpy()) if hasattr(file_idx_batch[i], 'numpy') else int(file_idx_batch[i])
                pred_dict_train.setdefault(idx, []).append(pred_batch[i, ..., -1])  # 取最终预测
                gt_dict_train.setdefault(idx, []).append(gt_batch[i, ..., -1])
        # 计算每个case的3D MAE和SSIM
        mae_list_train = []
        ssim_list_train = []
        for idx in pred_dict_train:
            pred_stack = np.stack(pred_dict_train[idx], axis=0)  # [d, h, w]
            gt_stack = np.stack(gt_dict_train[idx], axis=0)      # [d, h, w]
            mae = np.mean(np.abs(gt_stack - pred_stack))
            ssim_val = ssim(pred_stack[0], gt_stack[0], data_range=1.0, channel_axis=None)
            mae_list_train.append(mae)
            ssim_list_train.append(ssim_val)
            print(f"Train Case {idx} - MAE: {mae:.4f}, SSIM: {ssim_val:.4f}")
        mean_mae = np.mean(mae_list_train)
        mean_ssim = np.mean(ssim_list_train)
        print(f"Train - MAE: {mean_mae:.4f}, SSIM: {mean_ssim:.4f}")


        # 按 file_idx 分组收集所有 slice 的 pred 和 gt
        pred_dict = {}
        gt_dict = {}
        for input_batch, target_batch, file_idx_batch in val_dataset:
            # input_batch, target_batch: [bs, h, w, ch], file_idx_batch: [bs]
            x_T = tf.random.normal(tf.shape(input_batch))
            x_0 = self.diffusion_sampler.ddim_reverse(x_T, context=input_batch)  # [bs, h, w, ch]
            pred_batch = x_0.numpy() if hasattr(x_0, 'numpy') else x_0
            gt_batch = target_batch.numpy() if hasattr(target_batch, 'numpy') else target_batch
            for i in range(input_batch.shape[0]):
                idx = int(file_idx_batch[i].numpy()) if hasattr(file_idx_batch[i], 'numpy') else int(file_idx_batch[i])
                pred_dict.setdefault(idx, []).append(pred_batch[i, ..., -1])  # 取最终预测
                gt_dict.setdefault(idx, []).append(gt_batch[i, ..., -1])
        # 计算每个case的3D MAE和SSIM
        mae_list = []
        ssim_list = []
        for idx in pred_dict:
            pred_stack = np.stack(pred_dict[idx], axis=0)  # [d, h, w]
            gt_stack = np.stack(gt_dict[idx], axis=0)      # [d, h, w]
            mae = np.mean(np.abs(gt_stack - pred_stack))
            ssim_val = ssim(pred_stack[0], gt_stack[0], data_range=1.0, channel_axis=None)
            mae_list.append(mae)
            ssim_list.append(ssim_val)
            print(f"Val Case {idx} - MAE: {mae:.4f}, SSIM: {ssim_val:.4f}")
        mean_mae = np.mean(mae_list)
        mean_ssim = np.mean(ssim_list)
        return {'avg mae': mean_mae, 'avg ssim': mean_ssim}


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


    class DiffusionModelSaver(ModelSaver):
        """
        扩散模型专用的模型保存回调，继承基类 ModelSaver 并扩展手动训练循环所需的功能
        """
        def __init__(self, saver_config, validation_config=None, custom_log_file=None):
            # 不调用父类初始化，避免创建重复的 writer
            self.counter = 0
            self.period = saver_config['period']
            self.save = saver_config['save_fn']
            self.validation_config = validation_config
            self.logs = {}
            self.custom_log_file = custom_log_file
            
            # 与 ModelSaver 保持一致，统一使用 'avg' 目录
            log_dir = saver_config.get('log_dir', 'logs')
            self.train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'avg'))
            
            if validation_config is not None:
                self.validation_paths = validation_config['validation_paths']
                self.validation_fn = validation_config['validation_fn']
                # 与 ModelSaver 保持一致，验证指标写入 test 目录
                self.test_writer = tf.summary.create_file_writer(os.path.join(validation_config['log_dir'], 'test'))
            else:
                self.test_writer = None
                
            print(f"DiffusionModelSaver initialized - save period: {self.period} epochs")
            print(f"TensorBoard logs will be saved to: {log_dir}/avg")
        
        def on_train_begin(self, logs=None):
            """训练开始时的处理"""
            print(f"Starting diffusion training with save period of {self.period} epochs")
        
        def on_epoch_begin(self, epoch, logs=None):
            """Epoch 开始时的处理"""
            pass
        
        def on_batch_begin(self, batch, logs=None):
            """批次开始时的处理"""
            pass
        
        def on_epoch_end(self, epoch, logs=None):
            """统一的 epoch 结束处理，写入 TensorBoard 和管理模型保存"""
            if logs is None:
                logs = {}
                
            self.counter += 1
            
            # 写入主要的训练指标到 TensorBoard
            with self.train_writer.as_default():
                # 写入当前 epoch 的 loss 和学习率
                if 'loss' in logs:
                    tf.summary.scalar('loss', logs['loss'], step=epoch)
                if 'lr' in logs:
                    tf.summary.scalar('learning_rate', logs['lr'], step=epoch)
                
                # 如果有批次数据，也写入批次平均值
                if self.logs:
                    for key in self.logs.keys():
                        if 'loss' in key:
                            # 对于 loss 指标，过滤掉 0 值并计算平均
                            non_zero_values = np.array(self.logs[key])[np.nonzero(self.logs[key])]
                            if len(non_zero_values) > 0:
                                batch_avg = non_zero_values.mean()
                                tf.summary.scalar(f'batch_avg_{key}', batch_avg, step=epoch)
                        else:
                            batch_avg = np.nanmean(self.logs[key])
                            tf.summary.scalar(f'batch_avg_{key}', batch_avg, step=epoch)
                
                self.train_writer.flush()
            
            # 记录到自定义日志文件（如果有）
            if self.custom_log_file is not None:
                import csv
                import datetime
                with open(self.custom_log_file, 'a') as f:
                    writer = csv.writer(f, delimiter=';')
                    loss_val = logs.get('loss', 0.0)
                    lr_val = logs.get('lr', 0.0)
                    writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'),
                                   f'epoch: {epoch + 1}, loss: {loss_val:.6f}, lr: {lr_val:.8f}'])
            
            # 重置批次日志
            self.logs = {}
            
            # 保存和验证
            if (epoch + 1) % self.period == 0:    # 原来按counter（0,1,2...)保存
                print(f"\nSaving checkpoint at epoch {epoch + 1}...")
                self.save(epoch + 1, max_to_keep=3)  # 保留3个最新模型
                print(f"Checkpoint saved at epoch {epoch + 1}")

            # 每20个epoch进行一次val
            if (epoch + 1) % 10 == 0:
              if self.validation_config is not None:
                  # print("Running validation...")
                  try:
                      val_scores = self.validation_fn(self.validation_paths)    
                      # 记录验证结果到 TensorBoard
                      if hasattr(self, 'test_writer') and self.test_writer is not None:
                          with self.test_writer.as_default():
                              for metric in val_scores:
                                  tf.summary.scalar(metric, val_scores[metric], step=epoch + 1)
                              self.test_writer.flush()
                          print(f"Validation results: {val_scores}")
                  except Exception as e:
                      print(f"Validation failed: {str(e)}")
              
        
        def on_train_end(self, logs=None):
            """训练结束时的处理，关闭 TensorBoard writers"""
            print("Training completed, closing TensorBoard writers...")
            if hasattr(self, 'train_writer') and self.train_writer is not None:
                self.train_writer.close()
            if hasattr(self, 'test_writer') and self.test_writer is not None:
                self.test_writer.close()
            print("TensorBoard writers closed.")
    
    ########### 0724 
    def diffusion_test_batch(self, testing_paths, output_path, batch_size=4, sampler='ddpm', **kwargs):
        """
        批量推理的diffusion test，仿照PyTorch test.py实现：
        - 支持batch推理
        - 结果保存为hdf5
        - 计算MAE/SSIM
        - 可选保存部分样本可视化
        """
        # 确保每次test结果一致
        np.random.seed(42)
        tf.random.set_seed(42)

        print("开始批量扩散模型测试...")
        _loaded, self.counter = self.load()
        if _loaded:   # 重新加载模型后，需要重建sampler，否则默认用的是初始化的unet(除非test时也设置resume)
            self.diffusion_sampler = GaussianDiffusionSampler(
                model=self.unet,
                beta_1=self.beta_1,
                beta_T=self.beta_T,
                T=self.max_timesteps,
                infer_T=self.infer_T,
                squeue=self.squeue
            )
        else:
            raise ValueError("model load failed.")

        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print("Loading testing data...")
        # 创建测试集dataset
        test_dataset = self.get_tf_dataset(testing_paths, target_size=self.im_size, 
                                           batch_size=batch_size, shuffle=False, 
                                           slice_mode='all', mode='test')
        # test_dataset = self.build_all_slice_dataset(testing_paths, target_size=self.im_size,
        #                                              batch_size=batch_size, shuffle=False, per_slice=True)
        num_slices = 0
        for file in testing_paths:
            with h5py.File(file, 'r') as f:
                num_slices += f['input_images'].shape[0]
        print(f'Test slices: {num_slices}')

        if self.squeue is None:
            save_steps = 1
        else:
            save_steps = self.infer_T // self.squeue + 1  # 1000 // 250 + 1 = 5
        save_steps = getattr(self, 'save_steps', save_steps)  # 可根据实际采样步数调整
        # # output save file
        # save_h5_path = os.path.join(output_path, f"result_epoch_{self.counter}.hdf5")
        # f = h5py.File(save_h5_path, 'w')
        # output = f.create_dataset('out', shape=(num_slices, self.im_size[0], self.im_size[1], save_steps), dtype='float32')
        # lr = f.create_dataset('lr', shape=(num_slices, self.im_size[0], self.im_size[1]), dtype='float32')
        # hr = f.create_dataset('hr', shape=(num_slices, self.im_size[0], self.im_size[1]), dtype='float32')
        # print(f"Results will be saved to {save_h5_path}")

        sampling_times = []

        
        print(f">>>>>>>>>>> 测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")
        print('Start batch sampling...')
        pred_dict = {}
        gt_dict = {}
        input_dict = {}

        for batch_idx, (input_batch, target_batch, file_idx_batch) in enumerate(tqdm(test_dataset, desc='BatchTest', ncols=100, disable=True)):
            start_time = time.time()
            # input_batch, target_batch: [bs, h, w, c]
            bs = input_batch.shape[0]   # 由于最后一个batch size可能小于4，所以额外定义bs
            # 采样
            x_T = tf.random.normal(tf.shape(input_batch))
            if sampler=='ddim':
                x_0 = self.diffusion_sampler.ddim_reverse(x_T, context=input_batch)  # [bs, h, w, save_steps]
            elif sampler=='ddpm':
                x_0 = self.diffusion_sampler.reverse(x_T, context=input_batch)  # [bs, h, w, save_steps]

            #print(x_0.shape, tf.reduce_min(x_0).numpy(), tf.reduce_max(x_0).numpy())
            sampling_time = time.time() - start_time
            sampling_times.append(sampling_time)

            # 可视化前3个batch的第1个slice
            if batch_idx < 3:
                fig, axes = plt.subplots(1, save_steps+2, figsize=(3*(save_steps+2), 3))
                axes[0].imshow(input_batch[0,...,0], cmap='gray', vmin=-1, vmax=1)
                axes[0].set_title('Input')
                for k in range(save_steps):
                    axes[k+1].imshow(x_0[0,...,k], cmap='gray', vmin=-1, vmax=1)
                    axes[k+1].set_title(f'x0_{k}')
                axes[-1].imshow(target_batch[0,...,0], cmap='gray', vmin=-1, vmax=1)
                axes[-1].set_title('Target')
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f'sample_{batch_idx+1}.png'))
                plt.close(fig)

            pred_batch = x_0.numpy() if hasattr(x_0, 'numpy') else x_0
            gt_batch = target_batch.numpy() if hasattr(target_batch, 'numpy') else target_batch
            input_batch = input_batch.numpy() if hasattr(input_batch, 'numpy') else input_batch

            for i in range(bs):
                idx = int(file_idx_batch[i].numpy()) if hasattr(file_idx_batch[i], 'numpy') else file_idx_batch[i]
                pred_dict.setdefault(idx, []).append(pred_batch[i,...,-1])
                gt_dict.setdefault(idx, []).append(gt_batch[i,...,-1])
                input_dict.setdefault(idx, []).append(input_batch[i,...,-1])

            print(f'Batch {batch_idx+1}, sampling time: {sampling_time:.4f} s')

            # # # 计算每个case的metrics
            # for idx in pred_dict:
            #     pred_stack = np.stack(pred_dict[idx], axis=0)  # [d,h,w]
            #     gt_stack = np.stack(gt_dict[idx], axis=0)
            #     input_stack = np.stack(input_dict[idx], axis=0)
                #print('idx:', idx, 'pred_stack:', pred_stack.shape, 'gt_stack:', gt_stack.shape, 'input_stack:', input_stack.shape)


        #     # 保存hdf
        #     for i in range(bs):
        #         for k in range(save_steps):   # save_steps == x_0.shape[-1]
        #             img = x_0[i,...,k].numpy() if hasattr(x_0[i,...,k], 'numpy') else x_0[i,...,k]
        #             output[count+i,:, :, k] = img
        #         lr[count+i,:,:] = input_batch[i,...,0].numpy() if hasattr(input_batch[i,...,0], 'numpy') else input_batch[i,...,0]
        #         hr[count+i,:,:] = target_batch[i,...,0].numpy() if hasattr(target_batch[i,...,0], 'numpy') else target_batch[i,...,0]
        #     count += bs
        # f.close()


        if sampling_times:
            average_sampling_time = sum(sampling_times) / len(sampling_times)
            print(f"Average sampling time: {average_sampling_time:.4f} s")

        print('Finish batch sampling!!')
        print(f">>>>>>>>>>> 测试结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} <<<<<<<<<<<<")


        ## 存入hdf5
        case_ids = sorted(pred_dict.keys())
        save_dir = os.path.join(output_path, f"result_2d_epoch_{self.counter}")
        os.makedirs(save_dir, exist_ok=True)
        for idx in case_ids:
            pred_stack = np.stack(pred_dict[idx], axis=0)  # [d,h,w]
            gt_stack = np.stack(gt_dict[idx], axis=0)
            input_stack = np.stack(input_dict[idx], axis=0)
            print(f"Case {idx}: pred_stack.shape={pred_stack.shape}, gt_stack.shape={gt_stack.shape}, input_stack.shape={input_stack.shape}")

            save_h5_path = os.path.join(save_dir, f"case_{idx}.hdf5")
            with h5py.File(save_h5_path, 'w') as f:
                f.create_dataset('output', data=pred_stack, dtype='float32')
                f.create_dataset('input', data=input_stack, dtype='float32')
                f.create_dataset('target', data=gt_stack, dtype='float32')
            print(f"Saved results to {save_h5_path}")