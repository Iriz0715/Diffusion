import os, glob, re, random
import numpy as np
import h5py, json
import multiprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model, Sequential

from augmentation import *

from abstract_model import AbstractModel, AUGMENTATION_PARAMS
from utils_detection_3d import mean_average_precision_for_boxes, extract_bboxes_from_annotation

def create_fpr_dataset(cases, im_size=[48, 48], distance_threshold=10, spacing=[1.0, 1.0, 1.0]):
    """
    cases: case paths. Create fpr dataset for these cases. They should be in data_training/ or data_training/train/
    im_size: the crop size for each detected object. In the unit of mm.
    distance_threshold: bbox is considered as true positive within this threshold. In the unit of mm.
    spacing: physical spacing per voxel.
    """
    from tqdm import tqdm
    from utils_detection_3d import compute_distance_hit
    crop_patch_size = [3,] + list(im_size)
    spacing = np.array(spacing)
    naming_suffix = 'TASK-C'

    output_path = os.path.dirname(cases[0]).replace('/data_training/', '/fpr/data_training/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    case_number = 0
    for case in tqdm(cases):
        naming_prefix = os.path.basename(case).split('_')[0] + '_'
        case_number += 1
        if '/data_training/train/' in case:
            bbox_path = case.replace('/data_training/train/', '/output/cv/detection/Group0_').replace('.hdf5', '.npy')
        else:
            bbox_path = case.replace('/data_training/', '/output/test/detection/Group0_').replace('.hdf5', '.npy')
        predictions = np.load(bbox_path)
        
        with h5py.File(case, 'r') as f_h5:
            images = np.asarray(f_h5['images']).astype(np.float32)
            annotation = json.loads(f_h5.attrs['annotation'])
            if 'meta' in f_h5.attrs.keys():
                meta = json.loads(f_h5.attrs['meta'])
            else:
                meta = {}
        
        meta['image_path'] = str(case)
        meta['bbox_path'] = str(bbox_path)
    
        gt_bboxes = extract_bboxes_from_annotation(annotation)
        gt_bboxes = np.array(gt_bboxes).astype(np.float32)
        pred_bboxes = predictions[:, -6:].astype(np.float32)
    
        if gt_bboxes.shape[0] == 0:
            continue
    
        gt_bboxes[:, :3] += gt_bboxes[:, 3:] / 2  # to center
        gt_bboxes[:, 3:] = 2 * distance_threshold / spacing[::-1]
        gt_bboxes[:, :3] -= gt_bboxes[:, 3:] / 2  # to corner
        
        hit_matrix = compute_distance_hit(gt_bboxes, pred_bboxes, spacing[::-1])
        hit_by_pred = np.amax(hit_matrix, axis=0)
        bboxes = np.concatenate([pred_bboxes, gt_bboxes], axis=0)
        class_labels = np.append(hit_by_pred, [1] * len(gt_bboxes))

        for ax in ['AX', 'CO', 'SA']:
            if ax == 'AX':
                ax_bboxes = np.copy(bboxes)
                ax_images = np.copy(images)
                ax_spacing = np.copy(spacing)
            elif ax == 'CO':
                ax_bboxes = np.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 2], bboxes[:, 4], bboxes[:, 3], bboxes[:, 5]], 1)
                ax_images = np.transpose(images, [1, 0, 2])
                ax_spacing = np.array([spacing[2], spacing[0], spacing[1]])
            elif ax == 'SA':
                ax_bboxes = np.stack([bboxes[:, 2], bboxes[:, 0], bboxes[:, 1], bboxes[:, 5], bboxes[:, 3], bboxes[:, 4]], 1)
                ax_images = np.transpose(images, [2, 0, 1])
                ax_spacing = np.array([spacing[1], spacing[0], spacing[2]])
            for bbox, class_label in zip(ax_bboxes, class_labels):
                patch_size = [crop_patch_size[0], int(crop_patch_size[1] // ax_spacing[1]), int(crop_patch_size[2] // ax_spacing[0])]
            
                center = (bbox[:3] + bbox[3:] // 2).astype(int)
                patch = ax_images[
                    max(center[0] - patch_size[0] // 2, 0) : max(center[0] - patch_size[0] // 2, 0) + patch_size[0], 
                    max(center[1] - patch_size[1] // 2, 0) : max(center[1] - patch_size[1] // 2, 0) + patch_size[1], 
                    max(center[2] - patch_size[2] // 2, 0) : max(center[2] - patch_size[2] // 2, 0) + patch_size[2]
                ]
        
                if min(patch.shape) == 0:
                    continue
                if patch.shape != crop_patch_size:
                    pads = [max(0, crop_patch_size[ax] - patch.shape[ax]) for ax in range(len(crop_patch_size))]
                    pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                                (pads[1] // 2, pads[1] - pads[1] // 2), 
                                (pads[2] // 2, pads[2] - pads[2] // 2))
                    patch = np.pad(patch, pad_with, mode='edge')
                    
                patch = patch.transpose(1, 2, 0)
                
                output_name = naming_prefix + f'{case_number:06d}_' + f'{ax}_' + '-'.join(list(map(str, center))) 
                output_name += f'_class_{int(class_label)}_' + naming_suffix + '.hdf5'
                
                class_label = int(class_label)
                with h5py.File(os.path.join(output_path, output_name), 'w') as f_h5:
                    f_h5['images'] = patch.astype(np.float32)
                    f_h5.attrs['class_label'] = str(class_label)
                    f_h5.attrs['meta'] = json.dumps(meta)


class BaseModel3D(AbstractModel):
    def perform_augmentation(self, images, bboxes, patch_size, augmentation_params={}):
        AUGMENTATION_PARAMS.update(augmentation_params)
        bboxes_aug = np.array(bboxes)
        if len(images.shape) == 4:
            images_aug = np.expand_dims(np.transpose(images, (3, 0, 1, 2)), axis=0)
        else:
            images_aug = np.expand_dims(images, axis=(0, 1))
        if len(patch_size) == 4:
            patch_size = patch_size[:-1]
        images_aug, _ = augment_spatial_2(images_aug, None, patch_size=patch_size,
                                          patch_center_dist_from_border=
                                          AUGMENTATION_PARAMS.get('random_crop_dist_to_border'),
                                          do_elastic_deform=AUGMENTATION_PARAMS.get('do_elastic'),
                                          deformation_scale=AUGMENTATION_PARAMS.get('deformation_scale'),
                                          do_rotation=AUGMENTATION_PARAMS.get('do_rotation'),
                                          angle_x=AUGMENTATION_PARAMS.get('rotation_x'),
                                          angle_y=AUGMENTATION_PARAMS.get('rotation_y'),
                                          angle_z=AUGMENTATION_PARAMS.get('rotation_z'),
                                          p_rot_per_axis=AUGMENTATION_PARAMS.get('rotation_p_per_axis'),
                                          do_scale=False, # do scale with bbox below
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
        if AUGMENTATION_PARAMS.get('dummy_2D') and len(images_aug.shape) == 4 and dim == 3:
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
        
        if AUGMENTATION_PARAMS.get('do_scaling'):
            images_aug, bboxes_aug = augment_scaling_with_bbox(
                images_aug, bboxes_aug, 
                scale=AUGMENTATION_PARAMS.get('scale_range'),
                order_data=3,
                border_cval_data=0.,
                border_mode_data=AUGMENTATION_PARAMS.get('border_mode_data'),
                p_scale_per_sample=AUGMENTATION_PARAMS.get('p_scale'),
                independent_scale_for_each_axis=AUGMENTATION_PARAMS.get('independent_scale_factor_for_each_axis')
            )
        
        if self.mirror_config is not None and self.mirror_config.get('training_mirror', False):
            images_aug, bboxes_aug = augment_mirroring_with_bbox(images_aug, bboxes_aug)
        
        if len(images.shape) == 4:
            images_aug = np.transpose(images_aug, (1, 2, 3, 0))
        else:
            images_aug = np.squeeze(images_aug, axis=0)
        return images_aug, bboxes_aug

    @staticmethod
    def simple_upsampling(images, ratio):
        ratio = np.asarray(ratio).astype(int)
        upsampled_images = np.copy(images)
        
        for ax in range(len(ratio)):
            if ratio[ax] > 1:
                upsampled_images = upsampled_images.repeat(ratio[ax], axis=ax)
        
        return upsampled_images

    def read_training_inputs(self, file, im_size, enlarged_im_size, augmentation=True, class_balanced=True):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            # labels = np.asarray(f_h5['labels'], dtype=np.int32)
            annotation = json.loads(f_h5.attrs['annotation'])
            bboxes, cls_ids = extract_bboxes_from_annotation(annotation, return_categories=True)
            bboxes = np.asarray(bboxes, np.float32)
            bboxes[:, :3] = bboxes[:, :3] + (bboxes[:, 3:] - 1) / 2  # corner to center
            cls_ids = np.asarray(cls_ids, dtype=np.int32)
        
        # Skip no bboxes
        if len(cls_ids) == 0:
            return np.zeros(im_size, np.float32), bboxes, cls_ids
        
        # Upsampling
        images = self.simple_upsampling(images, self.detection_upsample)
        bboxes[:, :3] = bboxes[:, :3] * np.asarray(self.detection_upsample)
        bboxes[:, 3:] = bboxes[:, 3:] * np.asarray(self.detection_upsample)
        
        bboxes = np.round(bboxes).astype(np.int32)
        
        for i in range(len(enlarged_im_size)):
            if images.shape[i] < enlarged_im_size[i] + 1:
                padding = ((0, 0),) * i + ((0, enlarged_im_size[i] + 1 - images.shape[i]),)
                padding += ((0, 0),) * (len(images.shape) - i - 1)
                images = np.pad(images, padding, mode='constant')

        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)

        # Normalize images
        if self.norm_config['norm']:
            if len(images.shape) == 4:
                if self.norm_config['norm_channels'] == 'all_channels':
                    images = (images - np.mean(images, axis=(0, 1, 2))) / np.std(images, axis=(0, 1, 2))
                else:
                    for channel in self.norm_config['norm_channels']:
                        m = np.mean(images[..., channel])
                        s = np.std(images[..., channel])  
                        images[..., channel] = (images[..., channel] - m) / s
            else:
                images = (images - np.mean(images)) / np.std(images)

        x_range = full_size[0] - enlarged_im_size[0]
        y_range = full_size[1] - enlarged_im_size[1]
        z_range = full_size[2] - enlarged_im_size[2]

        x_offset = int(enlarged_im_size[0] / 2)
        y_offset = int(enlarged_im_size[1] / 2)
        z_offset = int(enlarged_im_size[2] / 2)
        
        x_border_width = int((enlarged_im_size[0] - im_size[0]) / 2)
        y_border_width = int((enlarged_im_size[1] - im_size[1]) / 2)
        z_border_width = int((enlarged_im_size[2] - im_size[2]) / 2)

        # Get sampling prob
        if np.random.random() > self.fg_sampling_ratio:
            # choose random
            o = np.random.choice(x_range * y_range * z_range)
        else:
            # choose foreground of this class
            p = np.zeros((x_range, y_range, z_range), dtype=np.float32)
            if len(bboxes) > 0:
                exist_classes = list(np.unique(cls_ids).astype(int))
                area_mask = np.zeros(full_size + [len(exist_classes),], dtype=np.uint8)
                for bbox, cls in zip(bboxes, cls_ids):
                    # cls + 1: in case cls_ids start from 0
                    # area_mask: the area set of all left-top corner patch origins
                    #   where the patch can cover at least one bbox
                    #   area_mask collected by each class
                    c = exist_classes.index(int(cls))
                    area_mask[max(0, bbox[0] + bbox[3] - im_size[0] - x_border_width) : bbox[0] - x_border_width, 
                              max(0, bbox[1] + bbox[4] - im_size[1] - y_border_width) : bbox[1] - y_border_width, 
                              max(0, bbox[2] + bbox[5] - im_size[2] - z_border_width) : bbox[2] - z_border_width,
                              c] = cls + 1
                am = area_mask[:x_range, :y_range, :z_range]
                # choose class
                if class_balanced:
                    selected_class = np.random.choice(exist_classes)
                    p[am[..., exist_classes.index(selected_class)] == selected_class + 1] = 1
                else:
                    p[np.any(am > 0, axis=-1)] = 1
                if np.amax(p) == 0:  # in case NaN for p
                    p = np.ones((x_range, y_range, z_range), dtype=np.float32)
            else:
                # if foreground is not present (gives NaN value for p)
                p = np.ones((x_range, y_range, z_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range * z_range, p=p)
        
        start_idx = np.unravel_index(o, (x_range, y_range, z_range))
        x_start, y_start, z_start = np.clip(start_idx, 0, (x_range, y_range, z_range))

        # Extracted the enlarged image
        images_extracted = images[x_start : x_start + enlarged_im_size[0], y_start : y_start + enlarged_im_size[1],
                                  z_start : z_start + enlarged_im_size[2]]
        bboxes_extracted = bboxes.copy()
        cls_ids_extracted = cls_ids.copy()
        bboxes_extracted[:, :3] = bboxes[:, :3] - np.array([x_start, y_start, z_start])
        # Deal with bbox outside patch
        keep_idx = [np.all(np.logical_and(bbox[:3] > 0, bbox[:3] < np.array(enlarged_im_size)))
                    for bbox in bboxes_extracted]
        bboxes_extracted = bboxes_extracted[keep_idx]
        cls_ids_extracted = cls_ids_extracted[keep_idx]

        if augmentation:
            images_extracted, bboxes_extracted = self.perform_augmentation(images_extracted, bboxes_extracted, 
                                                                           images_extracted.shape, 
                                                                           augmentation_params={'do_rotation': False})
        
        # Crop the im_size patch
        images_extracted = images_extracted[x_border_width : x_border_width + im_size[0], 
                                            y_border_width : y_border_width + im_size[1], 
                                            z_border_width : z_border_width + im_size[2]]
        
        bboxes_extracted[:, :3] = bboxes_extracted[:, :3] - (np.array(enlarged_im_size) - np.array(im_size)) / 2
        # Deal with bbox outside patch
        keep_idx = [np.all(
            np.logical_and(
                np.logical_and(bbox[:3] > 0, bbox[:3] < np.array(im_size)), # xyz within patch
                np.logical_and(bbox[3:] > 0, bbox[3:] < np.array(full_size))) # xyz_range < full_size
        ) for bbox in bboxes_extracted]
        bboxes_extracted = bboxes_extracted[keep_idx]
        cls_ids_extracted = cls_ids_extracted[keep_idx]

        bboxes_extracted = np.asarray(bboxes_extracted, dtype=np.float32)
        cls_ids_extracted = np.asarray(cls_ids_extracted, dtype=np.float32)
        
        return images_extracted, bboxes_extracted, cls_ids_extracted

    def read_testing_inputs(self, file, im_size, nstride=[1, 1, 1]):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
        
        images = self.simple_upsampling(images, self.detection_upsample)
        
        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)
        
        # Get sliding steps
        step_size = [int(im_size[ax] / nstride[ax]) for ax in range(len(im_size))]

        num_steps = [int(np.ceil((full_size[ax] - im_size[ax]) / step_size[ax])) + 1 for ax in range(len(im_size))]

        # Needs to be at least 1
        num_steps = [max(1, n_step) for n_step in num_steps]
        
        axis_steps = []
        for ax in range(len(im_size)):
            # the highest step value for this dimension is
            max_step_value = full_size[ax] - im_size[ax]
            if num_steps[ax] > 1:
                actual_step_size = max_step_value / (num_steps[ax] - 1)
            else:
                actual_step_size = 1e+8  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[ax])]

            axis_steps.append(steps_here)

        steps = []
        for ix, x_step in enumerate(axis_steps[0]):
            for iy, y_step in enumerate(axis_steps[1]):
                for iz, z_step in enumerate(axis_steps[2]):
                    steps.append([x_step, y_step, z_step])
        
        # Pad if full size is smaller than patch size
        # symmetric padding is always better than one-side padding
        pads = [max(0, im_size[ax] - full_size[ax]) for ax in range(len(im_size))]
        
        if np.any(np.array(pads) > 0):
            if len(images.shape) == 4:
                pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2), 
                            (pads[2] // 2, pads[2] - pads[2] // 2), (0, 0))
            else:
                pad_with = ((pads[0] // 2, pads[0] - pads[0] // 2), 
                            (pads[1] // 2, pads[1] - pads[1] // 2), 
                            (pads[2] // 2, pads[2] - pads[2] // 2))
            images = np.pad(images, pad_with, mode='constant')
        
        pad_size = list(images.shape)[:3]
        
        info = {
            'full_size': full_size,
            'pad_size': pad_size,
            'steps': steps,
        }
        
        # Normalize images
        if self.norm_config['norm']:
            if len(images.shape) == 4:
                if self.norm_config['norm_channels'] == 'all_channels':
                    images = (images - np.mean(images, axis=(0, 1, 2))) / np.std(images, axis=(0, 1, 2))
                else:
                    for channel in self.norm_config['norm_channels']:
                        m = np.mean(images[..., channel])
                        s = np.std(images[..., channel])
                        images[..., channel] = (images[..., channel] - m) / s
            else:
                images = (images - np.mean(images)) / np.std(images)

        if len(images.shape) == 4:
            all_images = np.empty([len(steps),] + im_size + [images.shape[-1]], dtype=np.float32)
        else:
            all_images = np.empty([len(steps),] + im_size, dtype=np.float32)
        
        for o, (x_step, y_step, z_step) in enumerate(steps):
            all_images[o] = images[x_step : x_step + im_size[0], 
                                   y_step : y_step + im_size[1],
                                   z_step : z_step + im_size[2], ...]
        
        return all_images, info
            
    def train_data_mapper(self, i):
        """Applies preprocessing step to a single sample

        Arguments:
          sample: An integer <= len(training_paths).

        Returns:
          image: A patch with augmentation.
          bboxes: Bounding boxes with the shape `(num_objects, 6)` where each box is
            of the format `[x, y, z, x_range, y_range, z_range]`.
          cls_ids: An tensor representing the class id of the objects, having
            shape `(num_objects,)`.
        """
        images, bboxes, cls_ids = self.read_training_inputs(self.training_paths[i], self.im_size, self.enlarged_im_size)
        if self.input_channels == 1 and len(images.shape) == 3:
            images = np.expand_dims(images, -1)
        return images, bboxes, cls_ids
    
    def process_data_batch(self, q, idx_list):
        while True:
            shuffle_list = np.random.permutation(idx_list)
            images_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], 
                                     self.input_channels), dtype=np.float32)
            bboxes_batch = []
            cls_ids_batch = []
            ib = 0
            idx = 0
            while ib < self.batch_size and idx < len(idx_list):
                i = shuffle_list[idx]
                idx += 1
                images, bboxes, cls_ids = self.train_data_mapper(i)
                if len(cls_ids) == 0:
                    continue
                images_batch[ib] = images
                ib += 1
                bboxes_batch.append(bboxes.tolist())
                cls_ids_batch.append(cls_ids.tolist())
            # Drop remainder
            if ib < self.batch_size:
                continue
            bboxes_batch = tf.keras.preprocessing.sequence.pad_sequences(bboxes_batch, dtype=np.float32, value=1e-8)
            cls_ids_batch = tf.keras.preprocessing.sequence.pad_sequences(cls_ids_batch, dtype=np.float32, value=-1.)
            
            batch_images, batch_labels = self.label_encoder.encode_batch(images_batch, bboxes_batch, cls_ids_batch)
            q.put((batch_images, batch_labels))

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
                it += 1
                yield stuff
            except:
                break
        pool.close()

    def train(self, run_validation=True, validation_paths=None, **kwargs):
        # Compile model for training
        self.compile_it()
        
        # Training config
        log_dir = os.path.join(self.log_dir, self.model_dir)
        if run_validation:
            profiler = 0
        else:
            profiler = 0
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=profiler)
        
        validation_config = None
        if run_validation:
            validation_config = {
                'training_paths': self.training_paths,
                'validation_paths': validation_paths,
                'validation_fn': self.validate,
        }
        saver_config = {
            'period': self.save_period,
            'log_dir': log_dir,
            'save_fn': self.save, 
        }
        saver_callback = self.ModelSaver(saver_config=saver_config, validation_config=validation_config)
        
        ## TODO: update to simple data generator
        # Prepare for data generator
        num_samples = len(self.training_paths)
        print('total training samples:', num_samples)

        num_samples = len(self.training_paths)
        if num_samples == 0:
            print('No training data')
            return
        idx_list = np.arange(num_samples)
        total_iters = self.steps_per_epoch * (self.epoch - self.counter)
        data_generator = self.data_generator(idx_list, total_iters)
        
        if total_iters > 0:
            self.networks.fit(data_generator, validation_data=None, verbose=2,
                              steps_per_epoch=self.steps_per_epoch, 
                              initial_epoch=self.counter, epochs=self.epoch,
                              callbacks=[tensorboard_callback, saver_callback])
            self.save(self.epoch)
        
        print('Training scores:')
        self.validate(self.training_paths, test_nstride=[1, 1, 1])
        if run_validation and validation_paths:
            print('Validation scores:')
            self.validate(validation_paths, test_nstride=[1, 1, 1])
        return
    
    def validate(self, validation_paths, test_nstride=[1, 1, 1], other_task_config={}):
        self.inference_model = self.build_inference_model()
        all_gt = np.zeros([0, 8])
        all_pred = np.zeros([0, 9])
        for validation_path in validation_paths:
            with h5py.File(validation_path, 'r') as f_h5:
                annotation = json.loads(f_h5.attrs['annotation'])
                gt_bboxes, gt_cls_ids = extract_bboxes_from_annotation(annotation, return_categories=True)
                gt_bboxes = np.asarray(gt_bboxes, np.float32)
                gt_cls_ids = np.asarray(gt_cls_ids, dtype=np.float32)
            
            if len(gt_cls_ids) == 0:
                continue
            
            bboxes, scores, cls_ids = self.run_test(validation_path, test_nstride)
            
            gt = np.concatenate(
                [np.tile(os.path.basename(validation_path), [len(gt_bboxes), 1]), 
                 np.expand_dims(gt_cls_ids, axis=1),
                 gt_bboxes], axis=1)
            all_gt = np.concatenate([all_gt, gt])
            
            pred = np.concatenate(
                [np.tile(os.path.basename(validation_path), [len(scores), 1]), 
                 np.expand_dims(cls_ids, axis=1),
                 np.expand_dims(scores, axis=1),
                 bboxes], axis=1)
            all_pred = np.concatenate([all_pred, pred])
        
        return all_pred, all_gt, mean_average_precision_for_boxes(all_gt, all_pred, iou_threshold=self.match_iou)[0]
    
    def test(self, testing_paths, output_path, other_task_config={}, **kwargs):
        if not self._loaded:
            self.load()
            raise Exception('No model is found, please train first')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.inference_model = self.build_inference_model()

        from tqdm import tqdm
        for input_file in tqdm(testing_paths):
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file)).replace('.hdf5', '.npy')
            if os.path.exists(output_file):
                continue
            
            bboxes, scores, cls_ids = self.run_test(input_file, self.test_nstride)
            
            pred = np.concatenate(
                [np.tile(os.path.basename(input_file), [len(scores), 1]), 
                 np.expand_dims(cls_ids, axis=1),
                 np.expand_dims(scores, axis=1),
                 bboxes], axis=1)

            np.save(output_file, pred)

    def build_inference_model(self):
        inputs = layers.Input(shape=[None, None, None, self.input_channels])
        predictions = self.networks(inputs)
        detections = self.prediction_decoder(inputs, predictions)
        inference_model = Model(inputs=inputs, outputs=detections)
        return inference_model
    
    def run_test(self, input_file, test_nstride):
        if self.inference_model is None:
            self.inference_model = self.build_inference_model()
        test_nstride = np.round(np.divide(test_nstride, self.detection_upsample), 1)
        all_images, info = self.read_testing_inputs(input_file, self.im_size, test_nstride)
        patch_origin = np.asarray(info['steps'], dtype=np.int32)

        all_bboxes = np.zeros((0, 6), dtype=np.float32)
        all_scores = np.zeros((0,), dtype=np.float32)
        all_cls_ids = np.zeros((0,), dtype=np.uint8)

        # Run prediction
        for n in range(all_images.shape[0]):
            tensor_im = tf.convert_to_tensor(np.expand_dims(all_images[n : n + 1], axis=-1), tf.float32)
            bboxes, scores, cls_ids = self.inference_model.predict_on_batch(tensor_im)
            # restore to global location
            if len(bboxes) > 0:
                bboxes[:, :3] = bboxes[:, :3] - (bboxes[:, 3:] - 1) / 2  # center2corner
                bboxes += np.append(patch_origin[n], [0, 0, 0])
                # resize back to image original spacings
                bboxes[:, :3] = bboxes[:, :3] / np.asarray(self.detection_upsample)
                bboxes[:, 3:] = bboxes[:, 3:] / np.asarray(self.detection_upsample)
                all_bboxes = np.concatenate([all_bboxes, bboxes], axis=0)
                all_scores = np.concatenate([all_scores, scores], axis=0)
                all_cls_ids = np.concatenate([all_cls_ids, cls_ids], axis=0)
        all_bboxes_nms = np.zeros((0, 6), dtype=np.float32)
        all_scores_nms = np.zeros((0,), dtype=np.float32)
        all_cls_ids_nms = np.zeros((0,), dtype=np.uint8)
        for c in np.unique(all_cls_ids):
            all_bboxes_per_class = all_bboxes[all_cls_ids==c]
            all_scores_per_class = all_scores[all_cls_ids==c]
            all_bboxes_per_class, all_scores_per_class = self.inference_model.layers[-1]._nms_eager(all_bboxes_per_class,
                                                                                               all_scores_per_class)
            all_bboxes_nms = np.concatenate([all_bboxes_nms, all_bboxes_per_class], axis=0)
            all_scores_nms = np.concatenate([all_scores_nms, all_scores_per_class], axis=0)
            all_cls_ids_nms = np.concatenate([all_cls_ids_nms, np.ones_like(all_scores_per_class) * c], axis=0)
        
        selected_idxs = np.argsort(all_scores_nms)[::-1][:self.max_detections]
        return all_bboxes_nms[selected_idxs], all_scores_nms[selected_idxs], all_cls_ids_nms[selected_idxs]
    
    class ModelSaver(tf.keras.callbacks.Callback):
        def __init__(self, saver_config, validation_config=None):
            self.counter = 0
            self.period = saver_config['period']
            self.save = saver_config['save_fn']
            self.validation_config = validation_config
            if validation_config is not None:
                self.training_paths = validation_config['training_paths']
                self.validation_paths = validation_config['validation_paths']
                self.validation_fn = validation_config['validation_fn']

        def on_epoch_end(self, epoch, logs={}):
            self.counter += 1
            if self.counter % self.period == 0:
                self.save(epoch + 1)
                if self.validation_config is not None:
                    for _, paths in enumerate([self.training_paths, self.validation_paths]):
                        if _ == 0:
                            print('Training scores:')
                        else:
                            print('Validation scores:')
                        path_groups = {}
    
                        # Group the file paths by basename prefix
                        for file_path in paths:
                            base_name = os.path.basename(file_path)
                            prefix = base_name.split('_')[0]
                        
                            if prefix not in path_groups:
                                path_groups[prefix] = []
                        
                            path_groups[prefix].append(file_path)
                        
                        for prefix in path_groups.keys():
                            print('----', prefix, '----')
                            selected_paths = path_groups[prefix][:10]
                            self.validation_fn(selected_paths, test_nstride=[1, 1, 1])