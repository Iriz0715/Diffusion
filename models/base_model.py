import os, glob, re, time, csv, datetime, shutil, pathlib, gc
from tqdm import tqdm
import numpy as np
import h5py, json
import multiprocessing, threading
from skimage import measure
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, Sequential

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
    'random_crop_dist_to_border': None,
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
        loaded = 0
        counter = 0
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            print('No model is found, please train first')
        else:
            ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
            if ckpt_files:
                ckpt_file = ckpt_files[-1]
                ckpt_name = os.path.basename(ckpt_file)
                ckpt_dirname = os.path.basename(os.path.dirname(ckpt_file))
                counter = int(re.findall(r'epoch\d+', ckpt_name)[0][5:])  #e.g. model_epoch600.h5
                self.unet = self.build_unet()
                try:
                    self.unet.load_weights(ckpt_file)
                    loaded = 1
                    print('Loaded model checkpoint:', ckpt_dirname, ckpt_name)
                except:
                    print('[Warning] Current architecture is different with the loading model.', 
                          'Trying to transfer and retrain.')
                    counter = 0
                    try:
                        self.load_weights_for_transfer_learning(ckpt_file, ignore_keywords=[])
                        loaded = 2
                        print('[Warning] Partially load model file:', ckpt_dirname, ckpt_name)
                        print('[Warning] Retrain from transferred model.')
                    except:
                        loaded = 3
                        print('[Warning] Failed to transfer the model file:', ckpt_dirname, ckpt_name)
                        print('[Warning] Retrain from scratched model.')
            else:
                print('Failed to find a checkpoint')
        
        self._loaded, self.counter = loaded, counter
        return loaded, counter

    @staticmethod
    def parse_loss_head(loss_head, rois):
        if loss_head:
            internal_loss_head = []
            weight_sum = 0.0
            num_bg_channels = 0
            for _head in loss_head:
                head = {'rois': [], 'activation': 'softmax', 'weight': 1.0}
                head['activation'] = _head.get('activation', 'softmax')
                # Convert the rois to roi channel indices
                # Softmax requires one additional channel for background, append it to the end.
                if head['activation'] == 'softmax':
                    head['rois'].append(len(rois) + num_bg_channels)
                    num_bg_channels += 1
                for roi in _head['rois']:
                    head['rois'].append(rois.index(roi))
                
                head['weight'] = float(_head.get('weight', 1.0))
                weight_sum += head['weight']
                internal_loss_head.append(head)
            # Normalize weights
            for head in internal_loss_head:
                head['weight'] /= weight_sum
        else:
            internal_loss_head = None
        return internal_loss_head

    @staticmethod
    def get_strides_list(layer_number, im_size):
        s_list = np.zeros([layer_number, len(im_size)], dtype=np.uint8)
        for i in range(layer_number):
            # stop pooling when size is odd && size <= 4
            to_pool = (np.array(im_size) % 2 ** (i + 1) == 0).astype(int)
            to_pool *= ((np.array(im_size) // 2 ** (i + 1)) > 4).astype(int)
            s_list[i] = 1 + to_pool
        strides_list = np.concatenate([s_list[::-2], s_list[layer_number % 2::2]])
        return strides_list
    
    def append_to_log_file(self, logs, custom_log_file):
        if custom_log_file is not None:
            with open(custom_log_file, 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'), logs])

    def load_weights_for_transfer_learning(self, old_model_file, custom_log_file=None, ignore_keywords=['logits']):
        # ignore_keywords: the keyword included in the layer name
        
        # load weights by name and skip mismatch weights shape
        pretrain_m = tf.keras.models.load_model(old_model_file, compile=False)
        for l in self.unet.layers:
            ignored = False
            for k in ignore_keywords:
                if k in l.name:
                    # self.append_to_log_file(f'{l.name} No transfer. [Ignored]', custom_log_file)
                    ignored = True
            if ignored:
                continue
            try:
                self.unet.get_layer(name=l.name).set_weights(pretrain_m.get_layer(name=l.name).get_weights())
                # self.append_to_log_file(f'{l.name} Successfully transferred.', custom_log_file)
            except:
                # self.append_to_log_file(f'{l.name} No transfer. [Incompatible shapes]', custom_log_file)
                continue
        
        # self.append_to_log_file(f'Loaded pre-trained model for transfer learning: {old_model_file}', custom_log_file)
        return

    @staticmethod
    def filter_training_cases(cases, rois):
        output_cases = []
        all_shapes = []
        all_case_rois = []
        numbers = np.zeros(len(rois))

        for case in tqdm(cases):
            with h5py.File(case, 'r') as f_h5:
                # The dimension of labels can be 3 or 4, 
                # 4 is used to support multiple channels for potentially overlapping labels
                labels = np.asarray(f_h5['labels'], dtype=np.float32)

            # roi labels only contain the labels within the model rois
            if len(labels.shape) == 4:
                shape = labels.shape[:3]
            else:
                shape = labels.shape

            roi_labels = np.zeros(shape, dtype=np.uint8)
            case_rois = []
            for ir, roi in enumerate(rois):
                if len(labels.shape) == 4:
                    channel = np.floor_divide(roi, 256)
                    value = roi - channel * 256
                    # Check if labels has the channel
                    if channel < labels.shape[-1]:
                        roi_labels[labels[..., channel] == value] = ir + 1
                        if np.amax(labels[..., channel] == value) > 0:
                            numbers[ir] += 1
                            case_rois.append(roi)
                else:
                    if roi < 256: # Probably unnecessary, but just in case there are some type casting in comparison
                        roi_labels[labels == roi] = ir + 1
                        if np.amax(labels == roi) > 0:
                            numbers[ir] += 1
                            case_rois.append(roi)

            if np.amax(roi_labels) > 0:
                output_cases.append(case)
                all_shapes.append(shape)
                all_case_rois.append(case_rois)

        return output_cases, all_case_rois, np.median(all_shapes, axis=0)
        
    def swap_left_right(self, labelmap, swap_pairs, is_prob=False):
        swapped = np.copy(labelmap)
        for left, right in swap_pairs:
            # if prob, the order along the last axis is the roi order within rois
            # if labelmap, the value is the value stored in roi
            if is_prob:
                swapped[..., left] = labelmap[..., right]
                swapped[..., right] = labelmap[..., left]
            else:
                swapped[labelmap == left] = right
                swapped[labelmap == right] = left
        return swapped
    
    def perform_augmentation(self, images, labels, patch_size):
        if len(labels.shape) == 4:
            labels_aug = np.expand_dims(np.transpose(labels, (3, 0, 1, 2)), axis=0)
        else:
            labels_aug = np.expand_dims(labels, axis=(0, 1))
        if len(images.shape) == 4:
            images_aug = np.expand_dims(np.transpose(images, (3, 0, 1, 2)), axis=0)
        else:
            images_aug = np.expand_dims(images, axis=(0, 1))
        
        # Convert to 2D tiles. Reshape axial and channel into one axis. Only applicable to 3D data.
        if AUGMENTATION_PARAMS.get('dummy_2D') and len(images_aug.shape) == 5:
            orig_img_shp = images_aug.shape
            images_aug = images_aug.reshape((orig_img_shp[0], orig_img_shp[1] * orig_img_shp[2],
                                             orig_img_shp[3], orig_img_shp[4]))
            orig_lbl_shp = labels_aug.shape
            labels_aug = labels_aug.reshape((orig_lbl_shp[0], orig_lbl_shp[1] * orig_lbl_shp[2], 
                                             orig_lbl_shp[3], orig_lbl_shp[4]))
            
            assert len(patch_size) == 3
            patch_size = patch_size[1:]
            
            # With dummy_2D aug, it should be 2D like.
            assert len(images_aug.shape) == 4 and len(labels_aug.shape) == 4
            
        images_aug, labels_aug = augment_spatial_2(images_aug, labels_aug, patch_size=patch_size,
                                                   patch_center_dist_from_border=None,
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
            current_lbl_shape = labels_aug.shape
            labels_aug = labels_aug.reshape((orig_lbl_shp[0], orig_lbl_shp[1], orig_lbl_shp[2], 
                                             current_lbl_shape[-2], current_lbl_shape[-1]))
        
        images_aug = np.squeeze(images_aug, axis=0)
        labels_aug = np.squeeze(labels_aug, axis=0)

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
            images_aug, labels_aug = augment_mirroring(images_aug, labels_aug, mirror_axes)

        if len(images.shape) == 4:
            images_aug = np.transpose(images_aug, (1, 2, 3, 0))
        else:
            images_aug = np.squeeze(images_aug, axis=0)
        if len(labels.shape) == 4:
            labels_aug = np.transpose(labels_aug, (1, 2, 3, 0))
        else:
            labels_aug = np.squeeze(labels_aug, axis=0)
        return images_aug, labels_aug

    def read_training_inputs(self, file, rois, im_size, enlarged_im_size, augmentation=True, class_balanced=True):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            labels = np.asarray(f_h5['labels'], dtype=np.float32)
        
        if self.left_right_swap_config is not None and self.left_right_swap_config['training_flip']:
            swap_pairs = self.left_right_swap_config['swap_pairs']
            if augmentation and np.random.uniform() > 0.5:
                images = images[:, :, ::-1, ...]  # Maybe 4 dims
                if len(labels.shape) == 4:
                    for channel in range(labels.shape[-1]):
                        # Get swap pairs in this channel only, make sure each pair is within one channel
                        new_swap_pairs = []
                        for pair in swap_pairs:
                            pair_channel = np.floor_divide(pair[0], 256)
                            if pair_channel == channel:
                                new_swap_pairs.append((pair[0] - pair_channel * 256, pair[1] - pair_channel * 256))
                        labels[..., channel] = labels[..., ::-1, channel]
                        labels[..., channel] = self.swap_left_right(labels[..., channel], new_swap_pairs)
                else:
                    labels = labels[..., ::-1]
                    labels = self.swap_left_right(labels, swap_pairs)                    

        nclass = self.nclass
        # roi labels only contain the labels within roi
        if self.loss_head:
            roi_labels = np.zeros(labels.shape[:3] + (len(self.loss_head),), dtype=np.float32)
        else:
            roi_labels = np.zeros(labels.shape[:3], dtype=np.float32) # Original labels may be 4D
        # Whether or not the organ exists from the global label maps
        roi_exists = []
        for ir, roi in enumerate(rois):
            if len(labels.shape) == 4:
                channel = np.floor_divide(roi, 256)
                value = roi - channel * 256
                if channel < labels.shape[-1] and np.amax(labels[..., channel] == value) > 0:
                    roi_exists.append(True)
                    if self.loss_head:
                        for ihead, head in enumerate(self.loss_head):
                            if ir in head['rois']:
                                roi_labels[labels[..., channel] == value, ihead] = ir + 1
                                break
                    else:
                        roi_labels[labels[..., channel] == value] = ir + 1
                else:
                    roi_exists.append(False)
            else:
                if roi >= 256:
                    roi_exists.append(False) # Just in case there is type conversion in comparison
                else:
                    if np.amax(labels == roi) == 0:
                        roi_exists.append(False)
                    else:
                        roi_exists.append(True)
                if self.loss_head:
                    for ihead, head in enumerate(self.loss_head):
                        if ir in head['rois']:
                            roi_labels[labels == roi, ihead] = ir + 1
                            break
                else:
                    roi_labels[labels == roi] = ir + 1
        
        for i in range(len(enlarged_im_size)):
            if images.shape[i] < enlarged_im_size[i] + 1:
                padding = ((0, 0),) * i + ((0, enlarged_im_size[i] + 1 - images.shape[i]),)
                padding += ((0, 0),) * (len(images.shape) - i - 1)
                images = np.pad(images, padding, mode='constant')
                padding = ((0, 0),) * i + ((0, enlarged_im_size[i] + 1 - roi_labels.shape[i]),)
                padding += ((0, 0),) * (len(roi_labels.shape) - i - 1)
                roi_labels = np.pad(roi_labels, padding, mode='constant')
        
        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)

        # ROI specific augmentation
        if augmentation:
            clip_range = [0, 1]
            for roi in self.aug_rois_config['rois']:
                # Here labels are roi_labels
                if np.random.uniform() < self.aug_rois_config['p_contrast']:
                    scale = self.aug_rois_config['contrast_scale']
                    rand_scale = np.random.uniform(scale[0], scale[1])
                    images[roi_labels == self.rois.index(roi) + 1] *= rand_scale
                    # Clip
                    images = np.clip(images, clip_range[0], clip_range[1])
        
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

            # Fill nan by -2
            nan_mask = np.isnan(images)
            if np.any(nan_mask):
                images[nan_mask] = -2

        # Channel dropout
        if self.channel_dropout:
            # 4D image
            already_missed = []
            for ch in range(images.shape[3]):
                # If all values are nan for this channel
                if np.amin(np.isnan(images[..., ch])):
                    already_missed.append(ch)
            
            # Relative probably to keep n channels
            # e.g. for input_channel = 4, p(n = 1) = 0.1, p(n = 2) = 0.2, p(n = 3) = 0.3, p(n = 4) = 0.4
            n_max = self.input_channels - len(already_missed)
            p = range(1, n_max + 1)
            p = p / np.sum(p)
            # Randomly sample how many channels to keep
            n_keep = np.random.choice(range(1, n_max + 1), p=p)
            # Randomly sample which channels to keep
            keeped_channels = np.random.choice([f for f in range(self.input_channels) if f not in already_missed], 
                                               n_keep, replace=False)
            # Replace non-kept channels to -2
            for ch in range(images.shape[3]):
                if ch not in keeped_channels:
                    images[..., ch] = -2

        x_range = full_size[0] - enlarged_im_size[0]
        y_range = full_size[1] - enlarged_im_size[1]
        z_range = full_size[2] - enlarged_im_size[2]

        x_offset = int(enlarged_im_size[0] / 2)
        y_offset = int(enlarged_im_size[1] / 2)
        z_offset = int(enlarged_im_size[2] / 2)

        if len(roi_labels.shape) == 4:
            random_channel = np.random.randint(roi_labels.shape[-1])
            la = roi_labels[x_offset : x_offset + x_range, y_offset : y_offset + y_range, z_offset : z_offset + z_range, 
                            random_channel]
        else:
            la = roi_labels[x_offset : x_offset + x_range, y_offset : y_offset + y_range, z_offset : z_offset + z_range]

        # Get sampling prob
        # Random sampling ? of the time and always choose fg the rest of the time
        if np.random.random() > self.fg_sampling_ratio:
            # choose random
            o = np.random.choice(x_range * y_range * z_range)
        else:
            exist_classes = np.unique(la)[1:].astype(int)  # except bg:0
            # choose foreground of this class
            p = np.zeros((x_range, y_range, z_range), dtype=np.float32)
            if len(exist_classes) > 0:
                # choose class
                if class_balanced:
                    if self.enhanced_sampling:
                        enhanced_classes = [self.rois.index(r) + 1 for r in self.enhanced_rois]
                        intersection = set(enhanced_classes) & set(exist_classes)
                        intersection = list(intersection)
                        if len(intersection) > 0 and np.random.random() < 0.75:
                            selected_class = np.random.choice(intersection)
                        else: # enhanced rois are not found, or 25% choices for other classes
                            selected_class = np.random.choice(exist_classes)
                    else:
                        selected_class = np.random.choice(exist_classes)
                    p[la == selected_class] = 1
                else:
                    p[la > 0] = 1
            else:
                # if foreground is not present (gives NaN value for p)
                p = np.ones((x_range, y_range, z_range), dtype=np.float32)
            p = p.flatten() / np.sum(p)
            o = np.random.choice(x_range * y_range * z_range, p=p)
        x_start, y_start, z_start = np.unravel_index(o, (x_range, y_range, z_range))

        # Random shift
        if self.aug_config.get('random_shift', False) and np.random.random() > 0.5:
            # ±25% patch size shift
            random_shift = np.random.randint(-np.array(im_size) // 4, np.array(im_size) // 4 + 1)
            x_start, y_start, z_start = np.clip(np.array([x_start, y_start, z_start]) + random_shift, 0, np.array(la.shape) - 1)
        
        images_extracted = images[x_start : x_start + enlarged_im_size[0], y_start : y_start + enlarged_im_size[1],
                                  z_start : z_start + enlarged_im_size[2], ...]  # Maybe 4 dims
        labels_extracted = roi_labels[x_start : x_start + enlarged_im_size[0], y_start : y_start + enlarged_im_size[1],
                                      z_start : z_start + enlarged_im_size[2], ...]
        
        if augmentation:
            images_extracted, labels_extracted = self.perform_augmentation(images_extracted, labels_extracted, 
                                                                           labels_extracted.shape[:3])

        x_border_width = int((enlarged_im_size[0] - im_size[0]) / 2)
        y_border_width = int((enlarged_im_size[1] - im_size[1]) / 2)
        z_border_width = int((enlarged_im_size[2] - im_size[2]) / 2)

        images_extracted = images_extracted[x_border_width : x_border_width + im_size[0], 
                                            y_border_width : y_border_width + im_size[1], 
                                            z_border_width : z_border_width + im_size[2], ...]  # Maybe 4 dims
        labels_extracted = labels_extracted[x_border_width : x_border_width + im_size[0], 
                                            y_border_width : y_border_width + im_size[1], 
                                            z_border_width : z_border_width + im_size[2], ...]

        final_labels = np.zeros(im_size + [nclass], dtype=np.float32)
        loss_weights = np.ones(im_size + [nclass], dtype=np.float32)
        if self.loss_head:
            # labels_extracted: 0 - bg, 1 - first roi, ...
            # final_labels: 0 - first roi, ..., bg head_0, bg head_1, ...
            for ihead, head in enumerate(self.loss_head):
                if head['activation'] == 'softmax':
                    head_rois_indices = head['rois'][1:]
                    head_bg_index = head['rois'][0]
                    for z in head_rois_indices:
                        final_labels[labels_extracted[..., ihead] == z + 1, z] = 1
                        if not roi_exists[z]:
                            loss_weights[..., z] = 1e-5  # Avoid nan loss
                            loss_weights[..., head_bg_index] = 1e-5
                    final_labels[..., head_bg_index] = np.amax(final_labels[..., head_rois_indices], axis=3) == 0
                elif head['activation'] == 'sigmoid':
                    for z in head['rois']:
                        final_labels[labels_extracted[..., ihead] == z + 1, z] = 1
                        if not roi_exists[z]:
                            loss_weights[..., z] = 1e-5
        else:
            # final_labels: 0 - bg, 1 - first roi, ...
            for z in range(1, nclass):
                final_labels[labels_extracted == z, z] = 1
                if not roi_exists[z - 1]:
                    loss_weights[..., z] = 1e-5
                    loss_weights[..., 0] = 1e-5
            final_labels[..., 0] = np.amax(final_labels[..., 1:], axis=3) == 0

            # In case one voxel is assigned to 1 more than 1 times (very rare)
            final_labels = final_labels / np.expand_dims(np.sum(final_labels, axis=3), axis=-1)
        
        return images_extracted, final_labels, loss_weights
    
    def read_testing_inputs(self, file, im_size, nstride=[1, 1, 1], drop_channels=[], guided=False):
        with h5py.File(file, 'r') as f_h5:
            images = np.asarray(f_h5['images'], dtype=np.float32)
            if guided:
                from utils_detection_3d import extract_bboxes_from_annotation
                annotation = json.loads(f_h5.attrs['annotation'])
                bboxes = np.asarray(extract_bboxes_from_annotation(annotation), np.int32)
                guided_centers = bboxes[:, :3] + bboxes[:, 3:] / 2
            
        if len(images.shape) == 4:
            full_size = list(images.shape)[:-1]
        else:
            full_size = list(images.shape)
        
        if guided:
            steps = [list(np.clip((center - np.array(im_size) / 2).astype(int), 0, 
                                  np.maximum((np.array(full_size) - np.array(im_size)).astype(int), 0)
                                 )) for center in guided_centers]
        else:
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

            # Fill nan by -2
            nan_mask = np.isnan(images)
            if np.any(nan_mask):
                images[nan_mask] = -2
        
        # Channel dropout during testing
        if len(drop_channels) > 0:
            images[..., drop_channels] = -2
        
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
        images = np.empty((self.im_size[0], self.im_size[1], self.im_size[2], self.input_channels), dtype=np.float32)
        labels = np.empty((self.im_size[0], self.im_size[1], self.im_size[2], self.nclass), dtype=np.float32)
        weights = np.empty((self.im_size[0], self.im_size[1], self.im_size[2], self.nclass), dtype=np.float32)
        if self.input_channels == 1:
            images[..., 0], labels, weights = self.read_training_inputs(self.training_paths[i], self.rois, self.im_size, 
                                                                        self.enlarged_im_size)
        else:
            images, labels, weights = self.read_training_inputs(self.training_paths[i], self.rois, self.im_size, 
                                                                self.enlarged_im_size)
        return images, labels, weights

    def process_data_batch(self, q, idx_list):
        while True:
            shuffle_list = np.random.permutation(idx_list)
            images_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], 
                                     self.input_channels), dtype=np.float32)
            labels_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass),
                                     dtype=np.float32)
            weights_batch = np.empty((self.batch_size, self.im_size[0], self.im_size[1], self.im_size[2], self.nclass), 
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
    
    def train(self, run_validation=False, validation_paths=None, custom_log_file=None, guided=False, **kwargs):
        self.guided = guided
        # Filter training inputs
        if not self._training_paths_filtered:
            print('Filter training cases', len(self.training_paths), self.rois)
            self.training_paths, self.case_rois, self.median_shape = self.filter_training_cases(self.training_paths, 
                                                                                                self.rois)
            self._training_paths_filtered = True
            
            self.enhanced_training_idx = []
            if len(self.enhanced_rois) > 0:
                for i, cr in enumerate(self.case_rois):
                    if self.enhanced_method == 'intersection':
                        keep = True
                    elif self.enhanced_method == 'union':
                        keep = False
                    for er in self.enhanced_rois:
                        if self.enhanced_method == 'intersection':
                            if er not in cr:
                                keep = False
                        elif self.enhanced_method == 'union':
                            if er in cr:
                                keep = True
                    if keep:
                        self.enhanced_training_idx.append(i)
        
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
            'period': 20,  # save/validate per 20 training epochs
            'log_dir': log_dir,
            'save_fn': self.save, 
        }
        
        saver_callback = self.ModelSaver(saver_config=saver_config, validation_config=validation_config, 
                                         custom_log_file=custom_log_file)                
        
        # Deal with missing rois in training
        end_epoch_enhancement = self.epoch // 4
        num_samples = len(self.enhanced_training_idx)
        total_iters = self.steps_per_epoch * (end_epoch_enhancement - self.counter)
        if num_samples > 0 and total_iters > 0:
            data_generator = self.data_generator(self.enhanced_training_idx, total_iters)
            
            print('Running on a selected dataset with total training samples:', num_samples)
            self.fit(data_generator, validation_data=None, verbose=2,
                     steps_per_epoch=self.steps_per_epoch, initial_epoch=self.counter, epochs=end_epoch_enhancement,
                     callbacks=[tensorboard_callback, saver_callback])
            self.counter = end_epoch_enhancement
            self.save(end_epoch_enhancement)
            # Only serveral cases are enough to check
            num_keep_per_roi = 3
            if self.enhanced_method == 'union':
                val_idx = []
                count_enhanced_rois = [0 for _ in self.enhanced_rois]
                for i, cr in enumerate(self.case_rois):
                    keep = False
                    for j, er in enumerate(self.enhanced_rois):
                        if er in cr and count_enhanced_rois[j] < num_keep_per_roi:
                            count_enhanced_rois[j] += 1
                            keep = True
                    if keep:
                        val_idx.append(i)
            else: # self.enhanced_method == 'intersection'
                val_idx = self.enhanced_training_idx[:num_keep_per_roi]
            training_dice = self.validate([self.training_paths[i] for i in val_idx])
            print('training dice:', training_dice)
            enhanced_rois_idx = [self.rois.index(roi) for roi in self.enhanced_rois]
            if np.nanmin(training_dice[np.array(enhanced_rois_idx)]) > 0.05:
                self.enhanced_sampling = False # Switch off afterward
                pass
            else:  # all nan or min_dice < 0.05
                self.need_retrain = True
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir)
                ckpt_dir = os.path.join(self.checkpoint_dir, self.model_dir)
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                print('[Warning] Abort this training due to missing rois.')
                return
        
        num_samples = len(self.training_paths)
        if num_samples == 0:
            print('No training data')
            return
        idx_list = np.arange(num_samples)
        total_iters = self.steps_per_epoch * (self.epoch - self.counter)
        data_generator = self.data_generator(idx_list, total_iters)
        
        # print('Running on complete dataset with total training samples:', num_samples)
        end_epoch = self.counter
        if total_iters > 0:
            self.fit(data_generator, validation_data=None, verbose=2,
                     steps_per_epoch=self.steps_per_epoch, initial_epoch=self.counter, epochs=self.epoch,
                     callbacks=[tensorboard_callback, saver_callback])
        
            end_epoch = self.history.epoch
        if isinstance(end_epoch, list):
            end_epoch = end_epoch[-1] + 1
        else:
            end_epoch = 0
        
        if custom_log_file is not None:
            # Terminate by invalid loss (incremental learning)
            if end_epoch != self.epoch:
                self.need_retrain = True
                shutil.rmtree(log_dir)
                ckpt_dir = os.path.join(self.checkpoint_dir, self.model_dir)
                shutil.rmtree(ckpt_dir)
                return
        if total_iters > 0:
            self.save(end_epoch)
        
        print('Model self-validation. Checking all rois are not missing.')
        # Only serveral cases are enough to check
        num_keep_per_roi = 5
        val_idx = []
        count_rois = [0 for _ in self.rois]
        for i, cr in enumerate(self.case_rois):
            keep = False
            for j, er in enumerate(self.rois):
                if er in cr and count_rois[j] < num_keep_per_roi:
                    count_rois[j] += 1
                    keep = True
            if keep:
                val_idx.append(i)
        print('Based on', val_idx, 'with total', len(val_idx), 'cases.')
        training_dice = self.validate([self.training_paths[i] for i in val_idx])
        print('Checking training roi dice:', training_dice)
        if np.nanmin(training_dice) > 0.05:
            pass
        else:  # all nan or min_dice < 0.05
            print('[Warning] Missing rois exist in this model. Consider to retrain.')
            print('Setting enhanced_rois, or increasing fg_sampling_ratio, or decreasing sgd_momentum may help.')
            ckpt_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            if os.path.exists(log_dir):
                trash_log_dir = os.path.join(self.log_dir, self.model_dir + '-MissingROIs')
                shutil.move(log_dir, trash_log_dir)
                print('Logs are moved from', log_dir, '->', trash_log_dir)
            if os.path.exists(ckpt_dir):
                trash_ckpt_dir = os.path.join(self.checkpoint_dir, self.model_dir + '-MissingROIs')
                shutil.move(ckpt_dir, trash_ckpt_dir)
                print('Model checkpoint is moved from', ckpt_dir, '->', trash_ckpt_dir)
        
        return

    @staticmethod
    def get_gaussian_map_importance(im_size):
        # Calculate Gaussian map importance
        sigma_scale = 1. / 8
        tmp = np.zeros(im_size)
        center_coords = [i // 2 for i in im_size]
        sigmas = [i * sigma_scale for i in im_size]
        tmp[tuple(center_coords)] = 1
        gaussian_map_importance = ndimage.filters.gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_map_importance = gaussian_map_importance / np.amax(gaussian_map_importance)
        gaussian_map_importance = gaussian_map_importance.astype(np.float32)
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        min_value = np.min(gaussian_map_importance[gaussian_map_importance != 0])
        gaussian_map_importance[gaussian_map_importance == 0] = min_value
        return gaussian_map_importance
    
    def validate(self, validation_paths, return_uncertainty_metrics=False):
        test_dice_mean, test_uncertainty_mean = 0., 0.
        dice_scores, uncertainty_metrics = [], []
        test_nstride = self.test_nstride

        # Remove previous inference model
        self.inference_model = None
        for validation_path in validation_paths:
            predictions = self.run_test(validation_path, test_nstride)
            with h5py.File(validation_path, 'r') as f_h5:
                labels = np.asarray(f_h5['labels'], dtype=np.float32)

            dice = []
            for ir, roi in enumerate(self.rois):
                if roi >= 256:
                    channel = np.floor_divide(roi, 256)
                    if channel >= labels.shape[-1]:
                        gt = np.zeros(predictions.shape[:3], dtype=np.float32)
                    else:
                        channel_labels = labels[..., channel]
                        gt = (channel_labels == roi - 256 * channel).astype(np.float32)
                else:
                    if len(labels.shape) == 4:
                        channel_labels = labels[..., 0]
                    else:
                        channel_labels = labels
                    gt = (channel_labels == roi).astype(np.float32)

                if self.loss_head:
                    model = np.zeros(predictions.shape[:3], dtype=np.float32)
                    for ihead, head in enumerate(self.loss_head):
                        if ir in head['rois']:
                            model = (predictions[..., ihead] == head['rois'].index(ir)).astype(np.float32)
                            break
                else:
                    model = (predictions == ir + 1).astype(np.float32)
                if np.amax(gt) == 0:
                    dice.append(np.nan)
                else:
                    d = 2.0 * np.sum(gt * model) / (np.sum(gt) + np.sum(model))
                    dice.append(d)
            dice_scores.append(dice)
            if return_uncertainty_metrics:
                pred_probs = self.run_test(validation_path, test_nstride, output_prob=True)
                predictions = np.argmax(pred_probs, axis=3)
                uncertainty = []
                for ir, roi in enumerate(self.rois):
                    pred_prob = pred_probs[..., ir + 1] / np.sum(pred_probs, axis=3) # softmax probability
                    model = (predictions == ir + 1).astype(np.float32)
                    if np.amax(model) == 0:
                        uncertainty.append(np.nan)
                    else:
                        uncertainty.append(np.mean(pred_prob[predictions == ir + 1]))
                uncertainty_metrics.append(uncertainty)

        dice_scores = np.asarray(dice_scores, dtype=np.float32)
        test_dice_mean = np.nanmean(dice_scores, axis=0)
        # Keep the same shape (n,)
        if len(test_dice_mean.shape) == 0:
            test_dice_mean = np.array([test_dice_mean])

        if not return_uncertainty_metrics:
            return test_dice_mean
        else:
            uncertainty_metrics = np.asarray(uncertainty_metrics, dtype=np.float32)
            test_uncertainty_mean = np.nanmean(uncertainty_metrics, axis=0)    
            if len(test_uncertainty_mean.shape) == 0:
                test_uncertainty_mean = np.array([test_uncertainty_mean])
            return test_dice_mean, test_uncertainty_mean
    
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

    @staticmethod
    def patch_segmentation_and_restore(pred_model, all_patches, info, overlap_count=None, test_config={}, debug=False):
        full_size = info['full_size']
        pad_size = info['pad_size']
        pads = [pad_size[ax] - full_size[ax] for ax in range(len(full_size))]
        steps = info['steps']
        im_size = info['im_size']
        if debug:
            s = time.time()
            print('[DEBUG] patch_segmentation_and_restore', im_size, info)
    
        # Prediction
        # For collecting probabilities
        label_probs = np.zeros(list(pad_size) + [pred_model.output_shape[-1],], dtype=np.float32)
        # Auto find best batch_size until GPU is saturated
        # Otherwise, larger batch_size cannot speed up
        # 16 * 16 * 64 * 64: Safe and fast. It only takes up to 5.5GB GPU memory for UNet3D inference.
        if 'batch_size' in test_config:
            batch_size = test_config['batch_size']
        else: # Auto
            batch_size = max(1, int((16 * 16 * 64 * 64) // np.prod(im_size)))
        n_patches = all_patches.shape[0]
        n_batches = (n_patches + batch_size - 1) // batch_size
        for n in range(n_batches):
            # Avoid memory leakage. 
            # Converting the numpy array to a tensor maintains the same signature and avoids creating new graphs.
            tensor_im = tf.convert_to_tensor(all_patches[n * batch_size : (n + 1) * batch_size, ...], dtype=tf.float32)
            probs = pred_model(tensor_im, training=False) # Keep the first dimension
            if debug:
                print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                      f"predict_on_batch (batch_size={batch_size}) batch # {n}")
            
            mirror_config = test_config.get('mirror_config', {})
            if mirror_config.get('testing_mirror', False):
                mirror_axes = mirror_config.get('mirror_axes', []) # e.g. [1, 2] # axis:0 is batch
                mirror_axes_comb = [[]]
                for ax in mirror_axes:  # get powerset
                    mirror_axes_comb += [sub + [ax] for sub in mirror_axes_comb]
                
                mirror_axes_comb = mirror_axes_comb[1:]  # e.g. [[1], [2], [1, 2]]
                count = 1.
                for axis in mirror_axes_comb:
                    probs_mirror = pred_model(tf.reverse(tensor_im, axis=axis), training=False)
                    probs_mirror = tf.reverse(probs_mirror, axis=axis)
                    probs += probs_mirror
                    count += 1.
                    if debug:
                        print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                              f"predict_on_batch mirror axis {axis} batch #", n)
                probs /= count  # actually same results after argmax, keep it for retraining same range.

            # Not compatible with mirror_config
            left_right_swap_config = test_config.get('left_right_swap_config', {})
            if left_right_swap_config.get('testing_flip', False):  # In RadAI
                swap_left_right = left_right_swap_config['fn']
                # Left right swap, not completely same as INTContour, the swap_pairs should follow the model output roi order
                prob_swap_pairs = left_right_swap_config.get('swap_pairs', [])
                
                probs_flip = pred_model(tf.reverse(tensor_im, axis=[3]), training=False) # axes: [bs, SI, AP, LR, ch]
                if debug:
                    print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                          'predict_on_batch left_right_swap (RadAI) batch #', n)
                probs_flip = tf.reverse(probs_flip, axis=[3])
                probs_flip = swap_left_right(probs_flip, prob_swap_pairs, True)
                probs = (probs + probs_flip) / 2.0
            elif left_right_swap_config.get('left_right_swap', False):
                swap_left_right = left_right_swap_config['fn']
                # INTContour
                swap_pairs = left_right_swap_config.get('swap_pairs', [])
                group_rois = left_right_swap_config.get('group_rois', [])
                prob_swap_pairs = []
                # Change to pairs of the roi
                for pair in swap_pairs:
                    if pair[0] in group_rois and pair[1] in group_rois:
                        prob_swap_pairs.append((group_rois.index(pair[0]) + 1, group_rois.index(pair[1]) + 1))

                probs_flip = pred_model(tf.reverse(tensor_im, axis=[3]), training=False) # axes: [bs, SI, AP, LR, ch]
                if debug:
                    print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                          "predict_on_batch left_right_swap (INTContour) batch #", n)
                probs_flip = tf.reverse(probs_flip, axis=[3])
                probs_flip = swap_left_right(probs_flip, prob_swap_pairs, True)
                probs = (probs + probs_flip) / 2.0
            
            # Restore testing labels
            for ib in range(batch_size):
                i_step = n * batch_size + ib
                if i_step >= len(steps):
                    continue
                x_step, y_step, z_step = steps[i_step]
                x_slice = slice(x_step, x_step + im_size[0])
                y_slice = slice(y_step, y_step + im_size[1])
                z_slice = slice(z_step, z_step + im_size[2])
                patch_prob = probs[ib]
    
                if overlap_count is not None:
                    # Vectorized division
                    patch_overlap = overlap_count[x_slice, y_slice, z_slice]
                    patch_prob = patch_prob / patch_overlap[..., None]
                
                label_probs[x_slice, y_slice, z_slice, :] += patch_prob
            if debug:
                print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                      "restore_testing_labels batch #", n)
    
        label_probs = label_probs[pads[0] // 2 : pads[0] // 2 + full_size[0],
                                  pads[1] // 2 : pads[1] // 2 + full_size[1], 
                                  pads[2] // 2 : pads[2] // 2 + full_size[2], :]
        
        if debug:
            print('[DEBUG] patch_segmentation_and_restore |', round(time.time() - s, 2), 's |', 
                  "restore_testing_labels")
        
        output_prob = test_config.get('output_prob', False)
        loss_head = test_config.get('loss_head')
        if output_prob:
            return label_probs
        
        if loss_head:
            output_labels = []
            start = 0
            for head in loss_head:
                end = start + len(head['rois'])
                output_labels.append(np.argmax(label_probs[..., start:end], axis=3).astype(np.uint8))
                start = end
            output_labels = np.stack(output_labels, axis=-1)
            return output_labels
        else:
            output_labels = np.argmax(label_probs, axis=3).astype(np.uint8)
            return output_labels

    def test(self, testing_paths, output_path, drop_channels=[], output_prob=False, use_gaussian=True, guided=False, **kwargs):
        self.guided = guided
        if not self._loaded:
            raise Exception('No model is found, please train first')
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Remove previous inference model
        self.inference_model = None
        
        if output_prob:
            output_dtype = np.float32
        else:
            output_dtype = np.uint8
        for input_file in tqdm(testing_paths):
            output_array = self.run_test(input_file, self.test_nstride, drop_channels, output_prob=output_prob)
            output_file = os.path.join(output_path, self.model_dir + '_' + os.path.basename(input_file))
            with h5py.File(output_file, 'w') as f_h5:
                f_h5['predictions'] = output_array.astype(output_dtype)
                
    def run_test(self, input_file, test_nstride, drop_channels=[], output_prob=False):
        if self.guided:
            test_nstride = [1, 1, 1]

        images = np.empty((1, self.im_size[0], self.im_size[1], self.im_size[2], self.input_channels), dtype=np.float32)
        all_patches, info = self.read_testing_inputs(input_file, self.im_size, test_nstride, drop_channels, self.guided)

        full_size = info['full_size']
        pad_size = info['pad_size']
        pads = [pad_size[ax] - full_size[ax] for ax in range(len(full_size))]
        steps = info['steps']
        im_size = self.im_size
        strides = test_nstride
        loss_head = self.loss_head
        
        if len(all_patches.shape) == len(im_size) + 1:
            all_patches = np.expand_dims(all_patches, -1)

        # Get patch overlap count
        overlap_count = np.zeros(pad_size, dtype=np.float32)
        for x_step, y_step, z_step in steps:
            overlap_count[x_step : x_step + im_size[0], 
                          y_step : y_step + im_size[1], 
                          z_step : z_step + im_size[2]] += 1
        overlap_count = np.where(overlap_count == 0, 1, overlap_count) # avoid zero division

        if self.inference_model is None:
            if max(strides) > 1:
                gaussian_map_importance = self.get_gaussian_map_importance(im_size)
            else:
                gaussian_map_importance = None
            
            logits = self.unet.get_layer('logits_0').get_output_at(0)
            if loss_head:
                outputs = []
                for head in loss_head:
                    head_outputs = tf.gather(logits, indices=head['rois'], axis=-1)
                    if head['activation'] == 'softmax':
                        head_outputs = tf.nn.softmax(head_outputs)
                    elif head['activation'] == 'sigmoid':
                        head_outputs = tf.nn.sigmoid(head_outputs)
                    outputs.append(head_outputs)
                outputs = tf.concat(outputs, axis=-1)
            else:
                outputs = tf.nn.softmax(logits)
            if gaussian_map_importance is not None:
                outputs = outputs * gaussian_map_importance[None, ..., None]
            self.inference_model = tf.keras.models.Model(inputs=self.unet.input, outputs=outputs)
        
        info['im_size'] = im_size
        info['strides'] = strides
        test_config = {}
        if self.left_right_swap_config is not None:
            left_right_swap_config = {
                'left_right_swap': self.left_right_swap_config.get('testing_flip', False),
                'swap_pairs': self.left_right_swap_config.get('swap_pairs', []),
                'group_rois': self.rois,
                'fn': self.swap_left_right
            }
            test_config['left_right_swap_config'] = left_right_swap_config
        if self.mirror_config is not None:
            mirror_axes = self.mirror_config.get('mirror_axes', [])
            if self.mirror_config.get('mirror_all_dimensions', False):
                mirror_axes = [1, 2, 3]
            mirror_config = {
                'testing_mirror': self.mirror_config.get('testing_mirror', False),
                'mirror_axes': mirror_axes
            }
            test_config['mirror_config'] = mirror_config
        test_config['output_prob'] = output_prob
        if loss_head:
            test_config['loss_head'] = loss_head
        output_labels = self.patch_segmentation_and_restore(self.inference_model, all_patches, info, 
                                                            test_config=test_config,
                                                            overlap_count=overlap_count, 
                                                            debug=False)
        return output_labels

    def adaptive_test(self, pre_path, post_paths, output_path, first_run):
        if isinstance(post_paths, (str, pathlib.Path)):
            post_paths = [post_paths]
        
        if first_run:
            opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.99, clipnorm=12, nesterov=True)
            self.compile(optimizer=opt, loss=self.loss_fn)
        else:
            # Load weights
            checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
            ckpt_file = ckpt_files[-1]
            self.unet.load_weights(ckpt_file)
        
        # Pre-adaptive
        self.test(post_paths, os.path.join(output_path, 'pre'))
        # Adaptive learning
        self.training_paths = [pre_path]
        num_samples = 1

        self.fit(self.data_generator(num_samples), validation_data=None, verbose=2, steps_per_epoch=100, epochs=1)
        # Post-adaptive
        self.test(post_paths, os.path.join(output_path, 'post'))

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
                        log_message = 'epoch: %d, ' % (epoch + 1)
                        log_message += 'dice: %.4f, loss: %.4f' % (record_logs['dice'], record_logs['loss'])
                        writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'), log_message])
                        # ROI logs
                        log_message = 'roi_dices: '
                        for roi in self.rois:
                            roi_dice = record_logs.get(f'roi_{roi}', float('nan'))
                            log_message += str(round(roi_dice, 4))
                            if roi != self.rois[-1]:
                                log_message += ', '
                        writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'), log_message])
                        if np.isnan(record_logs['loss']) or np.isinf(record_logs['loss']):
                            writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'), 
                                             'Invalid loss, terminating training.'])
                            self.model.stop_training = True
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
            
            # Avoid memory leakage
            gc.collect()