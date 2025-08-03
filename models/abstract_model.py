import os, glob, re, datetime, csv
import numpy as np
import tensorflow as tf

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

class AbstractModel(tf.keras.models.Model):
    def save(self, step, max_to_keep=1):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
        if len(ckpt_files) >= max_to_keep:
            os.remove(ckpt_files[0])
        
        self.networks.save(os.path.join(checkpoint_dir, 'model_epoch%06d.h5' % step))
    
    def load(self, force_load=False):
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
            try:
                self.networks.load_weights(ckpt_file)
            except:
                if force_load:
                    self.networks = tf.keras.models.load_model(ckpt_file, compile=False)
                    print('''[Warning] Current architecture is different with the loading model. 
                          Force loading the model. Press Ctrl+C to stop if it is unexpected.''')
                else:
                    print('''[Warning] Current architecture is different with the loading model. 
                          Trying to transfer and retrain. Press Ctrl+C to stop if it is unexpected.''')
                    self.load_weights_for_transfer_learning(ckpt_file, ignore_keywords=[])
                    self.counter = 0
            print('Loaded model checkpoint:', ckpt_name)
            return True, self.counter
        else:
            print('Failed to find a checkpoint')
            return False, 0
    
    @staticmethod
    def get_strides_list(layer_number, im_size, mode='symmetric'):
        # mode: symmetric. e.g. [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        # mode: last. e.g. [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
        s_list = np.zeros([layer_number, len(im_size)], dtype=np.uint8)
        for i in range(layer_number):
            # stop pooling when size is odd && size <= 4
            to_pool = (np.array(im_size) % 2 ** (i + 1) == 0).astype(int)
            to_pool *= ((np.array(im_size) // 2 ** (i + 1)) > 4).astype(int)
            s_list[i] = 1 + to_pool
        if mode == 'symmetric':
            strides_list = np.concatenate([s_list[::-2], s_list[layer_number % 2::2]])
        else:
            strides_list = np.array(s_list)
        return strides_list
    
    @staticmethod
    def append_to_log_file(logs, custom_log_file):
        if custom_log_file is not None:
            if not os.path.exists(custom_log_file):
                with open(custom_log_file, 'w') as f_0:
                    writer = csv.writer(f_0, delimiter=';')
                    writer.writerow(['time', 'value'])
            with open(custom_log_file, 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([datetime.datetime.now().strftime('%d/%b/%Y %H:%M:%S'), logs])

    def load_weights_for_transfer_learning(self, old_model_file, custom_log_file=None, ignore_keywords=['logits']):
        # ignore_keywords: the keyword included in the layer name
        
        # load weights by name and skip mismatch weights shape
        pretrain_m = tf.keras.models.load_model(old_model_file, compile=False)
        for l in self.networks.layers:
            ignored = False
            for k in ignore_keywords:
                if k in l.name:
                    # self.append_to_log_file(f'{l.name} No transfer. [Ignored]', custom_log_file)
                    ignored = True
            if ignored:
                continue
            try:
                self.networks.get_layer(name=l.name).set_weights(pretrain_m.get_layer(name=l.name).get_weights())
                # self.append_to_log_file(f'{l.name} Successfully transferred.', custom_log_file)
            except:
                # self.append_to_log_file(f'{l.name} No transfer. [Incompatible shapes]', custom_log_file)
                continue
        
        # self.append_to_log_file(f'Loaded pre-trained model for transfer learning: {old_model_file}', custom_log_file)
        return
    
    def perform_augmentation(self, images, labels, patch_size, dim=3, augmentation_params={}):
        AUGMENTATION_PARAMS.update(augmentation_params)
        if dim == 3:
            if len(labels.shape) == 4:
                labels_aug = np.expand_dims(np.transpose(labels, (3, 0, 1, 2)), axis=0)
            else:
                labels_aug = np.expand_dims(labels, axis=(0, 1))
            if len(images.shape) == 4:
                images_aug = np.expand_dims(np.transpose(images, (3, 0, 1, 2)), axis=0)
            else:
                images_aug = np.expand_dims(images, axis=(0, 1))
            if len(patch_size) == 4:
                patch_size = patch_size[:-1]
        elif dim == 2:
            if len(labels.shape) == 3:
                labels_aug = np.expand_dims(np.transpose(labels, (2, 0, 1)), axis=0)
            else:
                labels_aug = np.expand_dims(labels, axis=(0, 1))
            if len(images.shape) == 3:
                images_aug = np.expand_dims(np.transpose(images, (2, 0, 1)), axis=0)
            else:
                images_aug = np.expand_dims(images, axis=(0, 1))
            if len(patch_size) == 3:
                patch_size = patch_size[:-1]
        else:
            raise ValueError(f'Unexpected patch dimension {dim}')
        images_aug, labels_aug = augment_spatial_2(images_aug, labels_aug, patch_size=patch_size,
                                                   patch_center_dist_from_border=
                                                   AUGMENTATION_PARAMS.get('random_crop_dist_to_border'),
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
        if AUGMENTATION_PARAMS.get('dummy_2D') and len(images_aug.shape) == 4 and dim == 3:
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
            images_aug, labels_aug = augment_mirroring(images_aug, labels_aug)
        
        if dim == 3:
            if len(images.shape) == 4:
                images_aug = np.transpose(images_aug, (1, 2, 3, 0))
            else:
                images_aug = np.squeeze(images_aug, axis=0)
            if len(labels.shape) == 4:
                labels_aug = np.transpose(labels_aug, (1, 2, 3, 0))
            else:
                labels_aug = np.squeeze(labels_aug, axis=0)
        if dim == 2:
            if len(images.shape) == 3:
                images_aug = np.transpose(images_aug, (1, 2, 0))
            else:
                images_aug = np.squeeze(images_aug, axis=0)
            if len(labels.shape) == 3:
                labels_aug = np.transpose(labels_aug, (1, 2, 0))
            else:
                labels_aug = np.squeeze(labels_aug, axis=0)
        return images_aug, labels_aug