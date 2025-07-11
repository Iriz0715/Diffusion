import numpy as np
from builtins import range
import random
from batchgenerators.augmentations.utils import *
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares
from scipy.ndimage import gaussian_filter
from batchgenerators.augmentations.utils import uniform
from collections import OrderedDict

def seg_channel_selection_transform(label, channels):
    return label[:, channels]
    
def augment_spatial_2(data, seg, patch_size, patch_center_dist_from_border=30,
                      do_elastic_deform=True, deformation_scale=(0, 0.25),
                      do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                      do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                      border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                      p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                      p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
   
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if np.random.uniform() < p_el_per_sample and do_elastic_deform:
            mag = []
            sigmas = []

            # one scale per case, scale is in percent of patch_size
            def_scale = np.random.uniform(deformation_scale[0], deformation_scale[1])

            for d in range(len(data[sample_id].shape) - 1):
                # transform relative def_scale in pixels
                sigmas.append(def_scale * patch_size[d])

                # define max magnitude and min_magnitude
                max_magnitude = sigmas[-1] * (1 / 2)
                min_magnitude = sigmas[-1] * (1 / 8)

                # the magnitude needs to depend on the scale, otherwise not much is going to happen most of the time.
                # we want the magnitude to be high, but not higher than max_magnitude (otherwise the deformations
                # become very ugly). Let's sample mag_real with a gaussian
                # mag_real = np.random.normal(max_magnitude * (2 / 3), scale=max_magnitude / 3)
                # clip to make sure we stay reasonable
                # mag_real = np.clip(mag_real, 0, max_magnitude)

                mag_real = np.random.uniform(min_magnitude, max_magnitude)

                mag.append(mag_real)
            # print(np.round(sigmas, decimals=3), np.round(mag, decimals=3))
            coords = elastic_deform_coordinates_2(coords, sigmas, mag)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:
            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            # recenter coordinates
            coords_mean = coords.mean(axis=tuple(range(1, len(coords.shape))), keepdims=True)
            coords -= coords_mean

            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result
    
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1), p_per_sample=1):
    if np.random.uniform() < p_per_sample:                                         
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample
    
def augment_gaussian_blur(data_sample, sigma_range, per_channel=True, p_per_channel=1, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not per_channel:
            sigma = get_range_val(sigma_range)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                if per_channel:
                    sigma = get_range_val(sigma_range)
                data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample

def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        if not per_channel:
            data_sample *= multiplier
        else:
            for c in range(data_sample.shape[0]):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                data_sample[c] *= multiplier
    return data_sample
    
def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not per_channel:
            mn = data_sample.mean()
            if preserve_range:
                minm = data_sample.min()
                maxm = data_sample.max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample = (data_sample - mn) * factor + mn
            if preserve_range:
                data_sample[data_sample < minm] = minm
                data_sample[data_sample > maxm] = maxm
        else:
            for c in range(data_sample.shape[0]):
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                data_sample[c] = (data_sample[c] - mn) * factor + mn
                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample
    
def augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if not isinstance(zoom_range, (list, tuple, np.ndarray)):
            zoom_range = [zoom_range]

        shp = np.array(data_sample.shape[1:])
        dim = len(shp)

        if not per_channel:
            if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                assert len(zoom_range) == dim
                zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
            else:
                zoom = uniform(zoom_range[0], zoom_range[1])

            target_shape = np.round(shp * zoom).astype(int)

            if ignore_axes is not None:
                for i in ignore_axes:
                    target_shape[i] = shp[i]

        if channels is None:
            channels = list(range(data_sample.shape[0]))

        for c in channels:
            if np.random.uniform() < p_per_channel:
                if per_channel:
                    if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                        assert len(zoom_range) == dim
                        zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
                    else:
                        zoom = uniform(zoom_range[0], zoom_range[1])

                    target_shape = np.round(shp * zoom).astype(int)
                    if ignore_axes is not None:
                        for i in ignore_axes:
                            target_shape[i] = shp[i]

                downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                     anti_aliasing=False)
                data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge', anti_aliasing=False)

    return data_sample  

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False, p_per_sample=1):
    if np.random.uniform() < p_per_sample:
        if invert_image:
            data_sample = - data_sample
        if not per_channel:
            if retain_stats:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats:
                data_sample = data_sample - data_sample.mean()
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
                data_sample = data_sample + mn
        else:
             for c in range(data_sample.shape[0]):
                if retain_stats:
                    mn = data_sample[c].mean()
                    sd = data_sample[c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample[c].min()
                rnge = data_sample[c].max() - minm
                data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats:
                    data_sample[c] = data_sample[c] - data_sample[c].mean()
                    data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                    data_sample[c] = data_sample[c] + mn
        if invert_image:
            data_sample = - data_sample
    return data_sample
                                                     
def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            'Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either '
            '[channels, x, y] or [channels, x, y, z]')
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg

def augment_mirroring_with_bbox(sample_data, bboxes, axes=(0, 1, 2), p_mirror=0.5):
    """
    sample_data: with shape (channels, x, y, z) or (channels, x, y)
    bboxes: 2D array. [[x_center, y_center, z_center, x_range, y_range, z_range]] or [[x_center, y_center, x_range, y_range]]
    axes: the mirror axes
    """
    sample_data, bboxes = np.asarray(sample_data, np.float32), np.asarray(bboxes, np.float32)
    patch_size = sample_data.shape[1:]
    dim = len(patch_size)
    shp = np.asarray(patch_size, np.float32)
    mirror_center = shp - bboxes[:, :dim] - 1
    
    data_result = sample_data.copy()
    bboxes_result = bboxes.copy()
    if 0 in axes and np.random.uniform() < p_mirror:
        data_result = data_result[:, ::-1]
        bboxes_result[:, 0] = mirror_center[:, 0]
    if 1 in axes and np.random.uniform() < p_mirror:
        data_result = data_result[:, :, ::-1]
        bboxes_result[:, 1] = mirror_center[:, 1]
    if 2 in axes and len(sample_data.shape) == 4 and np.random.uniform() < p_mirror:
        data_result = data_result[:, :, :, ::-1]
        bboxes_result[:, 2] = mirror_center[:, 2]
    return data_result, bboxes_result

def augment_scaling_with_bbox(sample_data, bboxes, scale,
                              independent_scale_for_each_axis=False,
                              p_scale_per_sample=0.2,
                              p_independent_scale_per_axis=1.0,
                              order_data=3, 
                              border_mode_data='nearest', 
                              border_cval_data=0.):
    """
    sample_data: with shape (channels, x, y, z) or (channels, x, y)
    bboxes: 2D array. [[x_center, y_center, z_center, x_range, y_range, z_range]] or [[x_center, y_center, x_range, y_range]]
    scale: the scale ratio of coordinates. 2.0 means coordinates expand 2 times. bbox will be smaller.
    """
    sample_data, bboxes = np.array(sample_data), np.array(bboxes)
    data_result = sample_data.copy()
    bboxes_result = bboxes.copy()
    if np.random.uniform() < p_scale_per_sample:
        patch_size = sample_data.shape[1:]
        dim = len(patch_size)
        coords = create_zero_centered_coordinate_mesh(patch_size)
        if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)

        for d in range(dim):
            ctr = int(np.round(patch_size[d] / 2.))
            coords[d] += ctr
        
        shp = np.array(patch_size)
        new_center = np.clip(shp / 2 - (shp / 2 - np.array(bboxes)[:, :dim]) / sc, 0, shp)
        bboxes_result[:, :dim] = new_center
        bboxes_result[:, dim:] = np.clip(bboxes[:, dim:] / sc, 0, shp)
        bboxes_result.astype(bboxes.dtype)
        for channel_id in range(sample_data.shape[0]):
            data_result[channel_id] = map_coordinates(sample_data[channel_id].astype(float), coords,
                                                      order=order_data, mode=border_mode_data,
                                                      cval=border_cval_data).astype(sample_data.dtype)
    return data_result, bboxes_result