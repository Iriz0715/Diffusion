import os, sys, argparse, multiprocessing
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models'))
from models.ImageTransformUNet2D import ImageTransformUNet2D
from models.DDPM_UNet2D import DDPM_UNet2D
# from CycleGAN import CycleGAN
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--phase', type=int, default=0, help='0 - train, 1 - deploy')
parser.add_argument('-f', '--final', type=int, default=1, help='0 - Cross validation, 1 - Final')
parser.add_argument('-d', '--deploy_input', type=str, default='', help='Deploy input folder')
parser.add_argument('-m', '--model', type=str, default='DDPM', help='Model name [UNet2D, CycleGAN, DDPM]')
parser.add_argument('-c', '--cuda', type=int, default=0, help='Set visible cuda')
parser.add_argument('-t', '--thread', type=int, default=6, help='Set number of threads')
parser.add_argument('-i', '--identifier', type=str, default='DDPM', help='Identifier')
parser.add_argument('-r', '--resume',type=bool, default=False, help='Resume training from checkpoint')

# To do:
# 1. for abdomen CBCT, try to run a bias correction first to see if it can improve the performance
# 2. try physics-based CBCT augmentation, i.e., simulate CBCT artifacts from ground truth CT 
# https://github.com/nadeemlab/Physics-ArX

import numba
@numba.jit(nopython=True, parallel=True, fastmath=True)
def generate_noise(sigma_map, shape):
    noise_real = np.empty(shape, dtype=np.float32)
    noise_imag = np.empty(shape, dtype=np.float32)

    dims = len(shape)
    for i in numba.prange(shape[0]):
        for j in range(shape[1]):
            if dims == 3:
                for k in range(shape[2]):
                    std = sigma_map[i, j, k]
                    noise_real[i, j, k] = np.random.normal(0, std)
                    noise_imag[i, j, k] = np.random.normal(0, std)
            elif dims == 2:
                std = sigma_map[i, j]
                noise_real[i, j] = np.random.normal(0, std)
                noise_imag[i, j] = np.random.normal(0, std)
    return noise_real, noise_imag

def get_nonzero_range(mask, axis=0):
    axes = tuple([ax for ax in range(len(mask.shape)) if ax != axis])
    non_zero_axis = np.any(mask, axis=axes)
    if not np.any(non_zero_axis):
        min_range, max_range = 0, 0
    else:
        min_range, max_range = np.where(non_zero_axis)[0][[0, -1]]
    return min_range, max_range


def main(config):
    phase = config.phase
    final = config.final == 1
    deploy_input = config.deploy_input
    model_name = config.model
    num_threads = config.thread
    identifier = config.identifier
    resume = config.resume
    
    dir = '/mnt/newdisk/mri2ct' # Default directory for training data 

    if model_name not in ['UNet2D', 'CycleGAN','DDPM']:
        raise Exception('Unsupported model')
    
    if final:
        checkpoint_dir = identifier + '/checkpoint_T1000'
        log_dir = identifier + '/logs'
        # train_data_dir = identifier + '/data_training'    # 3个小数据集
        # train_data_dir = dir + '/data_training'   # 全部数据
        train_data_dir = dir + '/data_training/train'   ## 暂时只用训练集
        # train_data_dir = identifier + '/data_training/test_trainset' # 2个小数据集
        output_path = identifier + '/output/test'
    else:
        checkpoint_dir = identifier + '/checkpoint_cv'
        log_dir = identifier + '/logs_cv'
        # train_data_dir = identifier + '/data_training/train'
        train_data_dir = dir + '/data_training/train'
        output_path = identifier + '/output/cv/' + os.path.basename(deploy_input.strip('/'))
    
    if deploy_input == '':
        deploy_input = dir + '/data_training/test'
        # deploy_input = identifier + '/data_training' # 3个小数据集
        
    training_paths = [os.path.join(train_data_dir, name) for name in os.listdir(train_data_dir) if name.endswith('.hdf5')]
    testing_paths = [os.path.join(deploy_input, name) for name in os.listdir(deploy_input) if '.hdf5' in name]
    
    print('training_paths:', training_paths[:3], '...', len(training_paths))
    print('testing_paths', testing_paths[:3], '...', len(testing_paths))
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    model_config = None
    input_channels = 1
    output_channels = 1
    
    if identifier == 'cbct2ct':
        training_paths = [p for p in training_paths]
        validation_paths = [p for p in testing_paths]
        im_size = [256, 256]
        input_channels = 1
        sampling_config = {
            'fg_range': [0.2441, 0.2686], # HU from 0 to 100 for range of -1000 to 3096
            'threshold': 0.09766, # Threadhold of 400 HU
        }
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        
        # Identity sampling and mixup make the results worse
        model_config = {
            'epoch': 2000, 
            'batch_size': 8,
            'iters_per_epoch': 1280,
            'save_period': 100,
            'fg_sampling_ratio': 0.9,
            # 'add_identity_sample': True,
            # 'identity_sampling_ratio': 0.1,
            # 'augmentation_params': {'do_mixup': True, 'p_mixup': 0.5},
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'optimizer_type': 'Adam',
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': True
        }
        
    elif identifier == 'mri2ct':
        im_size = [256, 256]
        input_channels = 1
        sampling_config = {
            # intensity histogram based enhancement, focus on less frequent intensity
            'bin_weights': [2.97, 6.24, 0.29, 5.06, 7.15, 9.08, 12.82, 14.82, 15.47, 13.93]
        }
        
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        model_config = {
            'epoch': 5, 
            'batch_size': 8,
            'iters_per_epoch': 1280,
            'save_period': 100,
            'fg_sampling_ratio': 0.9,
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'optimizer_type': 'Adam',
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': True,
        }
        validation_paths = testing_paths
        



    ########## 20250630 diffusion
    elif identifier == 'DDPM':
        training_paths = [p for p in training_paths]
        validation_paths = [p for p in testing_paths]
        im_size = [256, 256]
        input_channels = 2    # x + c
        output_channels = 1



        model_config = {
            'epoch': 250, 
            'batch_size': 8,
            'iters_per_epoch': 1280,   # 每个epoch的迭代次数
            'save_period': 100,
            'optimizer_type': 'Adam',
            # 'initial_lr': 1e-4,
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': False,  # 改用time-embedding_resnet
            'temb_residual': True,
            'max_timesteps': 1000,
            'dropout': 0.1,
            # 添加扩散模型参数
            # 'is_diffusion': True,
            'beta_1': 0.0001,
            'beta_T': 0.02

        }




    elif identifier == 'thermaldenoising':
        im_size = [384, 384]
        input_channels = 1
        sampling_config = None
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        # mean: 0.0; std: 0.0~0.3
        bins = 100
        dist = [np.log(1 + 1/n) for n in range(1, bins + 1)]
        dist = np.array(dist) / np.sum(dist)
        random_std = lambda init, span: init + np.random.choice(range(bins), p=dist) / bins * span
        simulation_fn = lambda x: np.clip(x + np.random.normal(0.0, random_std(0.0, 0.3), x.shape), 0, 1)
        simulation_config = {'simulation_fn': simulation_fn}
        model_config = {
            'epoch': 100, 
            'batch_size': 8,
            'iters_per_epoch': 1280,
            'save_period': 10,
            'fg_sampling_ratio': 0.9,
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'simulation_config': simulation_config,
            'norm_config': {'norm': True, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None}
        }
        validation_paths = testing_paths[:100]
        
    elif identifier == 'ultrasound':
        im_size = [256, 256]
        input_channels = 1
        sampling_config = {
            'threshold': 0.1,
        }
        
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        model_config = {
            'epoch': 3000, 
            'batch_size': 8,
            'iters_per_epoch': 1280,
            'save_period': 100,
            'fg_sampling_ratio': 0.9,
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'optimizer_type': 'Adam',
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': True,
        }
        validation_paths = testing_paths
        
    elif identifier == 'ctenhance':
        im_size = [384, 384]
        input_channels = 1
        output_channels = 2
        
        sampling_config = None
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        model_config = {
            'epoch': 2000, 
            'batch_size': 4,
            'iters_per_epoch': 1280,
            'save_period': 100,
            'fg_sampling_ratio': 0.9,
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'optimizer_type': 'Adam',
            'loss_type': 'hybrid',
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': True,
            'activation': 'hybrid',
        }
        validation_paths = testing_paths
        
    elif identifier == 'mrdenoising':
        im_size = [256, 256]
        input_channels = 1
        output_channels = 1
        sampling_config = None
        mirror_config = {'training_mirror': True, 'testing_mirror': True,
                         'mirror_axes': [1, 2], 'rot90': False} # axis 0 is batch
        
        def add_rician_noise(image, boundary_mask, min_sigma=0.0, max_sigma=0.2):
            min_sigma, max_sigma = min(min_sigma, max_sigma), max(min_sigma, max_sigma)
            def random_sin():
                v_flip = np.random.uniform() > 0.5
                h_flip = np.random.uniform() > 0.5
                period = np.random.randint(1, 3)
                v_shift = 0
                h_shift = np.random.uniform(0, np.pi / 2)
                if period == 2:
                    h_shift = np.random.uniform(-np.pi / 2, np.pi / 2)
                start = h_shift
                end = period * np.pi / 2 + h_shift

                def func(x):
                    x = np.clip(x, 0.0, 1.0)
                    if h_flip:
                        x = 1 - x
                    y = np.clip(np.sin(start + x * (end - start)) + v_shift, 0, None)
                    if v_flip:
                        return 1 - y
                    else:
                        return y
                return func

            output_shape = image.shape
            funcs = [random_sin() for _ in output_shape]
            grid = np.ogrid[tuple(slice(0, 1, s * 1j) for s in output_shape)]
            for ax in range(len(output_shape)):
                ax_min, ax_max = get_nonzero_range(boundary_mask, axis=ax)
                ax_max += 1
                ax_min /= output_shape[ax]
                ax_max /= output_shape[ax]
                grid[ax] = (grid[ax] - ax_min) / (ax_max - ax_min)
            sigma_map = sum(funcs[o](grid[o]) for o in range(len(output_shape)))
            scale = sigma_map.max() - sigma_map.min()
            if scale == 0:
                sigma_map = min_sigma
                noise_real = np.random.normal(0, sigma_map, image.shape)
                noise_imag = np.random.normal(0, sigma_map, image.shape)
            else:
                sigma_map -= sigma_map.min()
                sigma_map /= scale
                sigma_map *= max_sigma - min_sigma
                sigma_map += min_sigma
                np.maximum(sigma_map, 0, out=sigma_map)
                noise_real, noise_imag = generate_noise(sigma_map, image.shape)
            noisy_image = np.sqrt((image + noise_real)**2 + noise_imag**2)
            return noisy_image

        def simulation_fn(x):
            min_sigma = np.clip(np.random.uniform(0.0, 0.04) - 0.02, 0, 0.02)
            max_sigma = np.random.uniform(min_sigma, 0.06 + min_sigma)
            fg_mask = x > np.percentile(x, 75)
            return add_rician_noise(x, fg_mask, min_sigma, max_sigma)
                
        model_config = {
            'epoch': 2000, 
            'batch_size': 8,
            'iters_per_epoch': 1280,
            'save_period': 100,
            'fg_sampling_ratio': 0.9,
            'sampling_config': sampling_config,
            'mirror_config': mirror_config,
            'optimizer_type': 'Adam',
            'simulation_config': {'simulation_fn': simulation_fn},
            'norm_config': {'norm': False, 'norm_channels': [0],
                            'norm_mean': None, 'norm_std': None},
            'residual': True,
        }
        
        validation_paths = testing_paths
    
    if phase == 0:
        is_train = True
    else:
        is_train = False
    
    if model_name == 'UNet2D':
        MainModel = ImageTransformUNet2D
    # elif model_name == 'CycleGAN':
    #    MainModel = CycleGAN
    elif model_name == 'DDPM':
        MainModel = DDPM_UNet2D
    else:
        raise Exception('Unsupported model name: {}'.format(model_name))
    model = MainModel(checkpoint_dir=checkpoint_dir, log_dir=log_dir, training_paths=training_paths, im_size=im_size, 
                      num_threads=num_threads, input_channels=input_channels, output_channels=output_channels, 
                      model_config=model_config, resume=resume)

    if is_train:
        if model_name == 'DDPM':
            model.diffusion_train()
        else:
          # model.train(run_validation=True, validation_paths=validation_paths)
          model.train(run_validation=False)
    else:
        if model_name == 'DDPM':
            model.diffusion_test(testing_paths=validation_paths, output_path=None, squeue=None)
        else:
            model.test(validation_paths, output_path)

    tf.keras.backend.clear_session()
    
if __name__ == '__main__':
    config = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER']= 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda)
    with multiprocessing.Pool(1, maxtasksperchild=1) as pool:
        pool.apply(main, [config,])
