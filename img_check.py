import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models'))
from models.dataset_tf import create_dataset
import matplotlib.pyplot as plt
import h5py
import numpy as np

testing_paths = []
test_dir = '/home/jiayizhang/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/data_training/test'
for file in os.listdir(test_dir):
  if file.endswith('.hdf5'):
    testing_paths.append(os.path.join(test_dir, file))



output_path = '/home/jiayizhang/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/output/test'

for i, input_file in enumerate(testing_paths):
    with h5py.File(input_file, 'r') as f_h5:
        filename = os.path.splitext(os.path.basename(input_file))[0]   # 文件名
        save_path = os.path.join(output_path, filename)

        if 'output_images' in f_h5.keys():
            ct = np.asarray(f_h5['output_images'], dtype=np.float32)
            ct = ct[ct.shape[0]//2,...]
            
            plt.figure(figsize=(6, 6))
            plt.imshow(ct, cmap='gray')
            plt.title('real x0')
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'real_x0.png'))
            plt.close()
        if 'input_images' in f_h5.keys():
            context = np.asarray(f_h5['input_images'], dtype=np.float32)
            context = context[context.shape[0]//2,...]
            plt.figure(figsize=(6, 6))
            plt.imshow(context, cmap='gray')
            plt.title('context')
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f'context.png'))
            plt.close()