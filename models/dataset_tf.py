import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import h5py
import numpy as np
import tensorflow as tf

## crop
def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)
    start_pos = start_pos.astype(int)
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    return y



def load_data_hdf5(file_path):
    with h5py.File(file_path, 'r') as f_h5:
        input_images = np.asarray(f_h5['input_images'], dtype=np.float32)
        output_images = np.asarray(f_h5['output_images'], dtype=np.float32)
    return input_images, output_images

def create_dataset(data_file, batch_size=32, shuffle=True, return_filenames=False):
    """
    创建TensorFlow数据集
    
    Args:
        data_file: 数据文件路径（字符串）或文件路径列表
        batch_size: 批处理大小
        shuffle: 是否打乱数据
        return_filenames: 是否返回文件名信息
    
    Returns:
        如果return_filenames=False: 返回dataset
        如果return_filenames=True: 返回(dataset, filenames)
    """
    if isinstance(data_file, str):   # str路径
        if data_file.endswith('.hdf5'):     # 单个 .hdf5 文件路径
          files = [data_file]
        elif os.path.isdir(data_file):  # 目录路径
          files = sorted([os.path.join(data_file, f) for f in os.listdir(data_file) if f.endswith('.hdf5')])
    elif isinstance(data_file, (list, tuple)): # list路径
      files = [f for f in data_file if f.endswith('.hdf5')]
    else:
      raise TypeError("data_file must be a string path or list of paths")
    
    # print(f"Found {len(files)} .hdf5 files")
    if len(files) == 0:
        raise ValueError("No .hdf5 files found in the specified path(s)")
    
    img_a = []
    img_b = []
    filenames = []  # 存储对应的文件名
    successful_files = 0
    
    for file_path in files:
        try:
          # print(f"Loading {file_path}...")
          input_images, output_images = load_data_hdf5(file_path)
          # print(f"Loaded data shapes - Input: {input_images.shape}, Output: {output_images.shape}")
          
          input_images = crop_pad3D(input_images, [256,256,256])
          output_images = crop_pad3D(output_images,[256,256,256])
          input_images = np.expand_dims(input_images, axis=-1)  # 添加channel维度
          output_images = np.expand_dims(output_images, axis=-1)
          
          img_a.append(input_images)
          img_b.append(output_images)
          # 添加文件名（不含路径和.hdf5后缀）
          filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
          filenames.append(filename_without_ext)
          successful_files += 1
        except Exception as e:     # 捕获所有异常
            print(f"Failed to load {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {successful_files} out of {len(files)} files")
    
    # 检查是否有有效数据
    if len(img_a) == 0 or len(img_b) == 0:
        raise ValueError("No valid data loaded. Please check your data files and paths.")
    
    # 合并为 NumPy
    img_a = np.stack(img_a, axis=0)
    img_b = np.stack(img_b, axis=0)
    
    # print(f"Final data shapes - Input: {img_a.shape}, Output: {img_b.shape}")
    # print(f"Corresponding filenames: {filenames}")

    if return_filenames:
        # 包含文件名的数据集
        dataset = tf.data.Dataset.from_tensor_slices((img_a, img_b, filenames))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_a))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    else:
        # 原有逻辑：只返回数据
        dataset = tf.data.Dataset.from_tensor_slices((img_a, img_b))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_a))  # 修正这里，应该用 img_a 的长度

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset




######## Example usage
if __name__ == "__main__":
  # file_path = '/home/jiayizhang/project/diffusion/DDPM/CBCT2CTTest'
  batch_size = 1

  identifier = '/mnt/newdisk/mri2ct'
  train_data_dir = identifier + '/data_training/test'
  training_paths = [os.path.join(train_data_dir, name) for name in os.listdir(train_data_dir) if name.endswith('.hdf5')]
  
  print("=== 示例1: 不返回文件名 ===")
  dataset = create_dataset(training_paths, batch_size)
  
  # 迭代数据集
  for i, (input_batch, output_batch) in enumerate(dataset.take(2)):
      print(f'Batch {i}: Input shape: {input_batch.shape}, Output shape: {output_batch.shape}')
      input_batch = input_batch[:,64,...]
      output_batch = output_batch[:,64,...]
      print(f'After slicing: Input shape: {input_batch.shape}, Output shape: {output_batch.shape}')
      
      


  # print("\n=== 示例2: 返回文件名 ===")
  # dataset_with_names = create_dataset(training_paths, batch_size, return_filenames=True)
  
  # # 迭代数据集（包含文件名）
  # for i, (input_batch, output_batch, filename_batch) in enumerate(dataset_with_names.take(2)):
  #     print(f'\nBatch {i}:')
  #     print(f'  Input shape: {input_batch.shape}, Output shape: {output_batch.shape}')
  #     print(f'  Filenames in batch: {[f.decode("utf-8") for f in filename_batch.numpy()]}')
  #     input_batch = input_batch[:,64,...]
  #     output_batch = output_batch[:,64,...]
  #     print(f'  After slicing: Input shape: {input_batch.shape}, Output shape: {output_batch.shape}')

