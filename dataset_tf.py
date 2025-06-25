import h5py
import numpy as np
import tensorflow as tf
import os

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

def create_tf_dataset(data_dir, batch_size=32, shuffle=False):
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    img_a = []
    img_b = []
    for file_path in files:
        try:
          input_images, output_images = load_data_hdf5(file_path)
        except OSError:     # 4 和 5 无法读取
            print(file_path)
            continue
        input_images = crop_pad3D(input_images, [256,256,256])
        output_images = crop_pad3D(output_images,[256,256,256])
        input_images = np.expand_dims(input_images, axis=-1)  # 添加channel维度
        output_images = np.expand_dims(output_images, axis=-1)
        img_a.append(input_images)
        img_b.append(output_images)
    # 合并为 NumPy
    img_a = np.stack(img_a, axis=0)
    img_b = np.stack(img_b, axis=0)

    dataset = tf.data.Dataset.from_tensor_slices((img_a,img_b))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_images))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
  file_path = '/home/jiayizhang/project/diffusion/DDPM/CBCT2CTTest'
  batch_size = 1

  dataset = create_tf_dataset(file_path, batch_size)

  # 迭代数据集
  for input_batch, output_batch in dataset:
      print(f'Input batch shape: {input_batch.shape}, Output batch shape: {output_batch.shape}')  # (1, 256, 256, 256, 1) 
