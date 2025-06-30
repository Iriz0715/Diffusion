import tensorflow as tf
import os
from dataset_tf import create_dataset
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from unet3d_zjy import UNet
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. 设置参数
BATCH_SIZE = 1
EPOCHS = 200
# LEARNING_RATE = 1e-4
T = 1000  # diffusion steps
infer_T = 1000 # inference steps, 可以小于 T
BETA_1 = 0.0001
BETA_T = 0.02
SUQEUE =500  # 每100个t保存一次中间结果

# 2. 创建保存目录
checkpoint_dir = "/home/jiayizhang/project/diffusion/DDPM/zjy/checkpoints"
sample_dir = "/home/jiayizhang/project/diffusion/DDPM/zjy/samples"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)


import tensorflow as tf
from unet3d_zjy import UNet

def load_model_with_weights(checkpoint_path):
    # 1. 创建模型实例
    model = UNet(
        T=1000,
        ch=16,
        ch_mult=[1, 2, 4],
        attn=[1],
        num_res_blocks=2,
        dropout=0.1,
        in_ch=2,
        out_c=1
    )
    
    # 2. 初始化模型变量 - 关键步骤
    dummy_x = tf.random.normal((1, 128, 128, 128, 1))
    dummy_t = tf.ones((1,), dtype=tf.int32)
    dummy_c = tf.random.normal((1, 128, 128, 128, 1))
    
    # 3. 进行一次前向传播以创建变量
    _ = model(dummy_x, dummy_t, dummy_c)
    
    # 4. 现在可以安全地加载权重
    model.load_weights(checkpoint_path)
    
    return model

# model = UNet(
#     T=T, 
#     ch=16,
#     ch_mult=[1, 2,4],
#     attn=[1],
#     num_res_blocks=2,
#     dropout=0.1,
#     in_ch=2,  # 输入通道数为2（x和condition）
#     out_c=1   # 输出通道数为1
# )


# model.load_weights(f"{checkpoint_dir}/model_epoch_{EPOCHS}.h5")


# 验证权重是否正确加载
model = load_model_with_weights(f"{checkpoint_dir}/model_epoch_{EPOCHS}.h5")

sampler = GaussianDiffusionSampler(
    model=model,
    beta_1=BETA_1,
    beta_T=BETA_T,
    T=T,
    infer_T=infer_T,   # infer_T 可以 < train_T
    squeue=SUQEUE     # 保存中间结果
)


# 5. 创建数据集
train_dataset = create_dataset(
    '/home/jiayizhang/project/diffusion/DDPM/CBCT2CTTest',
    batch_size=BATCH_SIZE,
    shuffle=True
)



def save_slice_views(img_3d, step_info, save_path):
    """保存3D图像的三个正交面视图
    Args:
        img_3d: 3D图像数据 [H, W, D]
        step_info: 时间步信息，用于标题
        save_path: 保存路径
    """
    center_x, center_y, center_z = [s // 2 for s in img_3d.shape]
    
    plt.figure(figsize=(15, 5))
    views = [
        (img_3d[center_x, :, :], 'Sagittal'),
        (img_3d[:, center_y, :], 'Coronal'),
        (img_3d[:, :, center_z], 'Axial')
    ]
    
    for idx, (slice_data, view_name) in enumerate(views, 1):
        plt.subplot(1, 3, idx)
        plt.imshow(slice_data, cmap='gray')
        plt.title(f'{view_name} ({step_info})')
        plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()



def save_samples(condition, squeue=None, save_path=None):
    """生成并保存样本
    Args:
        condition: 条件输入(CBCT图像)
        squeue: 保存中间结果的间隔步数
        save_dir: 保存目录
    """
    # 生成样本
    x_T = tf.random.normal((1, 128, 128, 128, 1))
    generated = sampler.reverse(x_T, context=condition)
    print(f"Generated shape: {generated.shape}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存结果
    if squeue is not None:
        # 保存中间结果
        for step in range(generated.shape[-1]):
            save_slice_views(
                generated[0, ..., step],
                f't={step*squeue}',
                os.path.join(save_path, f'step_{step*squeue:04d}.png')
            )
    else:
        # 只保存最终结果
        save_slice_views(
            generated[0, ..., 0],
            'final',
            os.path.join(save_path, 'final.png')
        )




def sample():
  # Load the trained model weights
  i = 0
  for cbct, _ in train_dataset:
      i += 1
      current_sample_dir = os.path.join(sample_dir, f'sample_{i}')
      save_samples(condition=cbct, squeue=SUQEUE, save_path=current_sample_dir)
      print(f"Sample {i} saved.")





if __name__ == "__main__":
    # 设置内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    sample()
