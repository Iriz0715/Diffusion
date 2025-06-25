import tensorflow as tf
import os
from dataset_tf import create_dataset
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from unet3d_zjy import UNet
import matplotlib.pyplot as plt

# 1. 设置训练参数
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-4
T = 100  # diffusion steps
BETA_1 = 0.0001
BETA_T = 0.02

# 2. 创建保存目录
checkpoint_dir = "/home/jiayizhang/project/diffusion/DDPM/zjy/checkpoints"
sample_dir = "/home/jiayizhang/project/diffusion/DDPM/zjy/samples"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

# 3. 创建模型和优化器
model = UNet(
    T=T, 
    ch=8,
    ch_mult=[1, 2],
    attn=[],
    num_res_blocks=2,
    dropout=0.1,
    in_ch=2,  # 输入通道数为2（x和condition）
    out_c=1   # 输出通道数为1
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# 4. 创建 Trainer 和 Sampler
trainer = GaussianDiffusionTrainer(
    model=model,
    beta_1=BETA_1,
    beta_T=BETA_T,
    T=T
)

sampler = GaussianDiffusionSampler(
    model=model,
    beta_1=BETA_1,
    beta_T=BETA_T,
    T=T
)


# 5. 创建数据集
train_dataset = create_dataset(
    '/home/jiayizhang/project/diffusion/DDPM/CBCT2CTTest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 7. 保存中间生成结果的函数
def save_samples(epoch, condition, save_path=None):
    x_T = tf.random.normal((1, 128, 128, 128, 1))  # 随机噪声作为起始点
    generated = sampler.reverse(
        x_T = x_T,
        context=condition
    )
    
    # 获取中心切片索引
    center_x = generated.shape[1] // 2
    center_y = generated.shape[2] // 2
    center_z = generated.shape[3] // 2
    
    # 创建图像网格
    plt.figure(figsize=(15, 5))
    
    # 显示三个正交面的中心切片
    plt.subplot(131)
    plt.imshow(generated[0, center_x, :, :, 0], cmap='gray')
    plt.title(f'Sagittal (epoch {epoch})')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(generated[0, :, center_y, :, 0], cmap='gray')
    plt.title(f'Coronal (epoch {epoch})')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(generated[0, :, :, center_z, 0], cmap='gray')
    plt.title(f'Axial (epoch {epoch})')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    # 打印调试信息
    print(f"Generated shape: {generated.shape}")
    print(f"Center slices at: {center_x}, {center_y}, {center_z}")

# 8. 主训练循环
def train():
  # loss 记录
  train_loss_metric = tf.keras.metrics.Mean()
  total_steps = sum(1 for _ in train_dataset)
  for epoch in range(EPOCHS):
      print(f"Epoch {epoch + 1}/{EPOCHS}")
      
      # 在每个epoch开始时重置度量
      train_loss_metric.reset_states()
      
      # 使用进度条显示训练过程
      progress_bar = tf.keras.utils.Progbar(
          target=total_steps, 
          stateful_metrics=['loss']
      )
      
      for step, (cbct, ct) in enumerate(train_dataset):
          with tf.GradientTape() as tape:
              # 计算当前batch的损失
              loss = trainer.forward(ct, context=cbct)
              loss = tf.reduce_mean(loss)
          
          # 梯度更新
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          
          train_loss_metric.update_state(loss)
          
          # 更新进度条
          progress_bar.update(
              step + 1,
              values=[('loss', train_loss_metric.result())]
          )
      
      epoch_loss = train_loss_metric.result()
      print(f"\nEpoch {epoch + 1} mean loss: {epoch_loss:.4f}")
      
      # save checkpoints
      if (epoch + 1) % 5 == 0:
          # 保存模型
          model.save_weights(f"{checkpoint_dir}/model_epoch_{epoch+1}")
          # 生成样本
          # save_samples(epoch, cbct,save_path = f"{sample_dir}/sample_epoch_{epoch+1}.png")  # 使用最后一个batch的CBCT作为条件
          
      # 可选：保存训练历史
      with open(f"{checkpoint_dir}/training_history.txt", "a") as f:
          f.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")   

  print("Training complete.")



if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    train()
