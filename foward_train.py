import tensorflow as tf
import os
from dataset_tf import create_dataset
from Diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from unet3d_zjy import UNet
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 1. 设置训练参数
BATCH_SIZE = 1
EPOCHS = 500
LEARNING_RATE = 1e-4
T = 1000  # diffusion steps
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
    ch=16,
    ch_mult=[1, 2,4],
    attn=[1],
    num_res_blocks=2,
    dropout=0.1,
    in_ch=2,  # 输入通道数为2（x和condition）
    out_c=1   # 输出通道数为1
)

# 修改优化器定义
initial_learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# 添加学习率调度器
lr_scheduler = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,        # 每次降低为原来的一半
    patience=10,       # 10个epoch损失没有改善就降低学习率
    min_lr=1e-6,      # 最小学习率
    verbose=1         # 打印学习率变化信息
)

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







def load_model_with_weights(checkpoint_path):
    # 1. 创建模型实例
    # model = UNet(
    #     T=1000,
    #     ch=16,
    #     ch_mult=[1, 2, 4],
    #     attn=[1],
    #     num_res_blocks=2,
    #     dropout=0.1,
    #     in_ch=2,
    #     out_c=1
    # )
    
    # 2. 初始化模型变量 - 关键步骤
    dummy_x = tf.random.normal((1, 128, 128, 128, 1))
    dummy_t = tf.ones((1,), dtype=tf.int32)
    dummy_c = tf.random.normal((1, 128, 128, 128, 1))
    
    # 3. 进行一次前向传播以创建变量
    _ = model(dummy_x, dummy_t, dummy_c)
    
    # 4. 现在可以安全地加载权重
    model.load_weights(checkpoint_path)
    
    return model
# 8. 主训练循环
def train(resume_from=None):
  """
    训练函数
    Args:
        resume_from: 可选，要恢复训练的检查点路径
  """
  # loss 记录
  train_loss_metric = tf.keras.metrics.Mean()
  total_steps = sum(1 for _ in train_dataset)

  # 如果指定了检查点，加载权重
  start_epoch = 0
  if resume_from:
      print(f"load checkpoints: {resume_from}")
      model=load_model_with_weights(resume_from)
      # 从文件名提取epoch数
      try:
          start_epoch = int(resume_from.split('epoch_')[-1].split('.')[0])
          print(f"从 epoch {start_epoch} 继续训练")
      except:
          print("无法从文件名获取epoch数，从0开始")

  for epoch in range(start_epoch, EPOCHS):
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
              loss = trainer.forward(x_0 = ct, context = cbct)
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
    
      # 更新学习率
      lr_scheduler.on_epoch_end(epoch, logs={'loss': epoch_loss.numpy()})
      current_lr = optimizer.learning_rate.numpy()
      print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")


      # save checkpoints
      if (epoch + 1) % 100 == 0:
          # 保存模型
          model.save(f"{checkpoint_dir}/model_epoch_{epoch+1}.h5")  # 保存整个model
          # 生成样本
          # save_samples(epoch, cbct,save_path = f"{sample_dir}/sample_epoch_{epoch+1}.png")  # 使用最后一个batch的CBCT作为条件
          
      # 可选：保存训练历史
      # 如果是第一个epoch，先清空文件
      if epoch == 0:
          with open(f"{checkpoint_dir}/training_history.txt", "w") as f:
            f.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")
      else:
          with open(f"{checkpoint_dir}/training_history.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")

  print("Training complete.")



if __name__ == "__main__":
    # 配置内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # 设置内存优化器选项
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
    })
    
    train(resume_from='/home/jiayizhang/project/diffusion/DDPM/zjy/checkpoints/model_epoch_200.h5')
