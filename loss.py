import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 读取 loss 文件
mat = sio.loadmat('/mnt/newdisk/diffusion/code_zjy/project/diffusion/DDPM/zjy/DDPM/Loss/ddpm_unet.mat')
tr_ls = mat['loss'].flatten()  # 保证是一维
print("loss长度:", len(tr_ls))

start_epoch = 10  # 从第10个epoch开始显示
plt.figure(figsize=(8,4))
plt.plot(np.arange(start_epoch+1, len(tr_ls)+1), tr_ls[start_epoch:], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve (from epoch 10)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/mnt/newdisk/diffusion/code_zjy/project/diffusion/DDPM/zjy/train_loss_curve.png')
plt.show()


