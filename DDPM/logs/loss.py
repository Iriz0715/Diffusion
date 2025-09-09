import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

dir = 'DDPM/logs'
mat = sio.loadmat(dir + '/ddpm_unet2d_all.mat')
tr_ls = mat['loss'].flatten()
print("loss长度:", len(tr_ls))

start_epoch =10 
plt.figure(figsize=(8,5))
plt.plot(np.arange(start_epoch+1, len(tr_ls)+1), tr_ls[start_epoch:], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve (from epoch {})'.format(start_epoch))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(dir + '/train_loss_curve_brain2d_all.png')
plt.show()
