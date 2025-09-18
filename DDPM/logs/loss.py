import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

dir = 'DDPM/logs'
mat = sio.loadmat(dir + '/loss.mat')
tr_ls = mat['loss'].flatten()

start_epoch =10 
plt.figure(figsize=(8,5))
plt.plot(np.arange(start_epoch+1, len(tr_ls)+1), tr_ls[start_epoch:], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve (from epoch {})'.format(start_epoch))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(dir + '/loss.png')
plt.show()
