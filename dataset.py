import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import glob
import os

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

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, transforms_=None, unaligned=False):
        self.hdf5_path = hdf5_path
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        # Get file list and sort by name
        self.file_list = sorted(glob.glob(os.path.join(hdf5_path, '*.hdf5')))
        self.dataset_length = len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            with h5py.File(file_path, 'r') as f:
                image_A = np.asarray(f['input_images'][()], dtype=np.float32)
                image_B = np.asarray(f['output_images'][()], dtype=np.float32)
        except OSError as e:
            print(f"Error reading file {file_path}: {e}")
            return None

        image_A = crop_pad3D(image_A, target_size=(256, 256, 256), shift=[0, 0, 0])
        image_B = crop_pad3D(image_B, target_size=(256, 256, 256), shift=[0, 0, 0])

        # Convert to torch tensors and add channel dimension
        item_A = torch.from_numpy(image_A).unsqueeze(0)
        item_B = torch.from_numpy(image_B).unsqueeze(0)

        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return {"cbct": item_A, "ct": item_B}

    def __len__(self):
        return self.dataset_length



# Example usage:
# if __name__ == "__main__":
data_dir = r'D:\virtualC\CBCT2CTTest'

train_dataloader = DataLoader(
    HDF5Dataset(data_dir, transforms_=False, unaligned=True),
    batch_size=1,
    shuffle=False,  
    num_workers=0,
)

for i, batch in enumerate(train_dataloader):
    if batch is None:
        continue  # Skip batches that failed to load
    cbct = batch['cbct']
    ct = batch['ct']
    print(f"Batch {i}: CBCT shape: {cbct.shape}, CT shape: {ct.shape}")
    if i == 0:  # Just to check the first batch
        break