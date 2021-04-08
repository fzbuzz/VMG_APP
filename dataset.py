from torch.utils.data import Dataset
from skimage import io, transform
import os
import torch
import h5py
import numpy as np

class CelebA50k(Dataset):
    """ Dataset of image, surface normal, & mask from CelebA """
    def __init__(self, root_dir, transform=None):
        self.handle = h5py.File(root_dir,mode='r')
        self.length = self.handle['data'].shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # data = self.handle['data'][idx,:,:,:]
        # img, mask, shading = data[:,:,0:3], data[:,:,6:7], data[:,:,3:6]

        if self.transform:
            img = self.transform(img)
        # return img, mask, shading
        return self.handle['data'][idx,:,:,:].astype(np.float32)



