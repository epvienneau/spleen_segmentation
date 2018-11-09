import os
import numpy as np
import nibabel as nib
from utils import soft_tissue_window, downsample
from torch.utils import data

class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list
        img_folder = subinfo[0]
        img_name = subinfo[1]
        img_file = os.path.join(img_folder, img_name)
        img = nib.load(img_file).get_fdata()
        #img = soft_tissue_window(img)
        img = downsample(img, 2)
        img = np.reshape(img, (256, 256, 1))
        img = np.moveaxis(img, -1, 0)
        return img

    def __len__(self): 
        return len(self.sub_list[0]) 


