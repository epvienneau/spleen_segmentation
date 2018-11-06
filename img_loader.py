import os
import numpy as np
import nibabel as nib
from utils import soft_tissue_window#, isolate_spleen
from torch.utils import data

class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list
        img_folder = subinfo[0][index] #data/training/slices/img
        label_folder = subinfo[1][index] #data/training/slices/label
        img_name = subinfo[2][index] #eg, img0001_0.nii.gz
        label_name = subinfo[3][index] #eg, label0001_0.nii.gz
        img_file = os.path.join(img_folder, img_name)
        label_file = os.path.join(label_folder, label_name)
        img = nib.load(img_file).get_fdata()
        #note that images have already been cropped to 128x128
        img = soft_tissue_window(img)
        label = nib.load(label_file).get_fdata()
        #label = isolate_spleen(label) #did this is slice_data 
        #img = np.concatenate((img, img, img), axis=0) #copy into three channels
        img = np.reshape(img, (128, 128, 1)) #now it's 128x128x1
        img = np.moveaxis(img, -1, 0) #now channel dimension is first
        #label = np.concatenate((label, label, label), axis=0) #copy into three channels
        label = np.reshape(label, (128, 128, 1)) #now it's 128x128x1
        label = np.moveaxis(label, -1, 0) #now channel dimension is first
        return [img, label]

    def __len__(self): 
        return len(self.sub_list[0]) 


