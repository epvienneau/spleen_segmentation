import os
import numpy as np
#from torch.utils import data
import nibabel as nib

def soft_tissue_window(img):
    #img = img.astype('float')
    img[np.where(img<100)] = 0
    img[np.where(img>400)] = 0
    return img

def isolate_spleen(label):
    label[np.where(label != 1)] = 0
    return label

class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list
        img_folder = subinfo[0][index] #data/training/img
        label_folder = subinfo[1][index] #data/training/label
        img_name = subinfo[2][index] #eg, img0001.nii.gz
        label_name = subinfo[3][index] #eg, label0001.nii.gz
        img_file = os.path.join(img_folder, img_name)
        label_file = os.path.join(label_folder, label_name)
        img = nib.load(img_file).get_fdata()
        img = soft_tissue_window(img)
        label = nib.load(label_file).get_fdata()
        label = isolate_spleen(label)
        #img = img.astype('float')
        #probs = probs.astype('long')
        return [img, label]

    def __len__(self): 
        return len(self.sub_list[0]) 


