import numpy as np
import nibabel as nib

def soft_tissue_window(img):
    img[np.where(img<100)] = 0
    img[np.where(img>400)] = 0
    return img

def isolate_spleen(label):
    label[np.where(label != 1)] = 0
    return label

def downsample(data, ds): #assumes square, downsample factor evenly divides image, and downsample factor is an int
    l = np.shape(data)[0]
    data = data[0:l-1:ds, 0:l-1:ds, :]
    return data

def upsample(data, ds):
    data = data.repeat(ds, axis=0).repeat(ds, axis=1)
    return data

def save(data, file_name, hdr):
    img = nib.Nifti1Image(data, np.eye(4), hdr) 
    nib.save(img, file_name)


