import numpy as np
import nibabel as nib

def soft_tissue_window(img):
    img[np.where(img<100)] = 0
    img[np.where(img>400)] = 0
    return img

def isolate_spleen(label):
    label[np.where(label != 10)] = 0
    return label

def crop(data, ds): #assumes square, downsample factor evenly divides image, and downsample factor is an int
    l = np.shape(data)[0]
    data = data[0:l-1:ds, 0:l-1:ds, :]
    return data

def save(data, name):
    img = nib.Nifti1Image(data, np.eye(4), nib.Nifti1Header()) #provide identity matrix as affine transformation
    nib.save(img, name)


