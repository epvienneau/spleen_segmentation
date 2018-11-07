import numpy as np
import nibabel as nib
from skimage.transform import resize

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
    data = scipy.misc.imresize(data, float(ds))

def save(data, file_name, hdr):
    img = nib.Nifti1Image(data, np.eye(4), hdr) #provide identity matrix as affine transformation
    nib.save(img, file_name)


