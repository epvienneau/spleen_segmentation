import os 
import numpy as np
import nibabel as nib

def main():
    img_folder = 'data/training/img/'
    label_folder = 'data/training/label/'
    cat_folder = 'data/training/cat'

    for filename in os.listdir(img_folder):
        img = nib.load(img_folder+filename).get_fdata()
        img = crop(img)
    save(concat_img, cat_folder+'img_concatenated.nii.gz')

    for filename in os.listdir(label_folder):
        label = nib.load(label_folder+filename).get_fdata()
        label = crop(label)
    save(concat_label, cat_folder+'label_concatenated.nii.gz')

def crop(data):
    data = data[0:511:4, 0:511:4, :]
    return data

def save(data, name):
    img = nib.Nifti1Image(data, np.eye(4)) #provide identity matrix as affine transformation
    img.to_filename(name) 

if __name__ == '__main__':
    main()
