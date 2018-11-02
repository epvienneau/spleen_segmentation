import os 
import numpy as np
import nibabel as nib

def main():
    img_folder = 'data/training/img/'
    label_folder = 'data/training/label/'
    img_slices_folder = 'data/training/slices/img/'
    label_slices_folder = 'data/training/slices/label/'
    
    for filename in os.listdir(img_folder):
        img = nib.load(img_folder+filename).get_fdata()
        img = crop(img)
        img_id = filename[0:-7]
        for s in range(np.shape(img)[2]):
            save(img[:,:,s], img_slices_folder+img_id+'_'+str(s)+'.nii.gz')

    for filename in os.listdir(label_folder):
        label = nib.load(label_folder+filename).get_fdata()
        label = crop(label)
        label_id = filename[0:-7]
        for s in range(np.shape(label)[2]):
            save(label[:,:,s], label_slices_folder+label_id+'_'+str(s)+'.nii.gz')

def crop(data):
    data = data[0:511:4, 0:511:4, :]
    return data

def save(data, name):
    img = nib.Nifti1Image(data, np.eye(4)) #provide identity matrix as affine transformation
    nib.save(img, name) 

if __name__ == '__main__':
    main()
