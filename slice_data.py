import os 
import numpy as np
import nibabel as nib
from utils import crop, save

def main():
    img_folder = 'data/testing/img/'
    label_folder = 'data/training/label/' #comment out to do this on training data
    img_slices_folder = 'data/testing/slices/img/'
    label_slices_folder = 'data/training/slices/label/' #comment out to do this on training data
    
    for filename in os.listdir(img_folder):
        img = nib.load(img_folder+filename).get_fdata()
        img = crop(img, 4)
        img_id = filename[0:-7]
        for s in range(np.shape(img)[2]):
            save(img[:,:,s], img_slices_folder+img_id+'_'+str(s)+'.nii.gz')

    #comment out to do this on training data
    for filename in os.listdir(label_folder):
        label = nib.load(label_folder+filename).get_fdata()
        label = crop(label)
        label_id = filename[0:-7]
        for s in range(np.shape(label)[2]):
            save(label[:,:,s], label_slices_folder+label_id+'_'+str(s)+'.nii.gz')

if __name__ == '__main__':
    main()
