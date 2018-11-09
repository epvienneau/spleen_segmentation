import os
import csv
import numpy as np
import nibabel as nib
from utils import downsample, save, isolate_spleen

def main():
    img_folder = 'data/training/img/'
    label_folder = 'data/training/label/' 
    test_img_folder = 'data/testing/img/'
    img_slices_folder = 'data/training/slices/img/'
    label_slices_folder = 'data/training/slices/label/' 
    test_img_slices_folder = 'data/testing/slices/img/'
    spleen_slices = []
    
    for filename in os.listdir(img_folder):
        data = nib.load(img_folder+filename)
        img = data.get_fdata()
        hdr = data.header
        #img = downsample(img, 4)
        img_id = filename[0:-7]
        for s in range(np.shape(img)[2]):
            save(img[:,:,s], img_slices_folder+img_id+'_'+str(s)+'.nii.gz', hdr)

    for filename in os.listdir(test_img_folder):
        data = nib.load(test_img_folder+filename)
        img = data.get_fdata()
        hdr = data.header
        #img = downsample(img, 4)
        img_id = filename[0:-7]
        for s in range(np.shape(img)[2]):
            save(img[:,:,s], test_img_slices_folder+img_id+'_'+str(s)+'.nii.gz', hdr)

    for filename in os.listdir(label_folder):
        data = nib.load(label_folder+filename)
        label = data.get_fdata()
        hdr = data.header
        #label = downsample(label, 4)
        label = isolate_spleen(label)
        label_id = filename[0:-7]
        for s in range(np.shape(label)[2]):
            save(label[:,:,s], label_slices_folder+label_id+'_'+str(s)+'.nii.gz', hdr)
            is_spleen = int(label[:,:,s].any())
            if is_spleen: 
                spleen_slices.append(2.92) #weighted probability to ensure 66% chance of drawing spleen
            else:
                spleen_slices.append(0.43) #weighted probability to ensure 33% chance of drawing not spleen
    with open('spleen_probs.txt', 'w', newline='') as f:
        f.write(str(spleen_slices))

if __name__ == '__main__':
    main()
