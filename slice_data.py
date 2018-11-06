import os
import csv
import numpy as np
import nibabel as nib
from utils import crop, save, isolate_spleen

def main():
    #img_folder = 'data/testing/img/'
    label_folder = 'data/training/label/' #comment out to do this on testing data
    #img_slices_folder = 'data/testing/slices/img/'
    label_slices_folder = 'data/training/slices/label/' #comment out to do this on testing data
    spleen_slices = []
    
    #for filename in  os.listdir(img_folder):
    #    img = nib.load(img_folder+filename).get_fdata()
    #    img = crop(img, 4)
    #    img_id = filename[0:-7]
    #    for s in range(np.shape(img)[2]):
    #        save(img[:,:,s], img_slices_folder+img_id+'_'+str(s)+'.nii.gz')

    #comment out to do this on testing data
    for filename in os.listdir(label_folder):
        label = nib.load(label_folder+filename).get_fdata()
        label = crop(label, 4)
        label = isolate_spleen(label)
        label_id = filename[0:-7]
        for s in range(np.shape(label)[2]):
            save(label[:,:,s], label_slices_folder+label_id+'_'+str(s)+'.nii.gz')
            is_spleen = int(label[:,:,s].any())
            if is_spleen: 
                spleen_slices.append(2.92) #weighted probability to ensure 66% chance of drawing spleen
            else:
                spleen_slices.append(0.43) #weighted probability to ensure 33% chance of drawing not spleen
    with open('spleen_probs.txt', 'w', newline='') as f:
        f.write(str(spleen_slices))

if __name__ == '__main__':
    main()
