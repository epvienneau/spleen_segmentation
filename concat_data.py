import os 
import numpy as np
import nibabel as nib

def main(img_id):
    img_folder = 'data/testing/slices/pred/ '
    cat_folder = 'data/testing/results/'

    concat_img = np.zeros((128, 128, 3779))

    index = 0
    num_slices = 0
    for filename in os.listdir(img_folder):
        img = nib.load(img_folder+filename).get_fdata()
        img = crop(img)
        num_slices = np.shape(img)[2]
        concat_img[:,:,index:index+num_slices] = img
        index = index + num_slices
    save(concat_img, cat_folder+'img_concatenated.nii.gz')

if __name__ == '__main__':
    main()
