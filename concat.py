import os 
import sys
import numpy as np
import nibabel as nib
from utils import save

def main(img_id):
    img_folder = 'data/testing/slices/pred/'
    out_folder = 'data/testing/results/'
    img_slices = [item for item in os.listdir(img_folder) if img_id in item]
    num_slices = len(img_slices)
    vol_img = np.zeros((128, 128, num_slices))
    for index, item in zip(range(num_slices), img_slices):
        img = nib.load(img_folder+filename).get_fdata()
        vol_img[:,:,index] = img
    save(vol_img, out_folder+img_id)

if __name__ == '__main__':
    main(sys.argv[1])
