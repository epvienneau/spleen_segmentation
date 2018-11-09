import os 
import sys
import numpy as np
import nibabel as nib
from utils import save

def main(img_id):
    img_folder = 'data/testing/slices/pred/'
    out_folder = 'data/testing/results/'
    img_slices = [item for item in os.listdir(img_folder) if img_id[0:-7] in item]
    num_slices = len(img_slices)
    vol_img = np.zeros((256, 256, num_slices))
    for index, item in zip(range(num_slices), img_slices):
        data = nib.load(img_folder+item)
        img = data.get_fdata()
        hdr = data.header
        vol_img[:,:,index] = img
    save(vol_img, out_folder+img_id, hdr)

if __name__ == '__main__':
    main(sys.argv[1])
