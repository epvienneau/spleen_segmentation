import os 
import sys
import numpy as np
import nibabel as nib
from utils import save
import csv

def main(img_id):
    img_folder = 'data/testing/slices/pred/'
    vol_folder = 'data/testing/img/' #for extracting correct header
    out_folder = 'data/testing/results/'
    all_img_slices = []
    with open('image_slice_paths.csv', 'r') as csv:
        for line in enumerate(csv):
            path = line[1]
            all_img_slices.append(path[68:-1])
    img_slices = [item for item in all_img_slices if img_id[0:-7] in item]
    num_slices = len(img_slices)
    vol_img = np.zeros((512, 512, num_slices))
    for index, item in zip(range(num_slices), img_slices):
        data = nib.load(img_folder+item)
        img = data.get_fdata()
        hdr = data.header
        vol_hdr = nib.load(vol_folder + item[0:7] + '.nii.gz').header
        hdr['pixdim'] = vol_hdr['pixdim'] #explicitly set this to force it to keep correct pixel dims
        vol_img[:,:,index] = img
    save(vol_img, out_folder+img_id, hdr)

if __name__ == '__main__':
    main(sys.argv[1])
