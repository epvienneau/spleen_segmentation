import numpy as np
import test
import sys
import os
import nibabel as nib
import concat

def main():
    img_path = 'data/testing/slices/img/'
    #img_id = sys.argv[1]    
    for img_id in os.listdir('data/testing/img'):
        all_img_slices = [item for item in os.listdir(img_path) if img_id in item]
        for item in all_img_slices:
            test.main(item)
        concat.main(img_id)

if __name__ == '__main__':
    main()
