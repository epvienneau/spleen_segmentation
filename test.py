from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from img_loader_test import img_loader
from unet import UNet
import sys
import os
import glob
import numpy as np
from utils import save, upsample
import nibabel as nib

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            output = output.squeeze()
            output_probs = torch.sigmoid(output) 
            output_mask = (output_probs > 0.5).float()
    return output_mask

def main(img_id):
    model = UNet(n_channels=1, n_classes=1)
    #model_file = max(glob.glob('models/*'), key=os.path.getctime) #detect latest version
    model_file = 'models/intensity_filtering_continued/Checkpoint_e1_d0.9755_l0.0008_2018-11-13_11:06:28.pth' #best one!!
    model.load_state_dict(torch.load(model_file))
    model = model.double()
    img_path = 'data/testing/slices/img/'
    img_vol_path = 'data/testing/img/' #this is for getting an accurate header
    data_test = [img_path, img_id] 
    test_loader = torch.utils.data.DataLoader(img_loader(data_test))
    hdr = nib.load(img_path+img_id).header
    vol_hdr = nib.load(img_vol_path+img_id[0:7]+'.nii.gz').header
    hdr['pixdim'] = vol_hdr['pixdim'] #explicitly set this to force it to keep the correct pixel dimensions
    prediction = test(model, test_loader).numpy()
    prediction = np.reshape(prediction, (256, 256))
    prediction = upsample(prediction, 2)
    save(prediction, 'data/testing/slices/pred/'+img_id, hdr)

if __name__ == '__main__':
    main(sys.argv[1])
