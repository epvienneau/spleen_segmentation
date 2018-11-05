from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from img_loader_test import img_loader
from unet import UNet
import sys
from utils import save

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            output = F.sigmoid(model(data))
    return output

def main(img_id):
    model = UNet(n_channels=1, n_classes=1)
    model = model.cuda()
    img_path = 'data/testing/slices/img/'
    data_test = [img_path, img_id] 
    test_loader = torch.utils.data.DataLoader(img_loader(data_test))
    prediction = test(model, test_loader)
    save(prediction, 'data/testing/slices/pred/'+img_id)

if __name__ == '__main__':
    main(sys.argv[1])
