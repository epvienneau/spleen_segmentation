from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from img_loader_test import img_loader
import numpy as np
import math
import sys
import scipy.misc
import cv2
import matplotlib.pyplot as plt

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            output = F.log_softmax(model(data[0]), dim=1)
            output = output.max(1, keepdim=True)[1]
    return output

def main():
    model = models.resnet152()
    for params in model.parameters():
        params.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 512)
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Linear(model.fc.in_features, 256)
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Linear(model.fc.in_features, 128)
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Linear(model.fc.in_features, 64)
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Linear(model.fc.in_features, 32)
    model.dropout = nn.Dropout(0.5)
    model.fc = nn.Linear(model.fc.in_features, 16)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.double()
    model.load_state_dict(torch.load('Resnetmodel.pt'))
    model = model.cuda()

    img_path = 'data/test/'
    img_name = sys.argv[1]
    data_test = [img_path, img_name] 
    test_loader = torch.utils.data.DataLoader(img_loader(data_test))
    prediction = test(model, test_loader)
    lookup = {'0': 'melanoma', '1': 'melanocytic nevus', '2': 'basal cell carcinoma', '3': 'actinic keratosis', '4': 'benign keratosis', '5': 'dermatofibroma', '6': 'vascular lesion'}
    prediction = str(prediction.numpy()[0][0])
    print('Prediction:')
    print(lookup[prediction])
    #plt.imshow(img)
    #plt.show()


if __name__ == '__main__':
    main()
