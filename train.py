from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from img_loader import img_loader
from dice_loss import dice_coeff
from unet import UNet
import math
import numpy as np
import csv
import sklearn.metrics as metrics
import datetime
import time
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

start_time = time.time()
training_loss = []
test_loss = []
dice_loss = []

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = F.sigmoid(model(data))
        criterion = nn.BCELoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    training_loss.append(loss.item())

def test(args, model, device, test_loader):
    model.eval()
    loss = 0
    avg_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.sigmoid(model(data))
            criterion = nn.BCELoss()
            loss += criterion(output, target).item() 
            avg_loss += loss
    avg_loss /= len(test_loader.dataset)
    test_loss.append(avg_loss)
    
    print('\nTest set statistics:') 
    print('Average loss: {:.4f}'.format(avg_loss)) 
    dice = dice_coeff(output, target)
    dice_loss.append(dice)
    print('Dice:')
    print(float(dice))
   #print('Dice: {:.4f}%'.format(dice))

def main():
    parser = argparse.ArgumentParser(description='PyTorch Mutliclass Classification')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam (default: 0.999)')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam (default: 1e-8)')
    #parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
 
    training_img_folder = ['data/training/slices/img']*3779
    training_label_folder  = ['data/training/slices/label']*3779
    training_img_files = os.listdir('data/training/slices/img')
    training_label_files = os.listdir('data/training/slices/label')
    
    indices = list(range(3779))
    testing_indices = np.random.choice(indices, size=945, replace=False)
    training_indices = list(set(indices)-set(testing_indices))
    training_sampler = SubsetRandomSampler(training_indices)
    testing_sampler = SubsetRandomSampler(testing_indices)

    training_data = [training_img_folder, training_label_folder, training_img_files, training_label_files]
    train_loader = torch.utils.data.DataLoader(img_loader(training_data), batch_size=args.batch_size, sampler=training_sampler)
    test_loader = torch.utils.data.DataLoader(img_loader(training_data), batch_size=args.test_batch_size, sampler=training_sampler) 

    model = UNet(n_channels=1, n_classes=1).to(device)
    model.double()
    model = model.cuda()
    #use adam optimizer, try SGD if not good results
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #print('First epoch runtime:')
        #print(time.time()-start_time)
        test(args, model, device, test_loader)

    current_daytime = str(datetime.datetime.now()).replace(" ", "_")[:-7]    
    model_file = 'models/UNetModel_'+current_daytime+'.pth'
    torch.save(model.state_dict(), model_file)
    loss_file = 'loss_outputs/loss_'+current_daytime
    with open(loss_file, 'w', newline='') as csvfile:
        losswriter = csv.writer(csvfile, dialect='excel', delimiter=' ', 
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        losswriter.writerow('Batch Size')
        losswriter.writerow(str(args.batch_size))
        losswriter.writerow('Test Batch Size')
        losswriter.writerow(str(args.test_batch_size))
        losswriter.writerow('Epochs')
        losswriter.writerow(str(args.epochs))
        losswriter.writerow('Learning Rate')
        losswriter.writerow(str(args.lr))
        losswriter.writerow('Beta 1')
        losswriter.writerow(str(args.beta1))
        losswriter.writerow('Beta 2')
        losswriter.writerow(str(args.beta2))
        losswriter.writerow('Epsilon')
        losswriter.writerow(str(args.eps))
        
        losswriter.writerow('DICE')
        for item in dice_loss:
            losswriter.writerow(str(round(float(item), 4)))
        
        losswriter.writerow('training')
        for item in training_loss:
            losswriter.writerow(str(round(item, 4)))
        
        losswriter.writerow('testing')
        for item in test_loss:
            losswriter.writerow(str(round(item, 4)))
    
    end_time = time.time()
    print('\nElapsed Time: {:.02f} seconds\n'.format(end_time-start_time))

if __name__ == '__main__':
    main()

