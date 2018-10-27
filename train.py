from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from img_loader import img_loader
import torch.utils.model_zoo as model_zoo
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn.metrics as metrics
import datetime
from tabulate import tabulate
from memory_profiler import profile
import time

start_time = time.time()
training_loss = []
test_loss = []
accuracy = []
precision = []
recall = []
confusion = []

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        target = torch.max(target, 1)[1]
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
    correct = 0
    true = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            criterion = nn.CrossEntropyLoss()
            target = torch.max(target, 1)[1]
            true.append(target)
            loss += criterion(output, target).item() 
            avg_loss += loss
            pred = output.max(1, keepdim=True)[1]
            predictions.append(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss /= len(test_loader.dataset)
    test_loss.append(avg_loss)
    
    print('\nTest set statistics:') 
    print('Average loss: {:.4f}'.format(avg_loss)) 
    #Accuracy: {}/{} ({:.0f}%)\n'.format(avg_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    true = np.reshape(torch.stack(true).cpu().data.numpy(), (1000))
    predictions = np.reshape(torch.stack(predictions).cpu().data.numpy(), (1000))
    a = metrics.accuracy_score(true, predictions)
    accuracy.append(a)
    print('Accuracy: {:.0f}%'.format(100. * a))
    r = metrics.recall_score(true, predictions, labels=[0, 1, 2, 3, 4, 5, 6], average='micro')
    recall.append(r)
    print('Recall: {:.2f} '.format(r))
    p = metrics.precision_score(true, predictions, labels=[0, 1, 2, 3, 4, 5, 6], average='micro')
    precision.append(p)
    print('Precision: {:.2f}'.format(p))
    c = metrics.confusion_matrix(true, predictions, labels=[0, 1, 2, 3, 4, 5, 6])
    for item in np.reshape(c, (49, 1)):
        confusion.append(item)
    print('Confusion:')
    print(tabulate(c))

#@profile
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
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # parse csv file to create dictionary for dataloader
    probs_train = []
    probs_test = []
    img_file_train = []
    img_file_test = []
    img_path_train = ['data/train/']*9015
    img_path_test = ['data/test/']*1000
    with open('data/labels/Train_labels.csv', 'r') as f:
        next(f)
        for count, line in enumerate(f):
            file_info = line.split()[0] #get single line
            file_info = file_info.split(',', 1) #separate file name from probs
            img_file_train.append(file_info[0]) #pull out img file str
            probs = file_info[1] #probs, as a single str with commas in it
            probs = probs.split(',') #probs, as a list of strings
            probs = list(map(int, probs)) #probs as a list of ints
            probs_train.append(probs)
    with open('data/labels/Test_labels.csv', 'r') as f:
        next(f)
        for count, line in enumerate(f):
            file_info = line.split()[0] #get single line
            file_info = file_info.split(',', 1) #separate file name from probs
            img_file_test.append(file_info[0]) #pull out img file str
            probs = file_info[1] #probs, as a single str with commas in it
            probs = probs.split(',') #probs, as a list of strings
            probs = list(map(int, probs)) #probs as a list of ints
            probs_test.append(probs)
    data_train = [img_path_train, img_file_train, probs_train]
    data_test = [img_path_test, img_file_test, probs_test] 
    train_loader = torch.utils.data.DataLoader(img_loader(data_train), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(img_loader(data_test), batch_size=args.test_batch_size, shuffle=True, **kwargs) 

    #model = models.resnet18(pretrained=True, **kwargs).to(device)
    model = models.resnet152(pretrained=True).to(device)
    for params in model.parameters():
        params.requires_grad = False
    #only the final classification layer is learnable
    num_ftrs = model.fc.in_features
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
    model = model.cuda()
    #need to use adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #print('First epoch runtime:')
        #print(time.time()-start_time)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), './Resnetmodel.pt')
    
    current_daytime = str(datetime.datetime.now()).replace(" ", "_")[:-7]    
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
        
        losswriter.writerow('accuracy')
        for item in accuracy:
            losswriter.writerow(str(round(item, 4)))
        
        losswriter.writerow('recall')
        for item in recall:
            losswriter.writerow(str(round(item, 4)))
        
        losswriter.writerow('precision')
        for item in precision:
            losswriter.writerow(str(round(item, 4)))
        
        losswriter.writerow('confusion')
        for item in confusion:
            losswriter.writerow(str(item))
        
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

