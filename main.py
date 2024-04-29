'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import logger
import adc_utils

# 32-bit precision not enough, as DQ(Q(X)) is not X
#torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', default=32, help='Batch size.')
parser.add_argument('--pretrain_epochs', default=50, help='number of epochs')
parser.add_argument('--train_epochs', default=10, help='number of epochs')
parser.add_argument('--log_dir', default="./log.txt", help='path to save log in')
parser.add_argument('--float', action='store_true', help='test floating point model')
parser.add_argument('--best_range_start', default=23, type=int, help='upper bit of best 8-bit range')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

logger = logger.Logger(args.log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#PRETRAIN_PATH = "./resnet18_pretrained.pth"
PRETRAIN_PATH = "./vgg_pretrained.pth"
if os.path.exists(PRETRAIN_PATH):
    checkpoint = torch.load(PRETRAIN_PATH)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    net.load_state_dict(checkpoint)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def observe_data():
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
    return

def change_mode(mode):
    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = mode
    #if mode == "quantize":
    #    net.prepare_for_quantized_training()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        PRETRAIN_PATH = "./resnet18_pretrained.pth"
        torch.save(net.state_dict(), PRETRAIN_PATH)

def run_training():
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()

for m in net.modules():
    if hasattr(m, 'best_range_start'):
        m.best_range_start = args.best_range_start

if adc_utils.positive_weights:
    counter = 0
    for m in net.modules():
        if isinstance(m, torch.nn.modules.Conv2d):
            # from matplotlib import pyplot as plt
            # plt.hist(m.weight.data.cpu().flatten().numpy(), label="before")
            
            m.weight.data = m.weight.data + 2*torch.sqrt(torch.var(m.weight.data))
            m.weight.data = torch.nn.functional.relu(m.weight.data)

            # plt.hist(m.weight.data.cpu().flatten().numpy(), label="after")
            # plt.legend()
            # plt.savefig("./hists/fig_" + str(counter) + ".png")
            # plt.clf()
            # counter += 1

change_mode("float")
PRETRAIN_PATH = "./resnet18_pretrained.pth"
#PRETRAIN_PATH = "./resnet18_cifar_sgd_positive_weights.pth" if adc_utils.positive_weights else "./resnet18_cifar_sgd.pth"
if not os.path.exists(PRETRAIN_PATH):
    #logger.log("PRETRAINING FLOATING POINT NETWORK")
    pretrain_accuracy = run_training(args.pretrain_epochs, trainloader, testloader, PRETRAIN_PATH)
else:
    #logger.log("LOADING PRETRAINED FLOATING POINT NETWORK")
    net.load_state_dict(torch.load(PRETRAIN_PATH), strict=False)
    #pretrain_accuracy = run_training(args.pretrain_epochs, trainloader, testloader, PRETRAIN_PATH)

#logger.log("TESTING QUANTIZED NETWORK")
if args.float:
    change_mode("float")
else:
    #logger.log("OBSERVING DATA")
    change_mode("observe")
    observe_data()
    change_mode("quantize")
#train_accuracy = run_training()
test_accuracy = test(0)
#logger.log("TESTING ACCURACY IS: " + str(test_accuracy))
logger.log(str(args.best_range_start) + ", " + str(test_accuracy))
