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
import cim_utils
import yaml
from pathlib import Path
from load_config import load_config

args = load_config()

torch.autograd.set_detect_anomaly(True)
quantize_mode=False

logger = logger.Logger(args.log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
n_epochs = args.train_epochs

FLOAT_CKPT_PATH = "./" + "_".join([args.net_name, "float"]) + ".pth"
QUANT_CKPT_PATH = "./" + "_".join([args.net_name, "quant"]) + ".pth"
CKPT_PATH = FLOAT_CKPT_PATH

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
net_dict = {
    'VGG': VGG,
    'ResNet18': ResNet18,
    'PreActResNet18': PreActResNet18,
    'GoogLeNet': GoogLeNet,
    'DenseNet121': DenseNet121,
    'ResNeXt29_2x64d': ResNeXt29_2x64d,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DPN92': DPN92,
    'ShuffleNetG2': ShuffleNetG2,
    'SENet18': SENet18,
    'ShuffleNetV2': ShuffleNetV2,
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF,
    'SimpleDLA': SimpleDLA
}

def observe_data():
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)

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
        if cim_utils.conv_mode=="quantize":
            for p in net.modules():
                if isinstance(p, nn.Conv2d):
                    p.weight.data.copy_(p.weight_org)
        optimizer.step()
        if cim_utils.conv_mode=="quantize":
            for p in net.modules():
                if isinstance(p, nn.Conv2d):
                    p.weight_org.data.copy_(p.weight.data)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    return acc

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if cim_utils.conv_mode=="quantize":
                for p in net.modules():
                    if isinstance(p, nn.Conv2d):
                        p.weight.data.copy_(p.weight_org)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    return acc

def run_training():
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(0, n_epochs):
        train_acc = train(epoch)
        test_acc = test(epoch)
        scheduler.step()
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), CKPT_PATH)
    return best_train_acc, best_test_acc

logger.log("INSTANTIATING " + args.net_name)
net = net_dict[args.net_name]()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

cim_utils.conv_mode = "float"
if os.path.exists(FLOAT_CKPT_PATH):
    logger.log("LOADING PRETRAINED FLOAT WEIGHTS")
    net.load_state_dict(torch.load(FLOAT_CKPT_PATH), strict=False)
else:
    logger.log("PRETRAINING FLOATING NETWORK")
    CKPT_PATH = FLOAT_CKPT_PATH
    train_accuracy, test_accuracy = run_training()

test_accuracy = test(0)
logger.log("FLOAT TESTING ACCURACY IS: " + str(test_accuracy))

# Fuse batchnorm
for p in net.modules():
    if isinstance(p, cim_utils.ADC_Conv2d):
        p.fuse_batchnorm()

test_accuracy = test(0)
logger.log("FUSED BN FLOAT TESTING ACCURACY IS: " + str(test_accuracy))

# TODO: Uncomment when Pytorch observers implemented
"""
# Observe in 2 passes- one for mean, one for var
logger.log("OBSERVING DATA")
cim_utils.conv_mode = "observe"
cim_utils.truncate_observe_mode = "mean"
observe_data()
cim_utils.truncate_observe_mode = "var"
observe_data()
"""

cim_utils.conv_mode = "quantize"
if os.path.exists(QUANT_CKPT_PATH):
    logger.log("LOADING PRETRAINED QUANTIZED NETWORK")
    net.load_state_dict(torch.load(QUANT_CKPT_PATH), strict=False)
else:
    cim_utils.range_mode = "exact"
    cim_utils.range_start = None
    logger.log("TRAINING QUANTIZED NETWORK")
    CKPT_PATH = QUANT_CKPT_PATH
    best_train_acc, best_test_acc = run_training()
    logger.log("BEST QUANTIZED TESTING ACCURACY IS: " + str(best_test_acc))

if args.qat:
    CKPT_PATH = QUANT_CKPT_PATH
    best_train_acc, best_test_acc = run_training()
else:
    net.load_state_dict(torch.load(QUANT_CKPT_PATH), strict=False)
test_acc = test(0)

with open('results.csv', 'a') as f:
    f.write(", ".join([args.net_name, args.signed_weight_approach, args.range_mode, str(args.range_start), str(test_acc)]) + "\n")