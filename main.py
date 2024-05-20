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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', default=32, help='Batch size.')
parser.add_argument('--pretrain_epochs', default=10, help='number of epochs')
parser.add_argument('--train_epochs', default=3, help='number of epochs')
parser.add_argument('--log_dir', default="./log.txt", help='path to save log in')
parser.add_argument('--float', action='store_true', help='test floating point model')
parser.add_argument('--best_range_start', default=12, type=int, help='upper bit of best 8-bit range')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--range_delta', default=4, type=int, help='best range delta for quantization')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--per_layer_granularity', action='store_true', help='whether to apply per-tensor granularity')
args = parser.parse_args()

logger = logger.Logger(args.log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

net_name = 'ResNet18'
polarity = "bipolar" #"positive" if adc_utils.positive_weights else "bipolar"
PRETRAIN_PATH = "./" + "_".join([net_name, "pretrained", polarity, "float"]) + ".pth"
TRAIN_PATH = "./" + "_".join([net_name, "pretrained", polarity, "quant"]) + ".pth"

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

# Model
print('==> Building model..')
net = net_dict[net_name]()
net = net.to(device)
print("INSTANTIATING " + net_name)

if os.path.exists(PRETRAIN_PATH):
    checkpoint = torch.load(PRETRAIN_PATH)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    net.load_state_dict(checkpoint, strict=False)


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

def get_mode():
    for m in net.modules():
        if hasattr(m, "mode"):
            return m.mode

def change_mode(mode):
    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = mode

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


def test(epoch, mode="train"):
    SAVE_PATH = PRETRAIN_PATH if mode=="pretrain" else TRAIN_PATH

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
        torch.save(net.state_dict(), SAVE_PATH)
    return acc

def run_training(mode="train"):
    EPOCHS = args.train_epochs if mode=="train" else args.pretrain_epochs
    for epoch in range(0, EPOCHS):
        train(epoch)
        test(epoch, mode=mode)
        scheduler.step()

change_mode("float")
if not os.path.exists(PRETRAIN_PATH):
    logger.log("PRETRAINING FLOATING POINT NETWORK")
    pretrain_accuracy = run_training(mode="pretrain")
else:
    logger.log("LOADING PRETRAINED FLOATING POINT NETWORK")
    net.load_state_dict(torch.load(PRETRAIN_PATH), strict=False)

if args.float:
    change_mode("float")
    test_accuracy = test(0)
    logger.log("TESTING ACCURACY IS: " + str(test_accuracy))
    exit(0)

logger.log("OBSERVING DATA")
change_mode("observe")
observe_data()

logger.log("TESTING QUANTIZED NETWORK")
change_mode("quantize")
#train_accuracy = run_training(mode="train")
test_accuracy = test(0, mode="train")

logger.log("TESTING ACCURACY IS: " + str(test_accuracy))
