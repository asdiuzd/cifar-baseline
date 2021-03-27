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

from lr_scheduler import LR_Scheduler

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--device', default=0, type=int, help='running devices')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--warmup_epochs', default=0, type=int, help='warm up epochs')
parser.add_argument('--scheduler_type', default="poly", type=str,
                    choices=['poly', 'circle', 'step', 'cos'],
                    help='learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--net',
                    default='resnet18',
                    type=str,
                    choices=[
                        'resnet18', 'resnet34', 'resnet34_val', 'resnet50',
                        'resnet101', 'mobilenet_v1', 'mobilenet_v2', 'dpn26',
                        'dpn92', "vgg16", 'densenet121', 'efficientnetb0',
                        'resnet18_transformer', 'resnet50_transformer',
                        'resnet18_imagenet'
                    ],
                    help='network')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


log_dir = f'logs/lr{args.lr}_epochs{args.epochs}_device_{args.device}_net{args.net}_lrtype{args.scheduler_type}_warmup{args.warmup_epochs}_batchsize{args.batch_size}'

# if args.mark != "":
#     log_dir = log_dir + f'_{args.mark}'
print(log_dir)
writer = SummaryWriter(log_dir=log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_classes=10

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
print("dataset size:", len(trainloader.dataset))
# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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
net = SimpleDLA()
if args.net == 'resnet18':
    net = ResNet18(num_classes=num_classes)
if args.net == 'resnet34':
    net = ResNet34(num_classes=num_classes)
if args.net == 'resnet34_val':
    net = ResNet34_val(num_classes=num_classes)
if args.net == 'resnet50':
    net = ResNet50(num_classes=num_classes)
elif args.net == 'resnet101':
    net = ResNet101(num_classes=num_classes)
elif args.net == 'mobilenet_v1':
    net = MobileNet(num_classes=num_classes)
elif args.net == 'mobilenet_v2':
    net = MobileNetV2(num_classes=num_classes)
elif args.net == 'dpn26':
    net = DPN26(num_classes=num_classes)
elif args.net == 'dpn92':
    net = DPN92(num_classes=num_classes)
elif args.net == 'densenet121':
    net = DenseNet121(num_classes=num_classes)
elif args.net == "vgg16":
    net = VGG("VGG16")
elif args.net == "efficientnetb0":
    net = EfficientNetB0(num_classes=num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


scheduler = LR_Scheduler(mode='poly', base_lr=args.lr, num_epochs=args.epochs, iters_per_epoch=len(trainloader), warmup_epochs=args.warmup_epochs)

iters_per_epoch=len(trainloader)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        scheduler(optimizer, batch_idx, epoch, best_acc)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), 100)
        
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar("train/loss_iter", loss.item(), iters_per_epoch * epoch + batch_idx)
        writer.add_scalar("train/acc_iter", predicted.eq(targets).sum().item() / targets.size(0), iters_per_epoch * epoch + batch_idx)

    writer.add_scalar("train/loss_epoch", train_loss/iters_per_epoch, epoch)
    writer.add_scalar("train/acc_epoch", 100.*correct/len(trainloader.dataset), epoch)

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
            writer.add_scalar("val/loss_iter", loss.item(), iters_per_epoch * epoch + batch_idx)
            writer.add_scalar("val/acc_iter", predicted.eq(targets).sum().item() / targets.size(0), iters_per_epoch * epoch + batch_idx)


    writer.add_scalar("val/loss_epoch", test_loss/iters_per_epoch, epoch)
    writer.add_scalar("val/acc_epoch", 100.*correct/len(testloader.dataset), epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_dir = log_dir + "/checkpoint"
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_dir + '/ckpt.pth')
        best_acc = acc
    print('best acc:', best_acc)


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
