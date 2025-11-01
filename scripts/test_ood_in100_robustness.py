import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torch.nn.functional as F
from resnet import ResNet_Model

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=160)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--choice', type=str, default='vanilla')
args = parser.parse_args()
print(args)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
std = [x / 255 for x in [63.0, 62.1, 66.7]]  # [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])



num_classes= 100
net = ResNet_Model(name='resnet34', num_classes=num_classes)
start_epoch = 0

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        subdir = 'energy_ft_sd'
        model_name = args.load

        if os.path.isfile(model_name):
            # net.load_state_dict(remove_data_parallel(torch.load(model_name)))
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume " + model_name

net.eval()


if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = 0  # In this case, we do not need to use it.
expected_ap = 0  # In this case, we do not need to use it.

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):  # torch.Size([160, 3, 224, 224]), torch.Size([160])
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()  # torch.Size([160, 3, 224, 224])

            # output,_,feature = net(data)
            output = net(data)  # torch.Size([160, 100])
            smax = to_np(F.softmax(output, dim=1))  # (160, 100)

            if args.use_xent:  # False
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':  # False
                    # breakpoint()
                    _score.append(-to_np(
                            (args.T * torch.logsumexp(output / args.T, dim=1))))
                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))  # 7 个 160, [(160,), (160,), (160,)]

            if in_dist:  # True
                preds = np.argmax(smax, axis=1)  # (160,)
                targets = target.numpy().squeeze()  # (160,)
                right_indices = preds == targets  # correct, (160,)
                wrong_indices = np.invert(right_indices)  # wrong, (160,)

                if args.use_xent:  # False
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    # smax[right_indices]: 分类正确的样本的预测分布, (136, 100)
                    # smax[wrong_indices]: 分类错误的样本的预测分布, (24, 100)
                    _right_score.append(-np.max(smax[right_indices], axis=1))  # (136,) array([-0.99785304, -0.97689795, -0.94793826])
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))  # (24,) array([-0.26154345, -0.6256833 , -0.28569818])

    if in_dist:  # True
        # _score: len = 7, ((160), (160), (40)) ID -概率值分数集合
        # _right_score: len = 7, ((136), (141), (124)) 预测正确的-概率值集合, 算指标时, label=0, -概率值小
        # _wrong_score：len = 7, ((24), (19), (36)) 预测错误的-概率值集合, 算指标时, label=1, -概率值大
        # (1000), (744), (256)
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def acc_print(test_loader):

    # ID -概率值分数集合, 预测正确的-概率值集合, 预测错误的-概率值集合
    # (1000), (744), (256)
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    num_right = len(right_score)  # 744
    num_wrong = len(wrong_score)  # 256

    return 100 - 100 * num_wrong / (num_wrong + num_right)  # 74.4


# /////////////// imagenet-v2 ///////////////
# len == 1000
test_data = \
    torchvision.datasets.ImageFolder(
    '/home/zrf/datasets/ImageNet_V2/imagenetv2_processed',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))

# len == 7 == 1000 / 160
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)

print(acc_print(test_loader))

# /////////////// imagenet-a ///////////////
# len == 1852
test_data = \
    torchvision.datasets.ImageFolder(
    '/home/zrf/datasets/ImageNet_A/imagenet-a_processed',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))

id_mapping = test_data.class_to_idx  # class -> idx
new_mapping = {}  # idx -> class
for key in list(id_mapping.keys()):
    new_mapping[id_mapping[key]] = int(key)  # idx : class


# len == 1852
test_data = \
    torchvision.datasets.ImageFolder(
    '/home/zrf/datasets/ImageNet_A/imagenet-a_processed',
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]), target_transform=lambda id: new_mapping[id])

# len == 12 == 1852 / 160
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)

print(acc_print(test_loader))

# /////////////// imagenet-100 ///////////////
# len == 5000
test_data = \
    torchvision.datasets.ImageFolder(
    os.path.join('/home/zrf/datasets/ImageNet_100', 'val'),
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))
# len == 32 == 5000 / 160
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                                num_workers=args.prefetch, pin_memory=True)
print(acc_print(test_loader))





