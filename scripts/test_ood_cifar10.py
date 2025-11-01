import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from resnet import ResNet_Model
from PIL import Image as PILImage


# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utils.svhn_loader as svhn
import utils.lsun_loader as lsun_loader
import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
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
parser.add_argument('--score', default='energy', type=str, help='score options: MSP|energy')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
args = parser.parse_args()
print(args)
torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
std = [x / 255 for x in [63.0, 62.1, 66.7]]  # [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# len = 10000
test_data = dset.CIFAR10('/home/zrf/datasets/CIFAR10', train=False, transform=test_transform)
num_classes = 10

# len = 50 = 10000 / 200
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


net = ResNet_Model(name='resnet34', num_classes=num_classes)

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):

        subdir = 'energy_ft_sd'
        # ./snapshots/energy_ft_sd/cifar10_wrn_s1_energy_ft_sd_slope_0_weight_2.5_samples__epoch_199.pt
        model_name = args.load
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume " + model_name

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:  # 1
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5  # 2000
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))  # 0.16666666666666666

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()



def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):  # torch.Size([200, 3, 32, 32]), torch.Size([200])

            #  OOD数据集只采用2000个数据样本
            #  batch_idx >= 2000 // 200 = 10, batch_idx只进行10次迭代，每次迭代200个样本，10*200=2000个样本
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()  # torch.Size([200, 3, 32, 32])

            output = net(data)  # torch.Size([200, 10])
            smax = to_np(F.softmax(output, dim=1))  # (200, 10)

            if args.use_xent:  # False
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if args.score == 'energy':  # energy
                    # 能量分数 E: (200,)
                    _score.append(-to_np(
                            (args.T * torch.logsumexp(output / args.T, dim=1))))  # 50 个 200, [(200), (200), (200)]
                    # _score.append(to_np(F.softmax(output,1)[:,10]))
                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)  # (200,)
                targets = target.numpy().squeeze()  # (200,)
                right_indices = preds == targets  # correct, (200,)
                wrong_indices = np.invert(right_indices)  # wrong, (200,)

                if args.use_xent:  # False
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    # smax[right_indices]: 分类正确的样本的预测分布, (188, 10)
                    # smax[wrong_indices]: 分类错误的样本的预测分布, (12, 10)
                    _right_score.append(-np.max(smax[right_indices], axis=1))  # (188,) array([-0.99959534, -0.9987826 , -0.9994697])
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))  # (12,) array([-0.5668179 , -0.62840706, -0.5405665])

    if in_dist:
        # _score: len = 50, ((200), (200), (200)) ID 能量分数的集合
        # _right_score: len = 50, ((188), (193), (197)) 预测正确的-概率值集合, 算指标时, label=0, -概率值小
        # _wrong_score：len = 50, ((12), (7), (3)) 预测错误的-概率值集合, 算指标时, label=1, -概率值大
        # (10000), (9527), (473)
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        # OOD数据集只采用2000个数据样本
        # _score: len = 10, ((200), (200), (200)) OOD能量分数的集合
        # concat(_score): (2000)
        # ood_num_examples: 2000
        # concat(_score)[:ood_num_examples]: (2000)
        return concat(_score)[:ood_num_examples].copy()


if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples,
                                                                 args.T, args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable

    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)


    train_data = dset.CIFAR10('/home/zrf/datasets/CIFAR10', train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
    in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count - 1, args.noise,
                                         num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:  # energy
    # ID 能量分数的集合, 预测正确的-概率值集合, 预测错误的-概率值集合
    # (10000), (9527), (473)
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)  # 9527
num_wrong = len(wrong_score)  # 473
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
# (473,), (9527,), 'cifar10_allconv_baseline'
show_performance(wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count - 1,
                                                  args.noise, num_batches)
        else:  # energy
            out_score = get_ood_scores(ood_loader)  # OOD 能量分数的集合: (2000)

        if args.out_as_pos:  # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            # ID负能量分数(-E)的集合 (10000), OOD负能量分数(-E)的集合 (2000)
            measures = get_measures(-in_score, -out_score)  # auroc, aupr, fpr
        aurocs.append(measures[0]);  # auroc
        auprs.append(measures[1]);  # aupr
        fprs.append(measures[2])  # fpr
    # 输出 前3个ID能量分数 和 前3个OOD能量分数
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)  # 5个分数，5个分数，5个分数
    else:
        print_measures(auroc, aupr, fpr, args.method_name)  # 1个分数，1个分数，1个分数

#
# /////////////// Textures ///////////////
# len = 5640
ood_data = dset.ImageFolder(root="/home/zrf/datasets/small_OOD_dataset/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
# len = 29 = 5640 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)


# /////////////// SVHN /////////////// # cropped and no sampling of the test set
# len = 10000
ood_data = svhn.SVHN(root='/home/zrf/datasets/small_OOD_dataset/SVHN', split="test",
                     transform=trn.Compose(
                         [  # trn.Resize(32),
                             trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
# len = 50 = 10000 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
# len = 10000
ood_data = dset.ImageFolder(root="/home/zrf/datasets/small_OOD_dataset/places365",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
# len = 50 = 10000 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)
#/nobackup-slow/dataset/places365_test/test_subset/

# /////////////// LSUN-C ///////////////
# len = 10000
ood_data = dset.ImageFolder(root="/home/zrf/datasets/small_OOD_dataset/LSUN",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
# len = 50 = 10000 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

# # /////////////// LSUN-R ///////////////
# len = 10000
ood_data = dset.ImageFolder(root="/home/zrf/datasets/small_OOD_dataset/LSUN_resize",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
# len = 50 = 10000 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////
# len = 8925
ood_data = dset.ImageFolder(root="/home/zrf/datasets/small_OOD_dataset/iSUN",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
# len = 45 = 8925 / 200
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
get_and_print_results(ood_loader)


# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)



