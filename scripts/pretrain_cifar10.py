# -*- coding: utf-8 -*-
import numpy as np
import os

import argparse
import time
import torch

import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from losses import DispLoss, CompLoss
from resnet_anchor import ResNet_Model




from utils.validation_dataset import validation_split
from utils.out_dataset import RandomImages50k

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--score', type=str, default='energy', help='OE|energy')
parser.add_argument('--add_slope', type=int, default=0)
parser.add_argument('--add_class', type=int, default=0)
parser.add_argument('--my_info', type=str, default='TODO')
parser.add_argument('--vanilla', type=int, default=0)
parser.add_argument('--oe', type=int, default=0)
parser.add_argument('--energy_weight', type=float, default=1)  # change this to 19.2 if you are using cifar-100.
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
# Contrastive learning loss
parser.add_argument('--w_disp', default=0.5, type=float, help='L uniform weight')
parser.add_argument('--w_comp', default=0.1, type=float, help='L uniform weight')
parser.add_argument('--proto_m', default=0.95, type=float, help='weight of prototype update')
parser.add_argument('--feat_dim', default=768, type=int, help='feature dim')
parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')
args = parser.parse_args()

save_info = 'text_condition_c10'

args.save = args.save + save_info  # ./snapshots/text_condition_c10
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]  # [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
std = [x / 255 for x in [63.0, 62.1, 66.7]]  # [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

# len = 50000
train_data_in = dset.CIFAR10('/home/zrf/datasets/CIFAR10', train=True, transform=train_transform)
# len = 10000
test_data = dset.CIFAR10('/home/zrf/datasets/CIFAR10', train=False, transform=test_transform)
# len = 10000
val_data = dset.CIFAR10('/home/zrf/datasets/CIFAR10', train=False, transform=test_transform)

if args.add_class:
    num_classes = 11
else:
    num_classes = 10

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

# 加载 CIFAR-10 文本嵌入 anchor
anchor = torch.from_numpy(np.load('./token_embed_c10.npy')).cuda()  # torch.Size([10, 768])


# 196 = 50000 / 256
train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

# 40 = 10000 / 256
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# 40 = 10000 / 256, 用于原型初始化
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

# Create model
net = ResNet_Model(name='resnet34', num_classes=num_classes)



def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = args.load
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# 优化器
optimizer = torch.optim.SGD(
    list(net.parameters()),
    state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

# /////////////// Training ///////////////
criterion = torch.nn.CrossEntropyLoss()
# 10, 768, 0.95, net, val_loader, 0.1
criterion_disp = DispLoss(num_classes, args.feat_dim, args.proto_m, net, val_loader, temperature=args.temp).cuda()
# 10, 0.1
criterion_comp = CompLoss(num_classes, temperature=args.temp).cuda()


def train_vanilla(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0

    for _, in_set in enumerate(train_loader_in):

        data = in_set[0]  # torch.Size([256, 3, 32, 32])
        target = in_set[1]  # torch.Size([256])

        data, target = data.cuda(), target.cuda()  # torch.Size([256, 3, 32, 32]), torch.Size([256])

        # forward
        x = net(data)  # torch.Size([256, 768])
        # x = torch.cdist(x, anchor, p=2.0, compute_mode="use_mm_for_euclid_dist") / 0.1

        # 计算余弦相似度, 其实就是先将 anchor 和 x 归一化, 然后在球面上计算内积相似度
        # anchor: torch.Size([10, 768]) -> torch.Size([1, 10, 768]) -> torch.Size([256, 10, 768])
        # x: torch.Size([256, 768]) -> torch.Size([256, 1, 768]) -> torch.Size([256, 10, 768])
        # / T：温度缩放操作，在这里 T=0.1，它的作用是 放大 余弦相似度的差异，从而使模型更容易学习到区分性特征。
        logits = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x), 1, 1),
                                     x.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1  # torch.Size([256, 10])

        # 相似度作为 logits, 同时 anchor 中的类别顺序和 target 对应的顺序是一致的
        ce_loss = F.cross_entropy(logits, target)  # torch.Size([256, 10]), torch.Size([256])

        # 初始化 comp_loss
        comp_loss = torch.zeros(1).cuda()[0]  # tensor(0., device='cuda:0')

        # 将特征归一化
        normed_features = F.normalize(x, dim=1)  # torch.Size([256, 768])

        # 缩小类间相似性
        # disp_loss = criterion_disp(normed_features, target)  # 归一化特征 torch.Size([256, 128]), torch.Size([256])

        # 动态更新原型
        criterion_disp.update_class_prototypes(normed_features, target)  # 归一化特征 torch.Size([256, 128]), torch.Size([256])

        if epoch >= int(args.epochs // 2):
            # 增大类内相似性
            comp_loss = criterion_comp(normed_features, criterion_disp.prototypes, target)  # 归一化特征 torch.Size([256, 128]), 归一化质心 torch.Size([10, 128]), torch.Size([256])

        # 0.5 * disp_loss + 1 * comp_loss + ce_loss
        # loss = args.w_disp * disp_loss + args.w_comp * comp_loss + ce_loss

        # 0.1 * comp_loss + ce_loss
        loss = args.w_comp * comp_loss + ce_loss
        print("args.w_comp: ", args.w_comp, " comp_loss: ", args.w_comp * comp_loss, " ce_loss: ", ce_loss)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2  # 0.46076436042785646
    print(scheduler.get_lr())
    state['train_loss'] = loss_avg





# Make save directory
if not os.path.exists(args.save):  # ./snapshots/text_condition_c10
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)


save_info = save_info + '_' + args.my_info  # text_condition_c10_TODO


# breakpoint()
# ./snapshots/text_condition_c10/cifar10_wrn_s1_text_condition_c10_TODO_training_results.csv
with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):  # range(0, 200)
    state['epoch'] = epoch

    begin_epoch = time.time()


    train_vanilla(epoch)


    # Save model
    # ./snapshots/text_condition_c10/cifar10_wrn_s1_text_condition_c10_TODO_epoch_0.pt
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + '.pt'))

    # Let us not waste space and delete the previous model
    # ./snapshots/text_condition_c10/cifar10_wrn_s1_text_condition_c10_TODO_epoch_-1.pt
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                             '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results
    # ./snapshots/text_condition_c10/cifar10_wrn_s1_text_condition_c10_TODO_training_results.csv
    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                      '_' + save_info + '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss']
        ))

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'])
    )

# save ID features.
# 采样计数，用于记录每个类别已采样特征的数量。{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
net.eval()
data_dict = torch.zeros(num_classes, 500, 768).cuda()  # torch.Size([10, 500, 768])
with torch.no_grad():
    for _, in_set in enumerate(train_loader_in):

        data = in_set[0]  # torch.Size([256, 3, 32, 32])
        target = in_set[1]  # torch.Size([256])
        data, target = data.cuda(), target.cuda()
        # forward
        feat = net(data)  # torch.Size([256, 768])
        target_numpy = target.cpu().data.numpy()  # array([6, 9, 4, 1, 7, 6, 4, 2, 8, 2, 9])
        for index in range(len(target)):  # range(0, 256)
            dict_key = target_numpy[index]  # 6
            if number_dict[dict_key] < 500:
                # data_dict[dict_key]: 接着已采样的位置(number_dict[dict_key])继续将当前类别的特征加入采样的集合
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1

np.save('./id_feat_cifar10_199epoch.npy', data_dict.cpu().numpy())