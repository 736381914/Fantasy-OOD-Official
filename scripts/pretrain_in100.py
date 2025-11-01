# -*- coding: utf-8 -*-
import numpy as np
import os
import torchvision
import argparse
import time
import torch

import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torch.nn.functional as F
from resnet_anchor import ResNet_Model
from utils.validation_dataset import validation_split
from losses import DispLoss, CompLoss


parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='in100',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn', help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=160, help='Batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# EG specific
parser.add_argument('--my_info', type=str, default='TODO')
parser.add_argument('--seed', type=int, default=1)
# Contrastive learning loss
parser.add_argument('--w_disp', default=0.5, type=float, help='L uniform weight')
parser.add_argument('--w_comp', default=0.1, type=float, help='L uniform weight')
parser.add_argument('--proto_m', default=0.95, type=float, help='weight of prototype update')
parser.add_argument('--feat_dim', default=768, type=int, help='feature dim')
parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')
args = parser.parse_args()

save_info = 'text_condition_in100'

args.save = args.save + save_info
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(args.seed)

traindir = os.path.join('/home/zrf/datasets/ImageNet_100', 'train')
valdir = os.path.join('/home/zrf/datasets/ImageNet_100', 'val')
normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

train_data_in = torchvision.datasets.ImageFolder(
    traindir,
    trn.Compose([
        trn.RandomResizedCrop(224),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        normalize,
    ]))

test_data = torchvision.datasets.ImageFolder(
    valdir,
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))

val_data = torchvision.datasets.ImageFolder(
    valdir,
    trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize,
    ]))

num_classes = 100

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

anchor = torch.from_numpy(np.load('./token_embed_in100.npy')).cuda()


train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)


test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

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

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict
# Restore model
model_found = False
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = args.load
        if os.path.isfile(model_name):
            net.load_state_dict(remove_data_parallel(torch.load(model_name)))
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
# 100, 768, 0.95, net, val_loader, 0.1
criterion_disp = DispLoss(num_classes, args.feat_dim, args.proto_m, net, val_loader, temperature=args.temp).cuda()
# 100, 0.1
criterion_comp = CompLoss(num_classes, temperature=args.temp).cuda()


def train_vanilla(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0

    for _, in_set in enumerate(train_loader_in):

        data = in_set[0]
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        logits = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x), 1, 1),
                                     x.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1

        ce_loss = F.cross_entropy(logits, target)

        # 初始化 comp_loss
        comp_loss = torch.zeros(1).cuda()[0]

        # 将特征归一化
        normed_features = F.normalize(x, dim=1)

        # 动态更新原型
        criterion_disp.update_class_prototypes(normed_features, target)

        if epoch >= int(args.epochs // 2):
            # 增大类内相似性
            comp_loss = criterion_comp(normed_features, criterion_disp.prototypes, target)

        # 0.1 * comp_loss + ce_loss
        loss = args.w_comp * comp_loss + ce_loss
        print("args.w_comp: ", args.w_comp, " comp_loss: ", args.w_comp * comp_loss, " ce_loss: ", ce_loss)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    print(scheduler.get_lr())
    state['train_loss'] = loss_avg


# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)


save_info = save_info + '_' + args.my_info

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                  '_' + save_info + '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()


    train_vanilla(epoch)


    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                            '_' + save_info + '_epoch_' + str(epoch) + '.pt'))

    if epoch % 10 != 0:
        prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model + '_s' + str(args.seed) +
                                 '_' + save_info + '_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)


    # Show results
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
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
net.eval()
data_dict = torch.zeros(num_classes, 1000, 768).cuda()
with torch.no_grad():
    for _, in_set in enumerate(train_loader_in):

        data = in_set[0]
        target = in_set[1]

        data, target = data.cuda(), target.cuda()
        # forward
        feat = net(data)
        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < 1000:
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1

np.save('./id_feat_in100_99epoch.npy', data_dict.cpu().numpy())

