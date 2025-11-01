import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DispLoss(nn.Module):
    def __init__(self, n_cls, feat_dim, proto_m, model, loader, temperature=0.1, base_temperature=0.1):
        super(DispLoss, self).__init__()
        self.n_cls = n_cls  # 10
        self.feat_dim = feat_dim  # 768
        self.proto_m = proto_m  # 0.95
        self.temperature = temperature  # 0.1
        self.base_temperature = base_temperature  # 0.1
        self.register_buffer('prototypes', torch.zeros(self.n_cls, self.feat_dim))  # torch.Size([10, 768])
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):  # torch.Size([256, 768]), torch.Size([256])
        # 用当前 batch 的特征去平滑更新它所属类别的 prototype（EMA 风格）
        prototypes = self.prototypes  # torch.Size([10, 768])
        num_cls = self.n_cls  # 10
        for j in range(len(features)):  # range(0, 256)  动态更新原型
            prototypes[labels[j].item()] = F.normalize(
                prototypes[labels[j].item()] * self.proto_m + features[j] *
                (1 - self.proto_m), dim=0)  # 特征所属类别的原型 * 0.95 + 该特征 * 0.05
        self.prototypes = prototypes.detach()  # torch.Size([10, 768])
        labels = torch.arange(0, num_cls).cuda()  # torch.Size([10]), tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels = labels.contiguous().view(-1, 1)  # torch.Size([10, 1])

        # torch.Size([10, 1]), torch.Size([1, 10])
        mask = (1 - torch.eq(labels, labels.mT).float()).cuda()  # torch.Size([10, 10]), 对角线全为 0

        # torch.Size([10, 768]), torch.Size([768, 10]), 0.1
        logits = torch.div(
            torch.matmul(prototypes, prototypes.mT),  # 计算两两质心的相似度
            self.temperature)  # torch.Size([10, 10])

        # torch.scatter(input, dim, index, src)
        # input: 原始张量（被修改）
        # dim: 要进行操作的维度
        # index: 索引张量，告诉 PyTorch 每个位置的“目标索引”
        # src: 要填入的值，可以是一个数，也可以是和 index 同形状的张量
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )  # torch.Size([10, 10]), 对角线全为 0
        mask = mask * logits_mask  # torch.Size([10, 10])
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))  # torch.Size([10])
        # ~ 是按位取反，意思是保留 不是 NaN 的元素
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]  # torch.Size([10])
        # 希望 loss 最小, 就是希望 mean_prob_neg.mean() 最小, 也就是希望不同类别质心的相似性最小
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def update_class_prototypes(self, features, labels):  # torch.Size([256, 768]), torch.Size([256])
        # 用当前 batch 的特征去平滑更新它所属类别的 prototype（EMA 风格）
        prototypes = self.prototypes  # torch.Size([10, 768])
        num_cls = self.n_cls  # 10
        for j in range(len(features)):  # range(0, 256)  动态更新原型
            prototypes[labels[j].item()] = F.normalize(
                prototypes[labels[j].item()] * self.proto_m + features[j] *
                (1 - self.proto_m), dim=0)  # 特征所属类别的原型 * 0.95 + 该特征 * 0.05
        self.prototypes = prototypes.detach()  # torch.Size([10, 768])

    def init_class_prototypes(self):
        """Initialize class prototypes."""
        self.model.eval()
        start = time.time()
        prototype_counts = [0] * self.n_cls  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with torch.no_grad():
            prototypes = torch.zeros(self.n_cls, self.feat_dim).cuda()  # torch.Size([10, 768])
            for i, (input, target) in enumerate(self.loader):  # torch.Size([256, 3, 32, 32]), torch.Size([256])
                input, target = input.cuda(), target.cuda()  # torch.Size([256, 3, 32, 32]), torch.Size([256])
                features = F.normalize(self.model(input), dim=1)  # torch.Size([256, 768])
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature  # 将 feature 加到它所属类别的原型中
                    prototype_counts[target[j].item()] += 1  # [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
            for cls in range(self.n_cls):  # range(0, 10)
                prototypes[cls] /= prototype_counts[cls]  # 累加的原型求平均
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            # 将原型归一化
            prototypes = F.normalize(prototypes, dim=1)  # torch.Size([10, 768])
            self.prototypes = prototypes  # torch.Size([10, 768])



class CompLoss(nn.Module):
    def __init__(self, n_cls, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.n_cls = n_cls  # 10
        self.temperature = temperature  # 0.1
        self.base_temperature = base_temperature  # 0.07

    def forward(self, features, prototypes, labels):  # torch.Size([256, 128]), torch.Size([10, 128]), torch.Size([256])
        device = torch.device('cuda')

        proxy_labels = torch.arange(0, self.n_cls).to(device)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        batch_size = features.shape[0]  # 256
        labels = labels.contiguous().view(-1, 1)  # torch.Size([256, 1])
        if labels.shape[0] != batch_size:  # False
            raise ValueError('Num of labels does not match num of features')

        # 属于哪个类别, 哪个类别就为1
        # torch.Size([256, 1]), torch.Size([10])
        mask = torch.eq(labels, proxy_labels.T).float().to(device)  # torch.Size([256, 10])

        # compute logits
        anchor_feature = features  # torch.Size([256, 128])
        # 归一化每个类别的质心
        # torch.Size([10, 128]) / torch.Size([10, 1])
        contrast_feature = prototypes / prototypes.norm(dim=-1, keepdim=True)  # torch.Size([10, 128])
        # 计算每个归一化特征与归一化质心的相似度
        # torch.Size([256, 128]), torch.Size([128, 10]), 0.1
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # torch.Size([256, 10])

        # for numerical stability
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # torch.Size([256, 1]), torch.Size([256, 1])
        logits = anchor_dot_contrast - logits_max.detach()  # torch.Size([256, 10])

        # compute log_prob
        exp_logits = torch.exp(logits)  # torch.Size([256, 10])
        # 计算 log-softmax
        # torch.Size([256, 10]), torch.Size([256, 1])
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # torch.Size([256, 10])
        # compute mean of log-likelihood over positive
        # 只保留 GT label 的 logits
        mean_log_prob_pos = (mask * log_prob).sum(1)  # torch.Size([256])
        # 希望 loss 最小, 就是希望 mean_log_prob_pos.mean() 最大, 也就是希望 每个特征和自己类别质心的相似度最大
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss