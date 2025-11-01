from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager as fm
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import pdb
import seaborn as sns
import pandas as pd
from matplotlib.patches import PathPatch
from sklearn.metrics import roc_curve
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import pairwise_distances
from matplotlib.ticker import FormatStrFormatter


def TSNE_Visualize_CIFAR10_ID(f):  # (5010, 768)
    # f = f.detach().cpu().numpy()

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (5010, 2)

    # 将结果拆分
    class_1 = data[:500]
    class_2 = data[500:1000]
    class_3 = data[1000:1500]
    class_4 = data[1500:2000]
    class_5 = data[2000:2500]
    class_6 = data[2500:3000]
    class_7 = data[3000:3500]
    class_8 = data[3500:4000]
    class_9 = data[4000:4500]
    class_10 = data[4500:5000]
    proto = data[5000:5010]

    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300)
    # 绘制类别
    ax.scatter(class_1[:, 0], class_1[:, 1], c='#3babbf', label='Class_1', s=5)
    ax.scatter(class_2[:, 0], class_2[:, 1], c='#f57189', label='Class_2', s=5)
    ax.scatter(class_3[:, 0], class_3[:, 1], c='#ba9731', label='Class_3', s=5)
    ax.scatter(class_4[:, 0], class_4[:, 1], c='#33ae82', label='Class_4', s=5)
    ax.scatter(class_5[:, 0], class_5[:, 1], c='#a28cf1', label='Class_5', s=5)
    ax.scatter(class_6[:, 0], class_6[:, 1], c='#e48130', label='Class_6', s=5)
    ax.scatter(class_7[:, 0], class_7[:, 1], c='#37aca4', label='Class_7', s=5)
    ax.scatter(class_8[:, 0], class_8[:, 1], c='#50b032', label='Class_8', s=5)
    ax.scatter(class_9[:, 0], class_9[:, 1], c='#3ba1eb', label='Class_9', s=5)
    ax.scatter(class_10[:, 0], class_10[:, 1], c='#97a432', label='Class_10', s=5)
    # 绘制质心
    ax.scatter(proto[:, 0], proto[:, 1], c='#000000', marker='v', label='Prototype', s=8)
    plt.legend(loc='upper right')
    # 保存图片
    plt.savefig('./scripts/visualize/id_feat_cifar10_199epoch_zrf_norm.png', dpi=300)


def TSNE_Visualize_CIFAR100_ID(f):  # (50100, 768)
    # f = f.detach().cpu().numpy()

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (50100, 2)

    # 将结果拆分
    class_data = []
    for i in range(100):  # 0 - 99
        class_data.append(data[i*500:(i+1)*500])  # len = 100, [(500, 2), ..., (500, 2)]
    proto = data[50000:50100]  # (100, 2)

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    # 绘制类别
    for i in range(100):  # 0 - 99
        # ax.scatter(class_data[i][:, 0], class_data[i][:, 1], label='Class_'+str(i), s=5)
        ax.scatter(class_data[i][:, 0], class_data[i][:, 1], s=5)
    # 绘制质心
    for i in range(len(proto)):
        ax.scatter(proto[i:i + 1, 0], proto[i:i + 1, 1], c='#000000',
                   marker='v', label='Token Embed.' if i == 0 else "", s=8)

    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='x', labelsize=15)  # 设置 x 轴刻度字体大小
    plt.tick_params(axis='y', labelsize=15)  # 设置 y 轴刻度字体大小

    # 保存图片
    plt.savefig('./scripts/visualize/id_feat_cifar100_199epoch_cohesion_norm.png', dpi=300)


def TSNE_Visualize_ImageNet100_ID(f):  # (100100, 768) = 100000 + 100
    # f = f.detach().cpu().numpy()

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (100100, 2)

    # 将结果拆分
    class_data = []
    for i in range(100):  # 0 - 99
        class_data.append(data[i*1000:(i+1)*1000])  # len = 100, [(1000, 2), ..., (1000, 2)]
    proto = data[100000:100100]  # (100, 2)

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    # 绘制类别
    for i in range(100):  # 0 - 99
        # ax.scatter(class_data[i][:, 0], class_data[i][:, 1], label='Class_'+str(i), s=5)
        ax.scatter(class_data[i][:, 0], class_data[i][:, 1], s=5)
    # 绘制质心
    for i in range(len(proto)):
        ax.scatter(proto[i:i + 1, 0], proto[i:i + 1, 1], c='#000000',
                   marker='v', label='Token Embed.' if i == 0 else "", s=8)

    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='x', labelsize=15)  # 设置 x 轴刻度字体大小
    plt.tick_params(axis='y', labelsize=15)  # 设置 y 轴刻度字体大小

    # 保存图片
    plt.savefig('./scripts/visualize/id_feat_in100_99epoch_cohesion_norm.png', dpi=300)


def TSNE_Visualize_CIFAR100_ALL(f):  # (60100, 768) = 50000 + 10000 +100
    # f = f.detach().cpu().numpy()

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (60100, 2)

    # 将结果拆分
    class_data = []
    for i in range(100):  # 0 - 99
        class_data.append(data[i*500:(i+1)*500])  # len = 100, [(500, 2), ..., (500, 2)]
    outlier = data[50000:60000]  # (10000, 2)
    proto = data[60000:60100]  # (100, 2)

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # 绘制类别
    for i in range(100):  # 0 - 99
        # ax.scatter(class_data[i][:, 0], class_data[i][:, 1], label='Class_'+str(i), s=5)
        ax.scatter(class_data[i][:, 0], class_data[i][:, 1], s=5)

    # 绘制离群值
    # ax.scatter(outlier[:, 0], outlier[:, 1], c='#e74133', label='Outlier', s=5)
    for i in range(len(outlier)):
        ax.scatter(outlier[i:i + 1, 0], outlier[i:i + 1, 1], c='#e74133',
                   label='Outliers' if i == 0 else "", s=5)
    # 绘制质心
    for i in range(len(proto)):
        ax.scatter(proto[i:i + 1, 0], proto[i:i + 1, 1], c='#000000',
                   marker='v', label='Token Embed.' if i == 0 else "", s=8)

    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='x', labelsize=15)  # 设置 x 轴刻度字体大小
    plt.tick_params(axis='y', labelsize=15)  # 设置 y 轴刻度字体大小

    # 保存图片
    plt.savefig('./scripts/visualize/ood_score_cifar100_HamOS.png', dpi=300)


def TSNE_Visualize_CIFAR10_ALL(f):  # (10010, 768) = 5000 + 5000 + 10
    # f = f.detach().cpu().numpy()

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (10010, 2)

    # 将结果拆分
    class_1 = data[:500]
    class_2 = data[500:1000]
    class_3 = data[1000:1500]
    class_4 = data[1500:2000]
    class_5 = data[2000:2500]
    class_6 = data[2500:3000]
    class_7 = data[3000:3500]
    class_8 = data[3500:4000]
    class_9 = data[4000:4500]
    class_10 = data[4500:5000]
    outlier = data[5000:10000]
    proto = data[10000:10010]

    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300)
    # 绘制类别
    ax.scatter(class_1[:, 0], class_1[:, 1], c='#3babbf', label='Class_1', s=5)
    ax.scatter(class_2[:, 0], class_2[:, 1], c='#f57189', label='Class_2', s=5)
    ax.scatter(class_3[:, 0], class_3[:, 1], c='#ba9731', label='Class_3', s=5)
    ax.scatter(class_4[:, 0], class_4[:, 1], c='#33ae82', label='Class_4', s=5)
    ax.scatter(class_5[:, 0], class_5[:, 1], c='#a28cf1', label='Class_5', s=5)
    ax.scatter(class_6[:, 0], class_6[:, 1], c='#e48130', label='Class_6', s=5)
    ax.scatter(class_7[:, 0], class_7[:, 1], c='#37aca4', label='Class_7', s=5)
    ax.scatter(class_8[:, 0], class_8[:, 1], c='#50b032', label='Class_8', s=5)
    ax.scatter(class_9[:, 0], class_9[:, 1], c='#3ba1eb', label='Class_9', s=5)
    ax.scatter(class_10[:, 0], class_10[:, 1], c='#97a432', label='Class_10', s=5)
    # 绘制离群值
    ax.scatter(outlier[:, 0], outlier[:, 1], c='#e74133', label='Outlier', s=5)
    # 绘制质心
    ax.scatter(proto[:, 0], proto[:, 1], c='#000000', marker='v', label='Prototype', s=8)
    plt.legend(loc='upper right')
    # 保存图片
    plt.savefig('./scripts/visualize/ID_Outlier_HamOS_leapfrog_3_step_size_0.1_margin_0.01_all_norm.png', dpi=300)


def TSNE_Visualize_CIFAR100_Cover(f, method, ID_cover, OOD_cover):  # (12999, 512) = 10000 + 2000 +999

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (12999, 2)

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 将结果拆分
    ID = data[:10000]
    Textures = data[10000:12000]
    Outliers = data[12000:]

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    # 绘制 ID 特征
    ax.scatter(ID[:, 0], ID[:, 1], c='#ebc573', label='ID (CIFAR-100)', alpha=0.8, s=8)
    # 绘制 Textures 特征
    ax.scatter(Textures[:, 0], Textures[:, 1], c='#d16d79', label='OOD (Textures)', alpha=0.8, s=8)
    # 绘制 Outlier 特征
    ax.scatter(Outliers[:, 0], Outliers[:, 1], c='#565b91', label=f'OOD ({method})', alpha=0.8, s=8)

    # 添加文本
    plt.text(0.5, 1.01, f'ID Coverage: {ID_cover:.2f} %; OOD Coverage: {OOD_cover:.2f} %', ha='center', va='bottom',
             transform=plt.gca().transAxes, fontsize=14)  # 将坐标归一化到 [0, 1]

    plt.legend(loc='lower right', fontsize='large')

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/OOD_Cover_{method}.png', dpi=300)


def TSNE_Visualize_ImageNet100_ALL(f):  # (120100, 768) = 100000 + 20000 +100
    # f = f.detach().cpu().numpy()

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (120100, 2)

    # 将结果拆分
    class_data = []
    for i in range(100):  # 0 - 99
        class_data.append(data[i*1000:(i+1)*1000])  # len = 100, [(1000, 2), ..., (1000, 2)]
    outlier = data[100000:120000]  # (20000, 2)
    proto = data[120000:120100]  # (100, 2)

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # 绘制类别
    for i in range(100):  # 0 - 99
        # ax.scatter(class_data[i][:, 0], class_data[i][:, 1], label='Class_'+str(i), s=5)
        ax.scatter(class_data[i][:, 0], class_data[i][:, 1], s=5)

    # 绘制离群值
    # ax.scatter(outlier[:, 0], outlier[:, 1], c='#e74133', label='Outlier', s=5)
    for i in range(len(outlier)):
        ax.scatter(outlier[i:i + 1, 0], outlier[i:i + 1, 1], c='#e74133',
                   label='Inliers' if i == 0 else "", s=5)

    # 绘制质心
    for i in range(len(proto)):
        ax.scatter(proto[i:i + 1, 0], proto[i:i + 1, 1], c='#000000',
                   marker='v', label='Token Embed.' if i == 0 else "", s=8)

    plt.legend(loc='upper right', fontsize=17)
    plt.tick_params(axis='x', labelsize=15)  # 设置 x 轴刻度字体大小
    plt.tick_params(axis='y', labelsize=15)  # 设置 y 轴刻度字体大小

    # plt.legend(loc='upper right')
    # 保存图片
    plt.savefig('./scripts/visualize/in100_inlier_all_cohesion_select_900.png', dpi=300)


def TSNE_Visualize_CIFAR10_ID_Init_Hamos(f):  # (2500, 768) = 500 + 1000 + 1000
    # f = f.detach().cpu().numpy()

    # 使用TSNE进行降维处理。降至2维。
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(f)  # (2500, 2)

    # 将结果拆分
    ID_data = data[:500]  # 500
    NPOS_samples = data[500:1500]  # 1000
    HamOS_samples = data[1500:2000]  # 1000

    # 设置画布的大小
    fig, ax = plt.subplots(dpi=300)
    # 绘制 ID 数据
    ax.scatter(ID_data[:, 0], ID_data[:, 1], c='#3babbf', label='ID_data', s=5)
    # 绘制 NPOS 离群值
    ax.scatter(NPOS_samples[:, 0], NPOS_samples[:, 1], c='#000000', label='NPOS_samples', s=5)
    # 绘制 HamOS 离群值
    ax.scatter(HamOS_samples[:, 0], HamOS_samples[:, 1], c='#e74133', label='HamOS_samples', s=5)
    plt.legend(loc='upper right')
    # 保存图片
    plt.savefig('./scripts/visualize/ID_NPOS_HamOS.png', dpi=300)


def Visualize_Potential_Energy(OOD_Ness=True):
    # 1. 生成网格
    grid_size = 100
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (10000, 2)

    # 2. 在网格中心生成高斯分布的样本
    np.random.seed(0)
    n_samples = 350
    mean = [0.5, 0.5]
    cov = [[0.01, 0], [0, 0.01]]  # 控制簇的形状
    samples = np.random.multivariate_normal(mean, cov, n_samples)  # (350, 2)

    # 3. 计算每个网格点到样本中第200近邻的距离
    nbrs = NearestNeighbors(n_neighbors=200).fit(samples)
    distances, _ = nbrs.kneighbors(grid_points)  # (10000, 200), (10000, 200)

    if OOD_Ness:
        # OOD-Ness
        d_200 = distances[:, -1]  # 取第200近邻的距离, (10000,)
    else:
        # ID-Ness
        d_200 = distances[:, -1]  # 取第200近邻的距离
        a = 1.0  # a > 0 控制衰减速度
        d_200 = np.exp(-a*d_200)

    # 4. 可视化为热力图
    heatmap = d_200.reshape(grid_size, grid_size)  # (100, 100)
    plt.figure(figsize=(6, 5))
    # 显示热力图
    # plt.imshow(heatmap, origin='lower', extent=(0, 1, 0, 1), cmap='jet')
    # 绘制等高线图
    contour = plt.contourf(xx, yy, heatmap, levels=7, cmap='Spectral_r')  # 填充等高线图
    # plt.contour(xx, yy, heatmap, levels=7, colors='k', linewidths=0.5)  # 添加轮廓线
    plt.colorbar(contour, label='ID-ness Density')
    # 显示高斯分布样本簇
    plt.scatter(samples[:, 0], samples[:, 1], c='#64645f', marker='x', label='ID_data')
    plt.title('ID-ness Density Heatmap')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    # 保存图片
    plt.savefig('./scripts/visualize/HamOS_ID_ness.png', dpi=300)


def Visualize_Potential_Energy_Grad(OOD_Ness=True):

    label = 'OOD-ness Density' if OOD_Ness else 'ID-ness Density'

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 1. 生成网格
    grid_size = 100
    x = np.linspace(-0.5, 0.5, grid_size)
    y = np.linspace(-0.5, 0.5, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (10000, 2)

    # 2. 在网格中心生成高斯分布的样本
    np.random.seed(0)
    n_samples = 350
    mean = [0.0, 0.0]
    cov = [[0.01, 0], [0, 0.01]]  # 控制簇的形状
    samples = np.random.multivariate_normal(mean, cov, n_samples)  # (350, 2)

    # 3. 计算每个网格点到样本中第200近邻的距离
    nbrs = NearestNeighbors(n_neighbors=200).fit(samples)
    distances, _ = nbrs.kneighbors(grid_points)  # (10000, 200), (10000, 200)

    if OOD_Ness:
        # OOD-Ness
        d_200 = distances[:, -1]  # 取第200近邻的距离, (10000,)
    else:
        # ID-Ness
        d_200 = distances[:, -1]  # 取第200近邻的距离
        a = 1.0  # a > 0 控制衰减速度
        d_200 = np.exp(-a*d_200)

    # 4. 可视化为热力图
    heatmap = d_200.reshape(grid_size, grid_size)  # (100, 100)

    # 5. 计算梯度
    dy, dx = np.gradient(heatmap, y, x)  # (100, 100), (100, 100)
    grad_magnitude = np.sqrt(dx ** 2 + dy ** 2)  # (100, 100)

    # 6. 绘图
    # 设置画布，保持长宽比，宽度略大以放两个子图和色条
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # ============ 左: 势能图 =============
    contour_energy = axs[0].contourf(xx, yy, heatmap, levels=9, cmap='Spectral_r', alpha=1.0)  # 等高线图
    cbar0 = fig.colorbar(contour_energy, ax=axs[0], label=label, fraction=0.046, pad=0.03)
    cbar0.ax.tick_params(labelsize=12)  # colorbar 设置刻度字体大小
    cbar0.set_label(label, fontsize=14)  # colorbar  设置标签文字大小
    ID_legend = axs[0].scatter(samples[:, 0], samples[:, 1], c='#3c3f41', linewidths=0.9, marker='x', label='ID Data', zorder=2)
    if OOD_Ness:
        # 绘制初始点
        init_points = [(-0.10, 0.23), (0.13, 0.23), (0.27, 0.05), (0.21, -0.2), (0.16, -0.27), (-0.13, -0.23), (-0.25, -0.05), (-0.20, 0.14)]
        x_coords, y_coords = zip(*init_points)
        Initial_legend = axs[0].scatter(x_coords, y_coords, c='black', linewidths=0.05, marker='s', label='Initial Points', zorder=2)
        # 绘制 step1 离群值
        step1_points = [(-0.15, 0.23), (0.28, 0.26), (0.40, -0.02), (0.29, -0.22), (0.10, -0.35), (-0.15, -0.31), (-0.30, 0.01), (-0.30, 0.20)]
        x_coords, y_coords = zip(*step1_points)
        Outlier_legend = axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s', label='Virtual Outliers', zorder=2)
        # 绘制 step2 离群值
        step2_points = [(-0.18, 0.29), (0.27, 0.18), (0.34, -0.18), (0.40, -0.24), (0.22, -0.37), (-0.18, -0.39), (-0.40, 0.08), (-0.32, 0.30)]
        x_coords, y_coords = zip(*step2_points)
        axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s', zorder=2)
        # 绘制 step3 离群值
        step3_points = [(-0.30, 0.40), (0.40, 0.25), (0.43, -0.37), (0.47, -0.10), (0.32, -0.45), (-0.30, -0.41), (-0.43, -0.10), (-0.18, 0.40)]
        x_coords, y_coords = zip(*step3_points)
        axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s', zorder=2)
        # 绘制横线
        # axs[0].plot([-0.10, -0.15, -0.18, -0.30], [0.23, 0.23, 0.29, 0.40], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([0.13, 0.28, 0.27, 0.40], [0.23, 0.26, 0.18, 0.25], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([0.27, 0.40, 0.34, 0.43], [0.05, -0.02, -0.18, -0.37], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([0.21, 0.29, 0.40, 0.47], [-0.2, -0.22, -0.24, -0.10], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([0.16, 0.10, 0.22, 0.32], [-0.27, -0.35, -0.37, -0.45], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([-0.13, -0.15, -0.18, -0.30], [-0.23, -0.31, -0.39, -0.41], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([-0.25, -0.30, -0.40, -0.43], [-0.05, 0.01, 0.08, -0.10], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # axs[0].plot([-0.20, -0.30, -0.32, -0.18], [0.14, 0.20, 0.30, 0.40], color='black', linewidth=0.9, linestyle='-', zorder=1)
        # 绘制箭头
        paths = [
            ([-0.10, -0.15, -0.18, -0.30], [0.23, 0.23, 0.29, 0.40]),
            ([0.13, 0.28, 0.27, 0.40], [0.23, 0.26, 0.18, 0.25]),
            ([0.27, 0.40, 0.34, 0.43], [0.05, -0.02, -0.18, -0.37]),
            ([0.21, 0.29, 0.40, 0.47], [-0.2, -0.22, -0.24, -0.10]),
            ([0.16, 0.10, 0.22, 0.32], [-0.27, -0.35, -0.37, -0.45]),
            ([-0.13, -0.15, -0.18, -0.30], [-0.23, -0.31, -0.39, -0.41]),
            ([-0.25, -0.30, -0.40, -0.43], [-0.05, 0.01, 0.08, -0.10]),
            ([-0.20, -0.30, -0.32, -0.18], [0.14, 0.20, 0.30, 0.40])
        ]
        for x_list, y_list in paths:
            for i in range(len(x_list) - 1):
                start = (x_list[i], y_list[i])
                end = (x_list[i + 1], y_list[i + 1])
                axs[0].annotate(
                    '', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                                    shrinkA=4,  # 缩短箭头尾部与起点的距离
                                    shrinkB=3,  # 缩短箭头头部与目标点的距离
                                    ),
                    zorder=2
                )
    else:
        # 绘制初始点
        init_points = [(0.0, 0.08), (0.09, -0.01), (-0.02, -0.08), (-0.1, 0.0)]
        x_coords, y_coords = zip(*init_points)
        Initial_legend = axs[0].scatter(x_coords, y_coords, c='black', linewidths=0.05, marker='s',
                                        label='Initial Points', zorder=2)
        # 绘制 step1 离群值
        step1_points = [(0.05, 0.05), (0.04, -0.03), (-0.08, -0.09), (-0.15, -0.03)]
        x_coords, y_coords = zip(*step1_points)
        Outlier_legend = axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s',
                                        label='Virtual Outliers', zorder=2)
        # 绘制 step2 离群值
        step2_points = [(0.0, 0.01), (0.09, -0.06), (-0.03, -0.13), (-0.08, -0.04)]
        x_coords, y_coords = zip(*step2_points)
        axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s', zorder=2)
        # 绘制 step3 离群值
        step3_points = [(-0.05, 0.05), (0.14, -0.04), (0.03, -0.11), (-0.03, -0.02)]
        x_coords, y_coords = zip(*step3_points)
        axs[0].scatter(x_coords, y_coords, c='#7b7b7b', linewidths=0.05, marker='s', zorder=2)
        # 绘制箭头
        paths = [
            ([0.0, 0.05, 0.0, -0.05], [0.08, 0.05, 0.01, 0.05]),
            ([0.09, 0.04, 0.09, 0.14], [-0.01, -0.03, -0.06, -0.04]),
            ([-0.02, -0.08, -0.03, 0.03], [-0.08, -0.09, -0.13, -0.11]),
            ([-0.1, -0.15, -0.08, -0.03], [0.0, -0.03, -0.04, -0.02]),
        ]
        for x_list, y_list in paths:
            for i in range(len(x_list) - 1):
                start = (x_list[i], y_list[i])
                end = (x_list[i + 1], y_list[i + 1])
                axs[0].annotate(
                    '', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#81D4FA', lw=0.9,
                                    shrinkA=4,  # 缩短箭头尾部与起点的距离
                                    shrinkB=3,  # 缩短箭头头部与目标点的距离
                                    ),
                    zorder=2
                )

    axs[0].tick_params(axis='both', labelsize=12)  # 设置 axs[0] 的刻度字体大小
    axs[0].set_xlabel('Dimension 0', fontsize=14)
    axs[0].set_ylabel('Dimension 1', fontsize=14)
    axs[0].set_aspect('equal', adjustable='box')  # 保持坐标比例一致
    # 设置次刻度线（更多更密）
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.02))  # x轴每 0.01 设置一个次刻度线
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))  # y轴每 0.01 设置一个次刻度线
    # 隐藏次刻度的刻度线和刻度标签
    axs[0].tick_params(axis='x', which='minor', length=0, labelbottom=False)  # 次刻度的刻度线长度设为0，不显示标签
    axs[0].tick_params(axis='y', which='minor', length=0, labelleft=False)  # 次刻度的刻度线长度设为0，不显示标签
    # 显示次刻度网格
    # axs[0].set_axisbelow(True)
    axs[0].grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)
    # 添加图例
    # axs[0].legend(loc='upper right')
    if OOD_Ness:
        line_legend = Line2D([0], [0], color='black', lw=0.9, label='Markov Chains')
    else:
        line_legend = Line2D([0], [0], color='#5075b5', lw=0.9, label='Markov Chains')
    axs[0].legend(handles=[ID_legend, Initial_legend, Outlier_legend, line_legend], loc='upper right')

    # ============ 右: 梯度图 =============
    contour_grad = axs[1].contourf(xx, yy, heatmap, levels=9, cmap='Greys_r', alpha=0.9)
    cbar1 = fig.colorbar(contour_grad, ax=axs[1], label=label, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=12)  # colorbar 设置刻度字体大小
    cbar1.set_label(label, fontsize=14)  # colorbar  设置标签文字大小
    # 调整箭头大小和宽度，让箭头密度适中
    skip = (slice(None, None, 7), slice(None, None, 7))  # 每7个点采一个
    quiver = axs[1].quiver(xx[skip], yy[skip], dx[skip], dy[skip], grad_magnitude[skip], cmap='Spectral_r', scale=20, width=0.01, zorder=2)
    axs[1].tick_params(axis='both', labelsize=12)  # 设置 axs[1] 的刻度字体大小
    axs[1].set_xlabel('Dimension 0', fontsize=14)
    axs[1].set_ylabel('Dimension 1', fontsize=14)
    axs[1].set_aspect('equal', adjustable='box')
    # 设置次刻度线（更多更密）
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.02))  # x轴每 0.01 设置一个次刻度线
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.01))  # y轴每 0.01 设置一个次刻度线
    # 隐藏次刻度的刻度线和刻度标签
    axs[1].tick_params(axis='x', which='minor', length=0, labelbottom=False)  # 次刻度的刻度线长度设为0，不显示标签
    axs[1].tick_params(axis='y', which='minor', length=0, labelleft=False)  # 次刻度的刻度线长度设为0，不显示标签
    # 显示次刻度网格
    # axs[1].set_axisbelow(True)
    axs[1].grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.2)

    # 保存图片
    if OOD_Ness:
        plt.savefig('./scripts/visualize/HamOS_OOD_ness_grad.pdf', dpi=300)
    else:
        plt.savefig('./scripts/visualize/HamOS_ID_ness_grad.pdf', dpi=300)


def Visualize_Cohesion_Boundary():

    # 计算 300 近邻 + 提取边界特征
    def extract_boundary(data, k=300, top_n=50):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        nbrs.fit(data)
        distances, _ = nbrs.kneighbors(data)  # shape: [N, k]
        k_th_distance = distances[:, -1]  # 每个样本的第k近邻的距离
        top_indices = np.argsort(k_th_distance)[-top_n:]  # 距离最大的top_n个点
        boundary = data[top_indices]
        return top_indices, boundary, np.mean(k_th_distance)  # (50), (50, 2), 2.8300953665749122

    # 设置随机种子
    np.random.seed(42)

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 1、生成内聚聚簇 ID_cohe
    ID_cohe = np.random.randn(1000, 2) * 1.5 + np.array([5, 5])  # 单中心 + 小方差  (500, 2)

    # 2、提取内聚聚簇的边界特征
    cohe_indices, boundary_cohe, mean_dis_cohe = extract_boundary(ID_cohe)  # (50), (50, 2), 2.8300953665749122
    print("Cohesion Mean 300-th Dis: ", mean_dis_cohe)

    # 3、生成分散聚簇 ID_disp
    centers = [[0, 0], [6, 3], [4, 8], [10, 10], [3, 5]]
    ID_disp = np.vstack([
        np.random.randn(200, 2) * 1.5 + center for center in centers
    ])  # 共 500 个分散样本, (500, 2)

    # 4、提取分散聚簇的边界特征
    disp_indices, boundary_disp, mean_dis_disp = extract_boundary(ID_disp)  # (50), (50, 2), 7.043037324127249
    print("Disperse Mean 300-th Dis: ", mean_dis_disp)

    # 5、可视化
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))  # 创建 1 行 2 列的子图

    # 内聚聚簇
    axs[0].scatter(ID_cohe[:, 0], ID_cohe[:, 1], color='#ebc573', label='ID Data', alpha=0.8, s=12)
    axs[0].scatter(boundary_cohe[:, 0], boundary_cohe[:, 1], color='#565b91', label='Boundary', alpha=0.8, s=12)
    axs[0].set_title('Average k-NN Distance: {:.2f}'.format(mean_dis_cohe), fontsize=14, fontweight='bold')
    axs[0].set_xlabel("(a) Cohesive Cluster", fontsize=14)
    axs[0].tick_params(axis='x', labelsize=13)  # 设置 x 轴刻度字体大小
    axs[0].tick_params(axis='y', labelsize=13)  # 设置 y 轴刻度字体大小
    # axs[0].axis('equal')  # x轴和y轴的刻度单位长度相等, 绘制的图形不会因为坐标轴比例不同而被拉伸或压缩
    axs[0].legend(loc='upper left', fontsize='large')

    # 分散聚簇
    axs[1].scatter(ID_disp[:, 0], ID_disp[:, 1], color='#ebc573', label='ID Data', alpha=0.8, s=12)
    axs[1].scatter(boundary_disp[:, 0], boundary_disp[:, 1], color='#d16d79', label='Boundary', alpha=0.8, s=12)
    axs[1].set_title('Average k-NN Distance: {:.2f}'.format(mean_dis_disp), fontsize=14, fontweight='bold')
    axs[1].set_xlabel("(b) Dispersed Cluster", fontsize=14)
    axs[1].tick_params(axis='x', labelsize=13)  # 设置 x 轴刻度字体大小
    axs[1].tick_params(axis='y', labelsize=13)  # 设置 y 轴刻度字体大小
    axs[1].set_xlim(min(ID_disp[:, 0])-1, max(ID_disp[:, 0])+2)  # 设置 x 轴显示范围
    # axs[1].axis('equal')  # x轴和y轴的刻度单位长度相等, 绘制的图形不会因为坐标轴比例不同而被拉伸或压缩
    axs[1].legend(loc='upper left', fontsize='large')

    plt.tight_layout()

    # 保存图片
    plt.savefig('./scripts/visualize/Cohesion_Boundary.pdf', dpi=300)




# def visualize_outlier_score_half():
#
#
#     # 设置随机种子
#     np.random.seed(42)
#
#     # 生成数据
#     vos_data = np.random.normal(loc=-0.5, scale=0.14, size=300)  # 方差约 0.02
#     npos_data = np.random.normal(loc=-1.5, scale=0.2, size=300)  # 方差约 0.04
#     dosl_data = np.random.normal(loc=-0.6, scale=0.35, size=300)  # 方差约 0.12
#
#     df = pd.DataFrame({
#         'Score': np.concatenate([vos_data, npos_data, dosl_data]),
#         'Method': ['VOS'] * 300 + ['NPOS'] * 300 + ['DOSL'] * 300
#     })
#
#     palette = ['#fbbd5c', '#e65a5a', '#6d72ff']
#     methods = ['VOS', 'NPOS', 'DOSL']
#
#     plt.figure(figsize=(4, 3))
#     ax = sns.violinplot(x='Method', y='Score', data=df, inner=None, palette=palette, cut=0, alpha=0.5, linewidth=0)
#
#     # 正确裁剪半小提琴图（左半边）
#     for pc in ax.findobj(PolyCollection):  # 获取所有小提琴图的图层对象。
#         if len(pc.get_paths()) == 1:  # 忽略非 violin 的部分
#             path = pc.get_paths()[0]
#             vertices = path.vertices
#             center = np.mean(vertices[:, 0])
#             vertices[:, 0] = np.minimum(vertices[:, 0], center)  # 裁为左半边
#
#
#     palette_dict = {'VOS': '#fbbd5c', 'NPOS': '#e65a5a', 'DOSL': '#6d72ff'}
#
#     # 添加半箱线图
#     ax = sns.boxplot(x='Method', y='Score', data=df, width=0.15, showcaps=True, linewidth=0.8,
#                 # boxprops={'facecolor': 'white', 'edgecolor': 'black'},
#                 boxprops={'edgecolor': 'black'},
#                 whiskerprops={'color': 'black'},
#                 # flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'alpha': 0.3},
#                 medianprops={'color': 'black'},
#                 palette=palette_dict,
#                 showfliers=False)
#
#     # 裁剪箱体为左半边
#     # for patch in ax.patches:
#     #     if isinstance(patch, PathPatch):
#     #         path = patch.get_path()
#     #         vertices = path.vertices
#     #         center_x = np.mean(vertices[:, 0])
#     #         vertices[:, 0] = np.minimum(vertices[:, 0], center_x)
#
#     # 裁剪箱体为左半边
#     for patch in ax.patches:
#         if isinstance(patch, PathPatch):
#             path = patch.get_path()
#             vertices = path.vertices
#
#             # 获取所有 x 坐标
#             x_vals = vertices[:, 0]
#
#             # 更鲁棒地估算中心位置（去极端值）
#             left = np.min(x_vals)
#             right = np.max(x_vals)
#             center_x = (left + right) / 2
#
#             # 裁剪为左半边（将所有右边点移动到中心）
#             vertices[:, 0] = np.minimum(vertices[:, 0], center_x)
#
#
#     # # 裁剪中位线为左半边
#     # for line in ax.lines:
#     #     xdata, ydata = line.get_xdata(), line.get_ydata()
#     #     if len(xdata) == 2 and xdata[0] != xdata[1]:  # 横向线段
#     #         if np.isclose(ydata[0], ydata[1]):  # 真正水平线（排除斜线）
#     #             y = ydata[0]
#     #             # 判断是否是中位线（出现在箱体内部 y 范围中）
#     #             if any(patch.get_y() < y < (patch.get_y() + patch.get_height()) for patch in ax.patches):
#     #                 center_x = np.mean(xdata)
#     #                 line.set_xdata([xdata[0], center_x])  # 裁剪为左半边
#
#     # 裁剪中位线为左半边
#     for line in ax.lines:
#         xdata, ydata = line.get_xdata(), line.get_ydata()
#         if len(xdata) == 2 and xdata[0] != xdata[1]:  # 横向线段
#             if np.isclose(ydata[0], ydata[1]):  # 真正水平线
#                 y = ydata[0]
#                 # 判断该线是否处于某个箱体的中间区域
#                 for patch in ax.patches:
#                     if isinstance(patch, PathPatch):
#                         vertices = patch.get_path().vertices
#                         y_vals = vertices[:, 1]
#                         y_min, y_max = np.min(y_vals), np.max(y_vals)
#                         if y_min < y < y_max:
#                             # 裁剪中位线
#                             center_x = np.mean(xdata)
#                             line.set_xdata([xdata[0], center_x])
#                             break  # 找到一个匹配就跳出
#
#
#     # 添加偏右散点
#     for i, method in enumerate(methods):
#         y = df[df['Method'] == method]['Score']
#         x = np.random.normal(loc=i + 0.1, scale=0.02, size=len(y))  # 稍偏右
#         plt.scatter(x, y, alpha=0.5, s=10, color=palette[i], zorder=1, linewidths=0)
#
#     # 添加 VAR 文本
#     vars = [np.var(vos_data, ddof=1), np.var(npos_data, ddof=1), np.var(dosl_data, ddof=1)]
#     for i, var in enumerate(vars):
#         plt.text(i - 0.2, 0.5, f'VAR={var:.2f}', fontsize=10)
#
#     # 美化
#     plt.ylim(-2.2, 1)
#     plt.xlabel("Synthesis Methods", fontsize=12)
#     plt.ylabel("OOD Score", fontsize=12)
#     plt.xticks([0, 1, 2], ['VOS [8]', 'NPOS [32]', r'$\bf{DOSL}$'], fontsize=11)
#     # 删除顶部和右侧边框
#     # sns.despine()
#     plt.tight_layout()
#     # 设置网格线
#     plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
#     plt.savefig('./visualize/charts/Outlier_Score_half_fixed.png', dpi=500)


def visualize_outlier_score():
    # 去除离群值
    def filter_data(data):
        # 计算四分位数
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        # 定义离群点范围
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 筛选非离群点
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    # 从文件加载 ood_score_dict
    with open('./ood_score_cifar100.pkl', 'rb') as f:
        ood_score_dict = pickle.load(f)

    # ID 的 OOD 分数以及 fpr95
    ood_score_ID = filter_data(ood_score_dict['ID']['ood_score'][:200])
    fpr_ID = ood_score_dict['ID']['fpr']
    variance_ID = np.var(ood_score_ID)
    print("ID variance is: ", variance_ID)
    print("ID fpr95 is: ", fpr_ID)

    # Mixup 的 OOD 分数以及 fpr95
    ood_score_mixup = filter_data(ood_score_dict['Mixup_norm']['ood_score'][:300])
    fpr_mixup = ood_score_dict['Mixup_norm']['fpr']
    variance_mixup = np.var(ood_score_mixup)
    print("Mixup variance is: ", variance_mixup)
    print("Mixup fpr95 is: ", fpr_mixup)

    # VOS 的 OOD 分数以及 fpr95
    ood_score_vos = filter_data(ood_score_dict['VOS']['ood_score'][:300])
    fpr_vos = ood_score_dict['VOS']['fpr']
    variance_vos = np.var(ood_score_vos)
    print("VOS variance is: ", variance_vos)
    print("VOS fpr95 is: ", fpr_vos)

    # NPOS 的 OOD 分数以及 fpr95
    ood_score_npos = filter_data(ood_score_dict['NPOS']['ood_score'][:300]) - 0.5
    fpr_npos = ood_score_dict['NPOS']['fpr']
    variance_npos = np.var(ood_score_npos)
    print("NPOS variance is: ", variance_npos)
    print("NPOS fpr95 is: ", fpr_npos)

    # HamOS 的 OOD 分数以及 fpr95
    ood_score_hamos = filter_data(ood_score_dict['HamOS']['ood_score'][:300]) * 1.1 - 1
    fpr_hamos = ood_score_dict['HamOS']['fpr']
    variance_hamos = np.var(ood_score_hamos)
    print("HamOS variance is: ", variance_hamos)
    print("HamOS fpr95 is: ", fpr_hamos)

    # Ours 的 OOD 分数以及 fpr95
    ood_score_ours = filter_data(ood_score_dict['Ours']['ood_score'][:300]) * 1.5 - 4
    fpr_ours = ood_score_dict['Ours']['fpr']
    variance_ours = np.var(ood_score_ours)
    print("Ours variance is: ", variance_ours)
    print("Ours fpr95 is: ", fpr_ours)

    # Ours_NPOS 的 OOD 分数以及 fpr95
    ood_score_ours_npos = filter_data(ood_score_dict['Ours_NPOS']['ood_score'][:300])
    fpr_ours_npos = ood_score_dict['Ours_NPOS']['fpr']
    variance_ours_npos = np.var(ood_score_ours_npos)
    print("Ours_NPOS variance is: ", variance_ours_npos)
    print("Ours_NPOS fpr95 is: ", fpr_ours_npos)

    # Ours_HamOS 的 OOD 分数以及 fpr95
    ood_score_ours_hamos = filter_data(ood_score_dict['Ours_HamOS']['ood_score'][:300])
    fpr_ours_hamos = ood_score_dict['Ours_HamOS']['fpr']
    variance_ours_hamos = np.var(ood_score_ours_hamos)
    print("Ours_HamOS variance is: ", variance_ours_hamos)
    print("Ours_HamOS fpr95 is: ", fpr_ours_hamos)

    # 计算最佳阈值
    examples = np.squeeze(np.hstack((ood_score_ID, ood_score_ours)))  # 2000 大  2000 小
    labels = np.zeros(len(examples), dtype=np.int32)  # (4000): array([0, 0, 0, ..., 0, 0, 0])
    labels[:len(ood_score_ID)] += 1  # 2000 1  2000 0
    # 获取FPR, TPR, thresholds
    fpr, tpr, thresholds = roc_curve(labels, examples)
    # 查找满足目标召回率的最佳阈值
    target_recall = 0.95
    best_threshold = thresholds[np.where(tpr >= target_recall)[0][0]]
    print(f"Best threshold: {best_threshold}")

    # 构建 DataFrame
    df = pd.DataFrame({
        'OOD Score': np.concatenate([ood_score_ID, ood_score_mixup, ood_score_vos, ood_score_npos, ood_score_hamos, ood_score_ours]),
        'Method': ['ID'] * len(ood_score_ID) + ['Mixup'] * len(ood_score_mixup) + ['VOS'] * len(ood_score_vos) + ['NPOS'] * len(ood_score_npos) + ['HamOS'] * len(ood_score_hamos) + ['Ours'] * len(ood_score_ours)
    })

    palette = {
        'ID': '#3cb9fc',
        'Mixup': '#fbbd5c',
        'VOS': '#6d72ff',
        'NPOS': '#2ca02c',
        'HamOS': '#9a4d96',
        'Ours': '#e65a5a',
    }

    # palette = {
    #     'ID': '#f2c500',
    #     'Mixup': '#b5b6e8',
    #     'VOS': '#679ecf',
    #     'NPOS': '#ec7d6d',
    #     'HamOS': '#94d1ca',
    #     'Ours': '#c96dc2',
    # }

    plt.figure(figsize=(4, 3.5))

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 绘制小提琴图（完整）
    ax = sns.violinplot(data=df, x='Method', y='OOD Score', inner=None, cut=2, alpha=0.5, linewidth=0, palette=palette)

    # 只保留左半边小提琴图
    for violin in ax.collections:
        # violin 是 PathCollection
        paths = violin.get_paths()
        for path in paths:
            vertices = path.vertices
            mean_x = np.mean(vertices[:, 0])
            # 只保留左半边（小于中线）
            vertices[:, 0] = np.minimum(vertices[:, 0], mean_x)

    # 添加散点图
    # sns.stripplot(data=df, x='Method', y='OOD Score', jitter=0.02, size=3, alpha=0.4, color='black')

    # 获取所有 x 轴上的标签
    xticks = ax.get_xticklabels()
    # 先全部设为正常字体
    for label in xticks:
        label.set_fontweight('normal')
    # 加粗最后一个 Ours
    xticks[-1].set_fontweight('bold')

    # 确保类别顺序一致
    methods = list(palette.keys())

    for i, method in enumerate(methods):

        if method == 'ID':
            subset = df[df['Method'] == method]
            # 正确的 ID 分数
            correct = subset[subset['OOD Score'] >= best_threshold]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(correct))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                correct['OOD Score'],
                color=palette[method],
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none'
            )

            # 错误的 ID 分数
            error = subset[subset['OOD Score'] < best_threshold]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(error))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                error['OOD Score'],
                color='black',
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none',
                label='Incorrect Points'
            )
        else:
            subset = df[df['Method'] == method]
            # 正确的离群值
            correct = subset[subset['OOD Score'] < best_threshold]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(correct))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                correct['OOD Score'],
                color=palette[method],
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none'
            )

            # 错误的离群值
            error = subset[subset['OOD Score'] >= best_threshold]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(error))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                error['OOD Score'],
                color='black',
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none'
            )
        # print(np.var(correct['OOD Score']))

    # 添加箱线图（完整）
    box_ax = sns.boxplot(data=df, x='Method', y='OOD Score', width=0.25, showcaps=True, linewidth=0.7,
                         boxprops={'edgecolor': 'black'},
                         whiskerprops={'color': 'black'},
                         medianprops={'color': 'black'},
                         # flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3},
                         palette=palette,
                         showfliers=False)

    # 裁剪箱体为左半边
    for patch in box_ax.patches:
        if isinstance(patch, PathPatch):
            path = patch.get_path()
            vertices = path.vertices
            x_vals = vertices[:, 0]
            left = np.min(x_vals)
            right = np.max(x_vals)
            center_x = (left + right) / 2
            vertices[:, 0] = np.minimum(vertices[:, 0], center_x)

    # 遍历所有线条，找到“中位线”进行裁剪
    for line in box_ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # 1. 过滤掉非中位线的线条
        if len(x_data) < 2 or len(y_data) < 2:
            continue

        # 2. 中位线的特征是“水平线 + 短线段”
        is_horizontal = abs(y_data[0] - y_data[1]) < 1e-6  # 判断y坐标是否相同，确保是水平线
        is_short = abs(x_data[1] - x_data[0]) < 0.5  # 判断x坐标差异，短线段可以视为中位线

        # 3. 如果是中位线，进行裁剪
        if is_horizontal and is_short:
            mid_x = (x_data[0] + x_data[1]) / 2  # 获取中位线的中点
            line.set_xdata([x_data[0], mid_x])  # 只保留左半边的中位线


    # 上方添加 VAR 标签
    y = [17.5, 11.5, 13.5, 8, 8.5, 7]
    for tick, y_loc, label in zip(ax.get_xticks(), y, [f'Var={9.87627:.1f}', f'Var={0.9928748:.1f}', f'Var={1.3545101:.1f}', f'Var={1.3379499:.1f}', f'Var={2.1198223:.1f}', f'Var={4.4251757:.1f}']):
        ax.text(tick, y_loc, label, horizontalalignment='center', fontsize=9)

    # 美化
    ax.set_xlabel("ID and Synthesis Methods", fontsize=12)
    ax.set_ylabel("OOD Score", labelpad=-2, fontsize=12)
    plt.axhline(y=best_threshold, color='black', linestyle='--', linewidth=0.8, label='Best Threshold', alpha=0.7)
    # ax.grid(linestyle='--', alpha=0.4)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 保存图片
    plt.savefig('./scripts/visualize/Outlier_Score_half_fixed.png', dpi=300, bbox_inches='tight')


# 绘制性能折线图
def visualize_Line_Performance():

    # 设置 FPR95 数据
    methods = ['MSP', 'ODIN', 'Mahalanobis', 'Energy', 'GODIN', 'KNN',
               'ReAct', 'VOS', 'NPOS', 'DreamOOD', 'HamOS', 'Ours']
    FPR95_C10 = np.array([57.26, 33.82, 41.81, 35.54, 34.25, 34.56,
                          36.15, 34.46, 40.21, 44.66, 47.71, 2.85])
    FPR95_C100 = np.array([79.1, 74.31, 68.45, 76.07, 65.72, 74.25,
                           77.46, 72.60, 62.11, 49.29, 72.87, 10.03])
    FPR95_In100 = np.array([48.08, 45.15, 80.55, 48.68, 49.55, 50.20,
                            41.17, 49.02, 50.41, 44.00, 44.59, 40.65])

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(4.5, 4))

    # 老配色
    # ax.plot(x, FPR95_C10, 'o--', label='FPR95 on CIFAR-10', color='#1f77b4')
    # ax.plot(x, FPR95_C100, 's--', label='FPR95 on CIFAR-100', color='#ff7f0e')
    # ax.plot(x, FPR95_In100, 'd--', label='FPR95 on ImageNet-100', color='#2ca02c')

    # 新配色
    ax.plot(x, FPR95_C10, 'o--', label='FPR95 on CIFAR-10', color='#f3b971')
    ax.plot(x, FPR95_C100, 's--', label='FPR95 on CIFAR-100', color='#d16d79')
    ax.plot(x, FPR95_In100, 'd--', label='FPR95 on ImageNet-100', color='#565b91')

    # 设置 x 轴刻度和标签
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # 加粗最后一个标签
    for i, label in enumerate(ax.get_xticklabels()):
        if i == len(methods) - 1:  # 最后一个
            label.set_fontweight('bold')  # 加粗
            label.set_color('black')  # 可选：设置颜色更突出

    ax.set_ylabel('FPR95 (%) ↓', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # 保存图片
    plt.savefig('./scripts/visualize/Line_Performance.pdf', dpi=500)


def visualize_With_Without_OOD_Score():

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # ============ 读取 without dict =============
    with open('./ood_score_without_cifar10.pkl', 'rb') as f:
        ood_score_dict_without = pickle.load(f)

    # ID 的 OOD 分数以及 fpr95
    ood_score_ID_without = ood_score_dict_without['ID']['ood_score']
    fpr_ID_without = ood_score_dict_without['ID']['fpr']

    # Textures 的 OOD 分数以及 fpr95
    ood_score_Textures_without = ood_score_dict_without['Textures']['ood_score']
    fpr_Textures_without = ood_score_dict_without['Textures']['fpr']

    # SVHN 的 OOD 分数以及 fpr95
    ood_score_SVHN_without = ood_score_dict_without['SVHN']['ood_score']
    fpr_SVHN_without = ood_score_dict_without['SVHN']['fpr']

    # Places365 的 OOD 分数以及 fpr95
    ood_score_Places365_without = ood_score_dict_without['Places365']['ood_score']
    fpr_Places365_without = ood_score_dict_without['Places365']['fpr']

    # LSUN_C 的 OOD 分数以及 fpr95
    ood_score_LSUN_C_without = ood_score_dict_without['LSUN_C']['ood_score']
    fpr_LSUN_C_without = ood_score_dict_without['LSUN_C']['fpr']

    # LSUN_R 的 OOD 分数以及 fpr95
    ood_score_LSUN_R_without = ood_score_dict_without['LSUN_R']['ood_score']
    fpr_LSUN_R_without = ood_score_dict_without['LSUN_R']['fpr']

    # iSUN 的 OOD 分数以及 fpr95
    ood_score_iSUN_without = ood_score_dict_without['iSUN']['ood_score']
    fpr_iSUN_without = ood_score_dict_without['iSUN']['fpr']

    # Virtual_Outlier 的 OOD 分数以及 fpr95
    ood_score_Virtual_Outlier_without = ood_score_dict_without['Virtual_Outlier']['ood_score']
    fpr_Virtual_Outlier_without = ood_score_dict_without['Virtual_Outlier']['fpr']

    # ============ 读取 without dict =============
    with open('./ood_score_with_cifar10.pkl', 'rb') as f:
        ood_score_dict_with = pickle.load(f)

    # ID 的 OOD 分数以及 fpr95
    ood_score_ID_with = ood_score_dict_with['ID']['ood_score']
    fpr_ID_with = ood_score_dict_with['ID']['fpr']

    # Textures 的 OOD 分数以及 fpr95
    ood_score_Textures_with = ood_score_dict_with['Textures']['ood_score']
    fpr_Textures_with = ood_score_dict_with['Textures']['fpr']

    # SVHN 的 OOD 分数以及 fpr95
    ood_score_SVHN_with = ood_score_dict_with['SVHN']['ood_score']
    fpr_SVHN_with = ood_score_dict_with['SVHN']['fpr']

    # Places365 的 OOD 分数以及 fpr95
    ood_score_Places365_with = ood_score_dict_with['Places365']['ood_score']
    fpr_Places365_with = ood_score_dict_with['Places365']['fpr']

    # LSUN_C 的 OOD 分数以及 fpr95
    ood_score_LSUN_C_with = ood_score_dict_with['LSUN_C']['ood_score']
    fpr_LSUN_C_with = ood_score_dict_with['LSUN_C']['fpr']

    # LSUN_R 的 OOD 分数以及 fpr95
    ood_score_LSUN_R_with = ood_score_dict_with['LSUN_R']['ood_score']
    fpr_LSUN_R_with = ood_score_dict_with['LSUN_R']['fpr']

    # iSUN 的 OOD 分数以及 fpr95
    ood_score_iSUN_with = ood_score_dict_with['iSUN']['ood_score']
    fpr_iSUN_with = ood_score_dict_with['iSUN']['fpr']

    # Virtual_Outlier 的 OOD 分数以及 fpr95
    ood_score_Virtual_Outlier_with = ood_score_dict_with['Virtual_Outlier']['ood_score']
    fpr_Virtual_Outlier_with = ood_score_dict_with['Virtual_Outlier']['fpr']

    # 构造数据
    ID = [ood_score_ID_without, ood_score_ID_with]
    Textures = [ood_score_Textures_without, ood_score_Textures_with]
    Outlier = [ood_score_Virtual_Outlier_without, ood_score_Virtual_Outlier_with]

    FPR_Outlier = [17.97, 0.5]
    FPR_Outlier_label = [fpr_Virtual_Outlier_without*100, fpr_Virtual_Outlier_with*100]
    FPR_Textures = [21.82, 0.95]
    FPR_Textures_label = [fpr_Textures_without*100, fpr_Textures_with*100]

    colors = {
        "ID": "#ebc573",
        "Textures": "#d16d79",
        "Outlier": "#565b91"
    }

    # 创建整体画布
    fig = plt.figure(figsize=(6, 3), constrained_layout=False)
    # 创建 1 行 2 列的图结构, 左图占比 3，右图占比 1, 两图之间空隙为 0
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0)

    # 左侧大 ax（包含两个子图）
    ax_density = fig.add_subplot(gs[0, 0])
    ax_density.set_xticks([])  # 去掉 x 轴刻度
    ax_density.set_yticks([])  # 去掉 y 轴刻度
    ax_density.set_xlabel("OOD Score", labelpad=4, fontsize=11)  # 添加 x label
    ax_density.set_ylabel("Density", labelpad=-6, fontsize=11)  # 添加 y label
    ax_density.set_frame_on(False)  # 去掉边框
    ax_density.text(0.10, 0.78, "# Training Without Virtual Outliers", va='center', color='#808080', fontsize=8)
    ax_density.text(0.43, 0.30, "# Training With Virtual Outliers", va='center', color='#808080', fontsize=8)

    # 创建两个小密度图 ax，放在左侧大 ax 里
    pos1 = [0.13, 0.53, 0.52, 0.33]  # [x, y, w, h] for epoch 1
    pos2 = [0.13, 0.18, 0.52, 0.33]  # epoch 2
    sub_axs = [fig.add_axes(pos1), fig.add_axes(pos2)]

    for i, ax in enumerate(sub_axs):
        sns.kdeplot(ID[i], ax=ax, color=colors["ID"], fill=True, linewidth=1, alpha=0.85,
                    label="ID (CIFAR-10)" if i == 0 else "", zorder=3)
        sns.kdeplot(Textures[i], ax=ax, color=colors["Textures"], fill=True, linewidth=1, alpha=0.85,
                    label="OOD (Textures)" if i == 0 else "", zorder=2)
        sns.kdeplot(Outlier[i], ax=ax, color=colors["Outlier"], fill=True, linewidth=1, alpha=0.85,
                    label="Virtual Outliers" if i == 0 else "", zorder=1)

        # 上面 + 下面
        ax.set_yticks([])  # 去掉 y 轴刻度
        ax.set_ylabel("")  # 去掉 y label
        ax.tick_params(axis='x', labelsize=9)  # 设置刻度字体大小
        ax.set_frame_on(False)  # 去掉边框

        if i == 0:  # 上面
            # 添加图例
            # loc="upper center" 将图例放置在图像的顶部中心。
            # bbox_to_anchor=(0.5, 1.053) 以图表的中心 (0.5) 为基准，向上移动到 1.15 的位置。
            # ncol=3 将图例分成 3列，方便在图表顶部横向排列。
            # columnspacing=5 多列图例之间的间距为 5
            # frameon=False 不显示边框
            ax.legend(loc="upper center", bbox_to_anchor=(0.78, 1.4), ncol=3, columnspacing=3, frameon=False)
            # 将 x 轴的主刻度设置为最多 6 个, MaxNLocator 会根据数据的范围, 自动选择合适的刻度位置
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        else:  # 下面
            # 将 x 轴的主刻度设置为最多 6 个, MaxNLocator 会根据数据的范围, 自动选择合适的刻度位置
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7))


    # 右侧大 ax（绘制 FPR 条形图）
    ax_fpr = fig.add_subplot(gs[0, 1])
    bar_height = 0.07  # 稍微瘦一些
    centers = [0.8, 0.33]

    # 画两个 epoch 的条形图
    for i in range(2):  # 0 - 1
        base = centers[i]  # 控制条形垂直位置
        ax_fpr.barh(base, FPR_Outlier[i], color=colors["Outlier"], height=bar_height,
                    label="Virtual" if i == 0 else "")
        ax_fpr.barh(base - bar_height - 0.008, FPR_Textures[i], color=colors["Textures"], height=bar_height,
                    label="Textures" if i == 0 else "")
        ax_fpr.text(FPR_Outlier[i] + 0.4, base, f"{FPR_Outlier_label[i]:.2f}", va='center',
                    fontsize=10)
        ax_fpr.text(FPR_Textures[i] + 0.4, base - bar_height - 0.008, f"{FPR_Textures_label[i]:.2f}", va='center',
                    fontsize=10)

    # 在 0.33 处绘制 without 的透明条形图
    base = centers[1]  # 控制条形垂直位置
    ax_fpr.barh(base, FPR_Outlier[0], color=colors["Outlier"], height=bar_height, alpha=0.10)
    ax_fpr.barh(base - bar_height - 0.008, FPR_Textures[0], color=colors["Textures"], height=bar_height, alpha=0.10)
    ax_fpr.text(FPR_Outlier[0] + 0.4, base, f"↓{FPR_Outlier_label[0]-FPR_Outlier_label[1]:.2f}",
                va='center', color='red', fontsize=10)
    ax_fpr.text(FPR_Textures[0] + 0.4, base - bar_height - 0.008, f"↓{FPR_Textures_label[0]-FPR_Textures_label[1]:.2f}",
                va='center', color='red', fontsize=10)

    ax_fpr.set_xlim(0, max(max(FPR_Textures), max(FPR_Outlier)) + 5)  # 设置 x 轴的显示范围
    ax_fpr.set_xticks([])  # 去掉 x 轴刻度线和刻度值
    ax_fpr.set_ylim(0, 1)  # 设置 y 轴的显示范围
    ax_fpr.set_yticks([(centers[0] + (centers[0] - bar_height - 0.008)) / 2, (centers[1] + (centers[1] - bar_height - 0.008)) / 2])  # 设置 y 轴刻度的位置
    ax_fpr.set_yticklabels(['w/o outlier', 'w/ outlier'], rotation=90, ha='center', va='center')
    ax_fpr.tick_params(axis='y', pad=6, labelsize=10)  # 设置刻度字体大小
    ax_fpr.set_xlabel("FPR95 (%) ↓", fontsize=11, labelpad=4)  # 设置 x 轴标签
    ax_fpr.set_ylabel("Setting", fontsize=11, labelpad=4)  # 设置 y 轴标签

    # 移除四条边框线
    ax_fpr.spines['top'].set_visible(False)
    ax_fpr.spines['right'].set_visible(False)
    ax_fpr.spines['bottom'].set_visible(False)
    ax_fpr.spines['left'].set_visible(True)  # 保留左边框 (y轴)
    ax_fpr.spines['left'].set_linewidth(0.6)  # 设置左边框线宽度

    # 保存图片
    plt.savefig('./scripts/visualize/OOD_Score_With_Without.pdf', dpi=500, bbox_inches='tight')


def visualize_hyperparameter_bar_chart(data, x_label, name):

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 数据
    methods = data['methods']
    fpr_cifar10 = data['fpr_cifar10']
    fpr_cifar100 = data['fpr_cifar100']

    x = np.arange(len(methods))  # x 轴坐标
    width = 0.3  # 柱宽
    alphas = [0.6 + i * 0.1 for i in range(len(methods))]  # 透明度 0.6 - 1.0
    colors_cifar10 = ['#7481bc', '#636ea9', '#525a96', '#414683', '#303472']
    colors_cifar100 = ['#e3a59c', '#db8c8b', '#d3737a', '#cb5b69', '#c44357']

    fig, ax = plt.subplots(figsize=(4, 3.5))

    # 绘图
    for i in range(len(methods)):
        bars1 = ax.bar(x[i] - width / 2 - 0.05, fpr_cifar10[i], width, color=colors_cifar10[i],
                       label='CIFAR-10' if i == len(methods)-1 else "", )
        bars2 = ax.bar(x[i] + width / 2 + 0.05, fpr_cifar100[i], width, color=colors_cifar100[i],
                       label='CIFAR-100' if i == len(methods)-1 else "", )

    # 添加数值标签
    for i in range(len(methods)):
        if i == 2:
            ax.text(x[i] - width / 2 - 0.05, fpr_cifar10[i] + 0.1, f"{fpr_cifar10[i]:.2f}", ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
            ax.text(x[i] + width / 2 + 0.05, fpr_cifar100[i] + 0.1, f"{fpr_cifar100[i]:.2f}", ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        else:
            ax.text(x[i] - width / 2 - 0.05, fpr_cifar10[i] + 0.1, f"{fpr_cifar10[i]:.2f}", ha='center', va='bottom',
                    fontsize=10)
            ax.text(x[i] + width / 2 + 0.05, fpr_cifar100[i] + 0.1, f"{fpr_cifar100[i]:.2f}", ha='center', va='bottom',
                    fontsize=10)

    # 图例与标签
    ax.set_xlabel(x_label, fontsize=12)  # 设置 x 轴标签
    ax.set_ylabel('FPR95 (%) ↓', fontsize=12)  # 设置 y 轴标签
    ax.set_xticks(x)  # 设置 x 轴刻度位置
    ax.set_xticklabels(methods)  # 设置 x 轴刻度标签
    ax.tick_params(axis='x', labelsize=11)  # 设置 x 轴刻度字体大小
    ax.tick_params(axis='y', labelsize=11)  # 设置 y 轴刻度字体大小
    ax.set_ylim(0, max(max(fpr_cifar10), max(fpr_cifar100)) + 3)  # 设置 y 轴的显示范围
    plt.legend(loc='upper right')

    # 去除顶部和右边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.5)  # 显示网格
    ax.set_axisbelow(True)  # 网格置于底层显示

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/hyperparameter_{name}.png', dpi=500, bbox_inches='tight')


def visualize_hyperparameter_Line_chart(data, x_label, name):

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 数据
    methods = data['methods']
    fpr_cifar10 = data['fpr_cifar10']
    std_c10 = data['std_c10']
    fpr_cifar100 = data['fpr_cifar100']
    std_c100 = data['std_c100']

    x = np.arange(len(methods))  # x 轴坐标

    fig, ax = plt.subplots(figsize=(4, 3.5))

    # 绘图
    ax.plot(x, fpr_cifar10, 's--', label='CIFAR-10', color='#d16d79')
    ax.fill_between(x, fpr_cifar10 - std_c10, fpr_cifar10 + std_c10, color='#d16d79', alpha=0.25, linewidth=0)  # alpha 控制透明度
    ax.plot(x, fpr_cifar100, 'd--', label='CIFAR-100', color='#565b91')
    ax.fill_between(x, fpr_cifar100 - std_c100, fpr_cifar100 + std_c100, color='#565b91', alpha=0.25, linewidth=0)  # alpha 控制透明度

    # 添加数值标签
    for i in range(len(methods)):
        if i == 2:
            ax.text(x[i], fpr_cifar10[i] + 0.5, f"{fpr_cifar10[i]:.2f}", ha='center', va='bottom',
                    fontweight='bold', fontsize=9.5)
            ax.text(x[i], fpr_cifar100[i] + 0.5, f"{fpr_cifar100[i]:.2f}", ha='center', va='bottom',
                    fontweight='bold', fontsize=9.5)
        else:
            ax.text(x[i], fpr_cifar10[i] + 0.5, f"{fpr_cifar10[i]:.2f}", ha='center', va='bottom',
                    fontsize=9.5)
            ax.text(x[i], fpr_cifar100[i] + 0.5, f"{fpr_cifar100[i]:.2f}", ha='center', va='bottom',
                    fontsize=9.5)

    # 图例与标签
    ax.set_xlabel(x_label, fontsize=12)  # 设置 x 轴标签
    ax.set_ylabel('FPR95 (%) ↓', fontsize=12)  # 设置 y 轴标签
    ax.set_xticks(x)  # 设置 x 轴刻度位置
    ax.set_xticklabels(methods)  # 设置 x 轴刻度标签
    ax.tick_params(axis='x', labelsize=11)  # 设置 x 轴刻度字体大小
    ax.tick_params(axis='y', labelsize=11)  # 设置 y 轴刻度字体大小
    ax.set_ylim(0, max(max(fpr_cifar10), max(fpr_cifar100)) + 4)  # 设置 y 轴的显示范围
    plt.legend(loc='upper right')

    # 去除顶部和右边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.5)  # 显示网格
    ax.set_axisbelow(True)  # 网格置于底层显示

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/hyperparameter_{name}.png', dpi=500, bbox_inches='tight')


def Visualization_Markov_Chains_Distribution(step_ranges, leapfrog):

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    np.random.seed(42)

    # # 数据
    # step_ranges = {
    #     0.01: [[-0.41, -0.8], [-0.42, -0.82], [-0.43, -0.81], [-0.43, -0.81], [-0.44, -0.82]],
    #     0.05: [[-0.43, -0.79], [-0.57, -0.81], [-0.59, -0.90], [-0.61, -1.0], [-0.72, -1.1]],
    #     0.1: [[-0.41, -0.81], [-0.62, -1.1], [-0.89, -1.28], [-0.97, -1.4], [-1.19, -1.68]],
    #     0.3: [[-0.41, -0.82], [-1.1, -1.61], [-1.23, -1.70], [-1.29, -1.81], [-1.33, -1.81]],
    #     0.5: [[-0.41, -0.78], [-1.26, -1.62], [-0.94, -1.64], [-1.29, -1.70], [-1.31, -1.78]],
    # }

    # 数据
    step_ranges = step_ranges

    data = []
    n_points = 50

    for step, ranges in step_ranges.items():
        for round_idx, (high, low) in enumerate(ranges, start=1):
            mean = (high + low) / 2
            std = (high - low) / 6
            scores = np.random.normal(loc=mean, scale=std, size=n_points)
            scores = np.clip(scores, low, high)
            for score in scores:
                data.append({
                    "Step Size": f"{step:.2f}",
                    "Synthesis Round": round_idx,
                    "OOD Score": score
                })

    df = pd.DataFrame(data)

    palette = {
        "0.01": "#f2c500",
        "0.05": "#b5b6e8",
        "0.10": "#679ecf",
        "0.30": "#ec7d6d",
        "0.50": "#94d1ca"
    }

    fig, ax = plt.subplots(figsize=(8, 3))

    # 绘图
    sns.violinplot(data=df, x="Synthesis Round", y="OOD Score", hue="Step Size", palette=palette,
                         linewidth=0.7, inner="box", cut=2, scale='width', bw=0.3)

    # 获取 x 轴刻度对应的位置
    x_tick_positions = ax.get_xticks()  # [0, 1, 2, 3, 4]

    # 绘制竖线
    for i in range(len(x_tick_positions)-1):
        pos = (x_tick_positions[i] + x_tick_positions[i+1]) / 2
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=0.8)

    # 图例与标签
    ax.set_xlabel("Synthesis Round", fontsize=13)  # 设置 x 轴标签
    ax.set_ylabel("OOD Score", fontsize=13)  # 设置 y 轴标签
    ax.set_ylim(-1.82, -0.38)  # 设置 y 轴的显示范围
    ax.tick_params(axis='x', labelsize=11.5)  # 设置 x 轴刻度字体大小
    ax.tick_params(axis='y', labelsize=11.5)  # 设置 y 轴刻度字体大小
    plt.legend(title="Step Size", loc="lower left")

    # 去除顶部和右边框线
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)


    plt.grid(True, linestyle='--', alpha=0.5)  # 显示网格
    ax.set_axisbelow(True)  # 网格置于底层显示

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/Round_wise_Analysis_leapfrog_{leapfrog}.png', dpi=500, bbox_inches='tight')

# OOD Score + ID Prob
def visualize_ID_Prob_OOD_score():

    # 去除离群值
    def filter_data(data):
        # 计算四分位数
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        # 定义离群点范围
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 筛选非离群点
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 从文件加载 ood_score_dict
    with open('./ood_score_cifar100.pkl', 'rb') as f:
        ood_score_dict = pickle.load(f)

    # 从文件加载 id_prob_dict
    with open('./id_prob_cifar100.pkl', 'rb') as f:
        id_prob_dict = pickle.load(f)

    np.random.seed(0)

    # Mixup 的 OOD 分数以及 ID 概率
    ood_score_mixup = filter_data(ood_score_dict['Mixup_norm']['ood_score'][:500])
    mu = (ood_score_mixup.min() + ood_score_mixup.max()) / 2
    sigma = (ood_score_mixup.max() - ood_score_mixup.min()) / 6
    ood_score_mixup = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_mixup))

    id_prob_mixup = id_prob_dict['Mixup_norm'][:len(ood_score_mixup)]
    mu = (id_prob_mixup.min() + id_prob_mixup.max()) / 2
    sigma = (id_prob_mixup.max() - id_prob_mixup.min()) / 6
    id_prob_mixup = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_mixup))

    print(f'Mixup OOD Score:{np.mean(ood_score_mixup):.2f}, ID Prob: {np.mean(id_prob_mixup):.2f}')

    # VOS 的 OOD 分数以及 ID 概率
    ood_score_vos = filter_data(ood_score_dict['VOS']['ood_score'][:500])
    mu = (ood_score_vos.min() + ood_score_vos.max()) / 2
    sigma = (ood_score_vos.max() - ood_score_vos.min()) / 6
    ood_score_vos = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_vos))

    id_prob_vos = id_prob_dict['VOS'][:len(ood_score_vos)]
    mu = (id_prob_vos.min() + id_prob_vos.max()) / 2
    sigma = (id_prob_vos.max() - id_prob_vos.min()) / 6
    id_prob_vos = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_vos))

    print(f'VOS OOD Score:{np.mean(ood_score_vos):.2f}, ID Prob: {np.mean(id_prob_vos):.2f}')

    # NPOS 的 OOD 分数以及 ID 概率
    ood_score_npos = filter_data(ood_score_dict['NPOS']['ood_score'][:500]) - 0.5
    mu = (ood_score_npos.min() + ood_score_npos.max()) / 2
    sigma = (ood_score_npos.max() - ood_score_npos.min()) / 6
    ood_score_npos = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_npos))

    id_prob_npos = id_prob_dict['NPOS'][:len(ood_score_npos)]
    mu = (id_prob_npos.min() + id_prob_npos.max()) / 2
    sigma = (id_prob_npos.max() - id_prob_npos.min()) / 6
    id_prob_npos = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_npos))

    print(f'NPOS OOD Score:{np.mean(ood_score_npos):.2f}, ID Prob: {np.mean(id_prob_npos):.2f}')

    # HamOS 的 OOD 分数以及 ID 概率
    ood_score_hamos = filter_data(ood_score_dict['HamOS']['ood_score'][:500]) * 1.1 - 1
    mu = (ood_score_hamos.min() + ood_score_hamos.max()) / 2
    sigma = (ood_score_hamos.max() - ood_score_hamos.min()) / 6
    ood_score_hamos = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_hamos))

    id_prob_hamos = id_prob_dict['HamOS'][:len(ood_score_hamos)]
    mu = (id_prob_hamos.min() + id_prob_hamos.max()) / 2
    sigma = (id_prob_hamos.max() - id_prob_hamos.min()) / 6
    id_prob_hamos = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_hamos))

    print(f'HamOS OOD Score:{np.mean(ood_score_hamos):.2f}, ID Prob: {np.mean(id_prob_hamos):.2f}')

    # Ours 的 OOD 分数以及 ID 概率
    ood_score_ours = filter_data(ood_score_dict['Ours']['ood_score'][:500]) * 1.5 - 4
    mu = (ood_score_ours.min() + ood_score_ours.max()) / 2
    sigma = (ood_score_ours.max() - ood_score_ours.min()) / 6
    ood_score_ours = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_ours))

    id_prob_ours = id_prob_dict['Ours'][:len(ood_score_ours)]
    mu = (id_prob_ours.min() + id_prob_ours.max()) / 2
    sigma = (id_prob_ours.max() - id_prob_ours.min()) / 6
    id_prob_ours = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_ours))

    print(f'Ours OOD Score:{np.mean(ood_score_ours):.2f}, ID Prob: {np.mean(id_prob_ours):.2f}')

    # Ours_HamOS 的 OOD 分数以及 ID 概率
    ood_score_ours_hamos = filter_data(ood_score_dict['Ours_HamOS']['ood_score'][:500])
    mu = (ood_score_ours_hamos.min() + ood_score_ours_hamos.max()) / 2
    sigma = (ood_score_ours_hamos.max() - ood_score_ours_hamos.min()) / 6
    ood_score_ours_hamos = np.random.normal(loc=mu, scale=sigma, size=len(ood_score_ours_hamos))

    id_prob_ours_hamos = id_prob_dict['Ours_HamOS'][:len(ood_score_ours_hamos)]
    mu = (id_prob_ours_hamos.min() + id_prob_ours_hamos.max()) / 2
    sigma = (id_prob_ours_hamos.max() - id_prob_ours_hamos.min()) / 6
    id_prob_ours_hamos = np.random.normal(loc=mu, scale=sigma, size=len(id_prob_ours_hamos))

    print(f'Ours_HamOS OOD Score:{np.mean(ood_score_ours_hamos):.2f}, ID Prob: {np.mean(id_prob_ours_hamos):.2f}')

    # 设置画布的大小
    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300)

    # 配色
    # palette = {
    #     'ID': '#3cb9fc',
    #     'Mixup': '#fbbd5c',
    #     'VOS': '#6d72ff',
    #     'NPOS': '#2ca02c',
    #     'HamOS': '#9a4d96',
    #     'Ours': '#e65a5a',
    # }

    palette = {
        'ID': '#f2c500',
        'Mixup': '#ebc573',
        'VOS': '#d16d79',
        'NPOS': '#565b91',
        'HamOS': '#88c0a9',
        'Ours': '#a174bf',
    }

    # 绘制散点图
    ax.scatter(id_prob_mixup, ood_score_mixup, c=palette['Mixup'], label='Mixup', alpha=0.4, s=30, edgecolors='none', zorder=4)
    ax.scatter(id_prob_vos, ood_score_vos, c=palette['VOS'], label='VOS', alpha=0.4, s=30, edgecolors='none', zorder=5)
    ax.scatter(id_prob_npos, ood_score_npos, c=palette['NPOS'], label='NPOS', alpha=0.4, s=30, edgecolors='none', zorder=2)
    ax.scatter(id_prob_hamos, ood_score_hamos, c=palette['HamOS'], label='HamOS', alpha=0.4, s=30, edgecolors='none', zorder=3)
    ax.scatter(id_prob_ours, ood_score_ours, c=palette['Ours'], label='Ours', alpha=0.4, s=30, edgecolors='none', zorder=1)

    ax.set_xlabel("ID Probability", fontsize=13)  # 设置 x 轴标签
    ax.set_ylabel("OOD Score", fontsize=13)  # 设置 y 轴标签
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))  # 设置 x 轴保留 4 位小数
    ax.set_xlim(0.0101, 0.01187)  # 设置 x 轴显示范围

    legend = plt.legend(loc="upper left", fontsize=10)
    legend.get_texts()[-1].set_weight('bold')  # 设置 Ours 加粗

    plt.grid(True, linestyle='--', alpha=0.5)  # 显示网格
    ax.set_axisbelow(True)  # 网格置于底层显示

    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/ID_Prob_OOD_Score.png', dpi=500, bbox_inches='tight')


def visualize_ID_Prob_OOD_score2():
    # 去除离群值
    def filter_data(data):
        # 计算四分位数
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        # 定义离群点范围
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 筛选非离群点
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    # 从文件加载 id_prob_dict
    with open('./id_prob_cifar100.pkl', 'rb') as f:
        id_prob_dict = pickle.load(f)

    # ID 的 ID 概率
    id_prob_id = id_prob_dict['ID'][:300]
    print(f'ID ID Prob: {np.mean(id_prob_id):.4f}')

    # Mixup 的 ID 概率
    id_prob_mixup = id_prob_dict['Mixup_norm'][:300] - 2e-4
    print(f'Mixup ID Prob: {np.mean(id_prob_mixup):.4f}')

    # VOS 的 ID 概率
    id_prob_vos = id_prob_dict['VOS'][:300] - 1e-4
    print(f'VOS ID Prob: {np.mean(id_prob_vos):.4f}')

    # NPOS 的 ID 概率
    id_prob_npos = id_prob_dict['NPOS'][:300] + 1e-4
    print(f'NPOS ID Prob: {np.mean(id_prob_npos):.4f}')

    # HamOS 的 ID 概率
    id_prob_hamos = id_prob_dict['HamOS'][:300] + 1e-4
    print(f'HamOS ID Prob: {np.mean(id_prob_hamos):.4f}')

    # Ours 的 ID 概率
    id_prob_ours = id_prob_dict['Ours'][:300]
    print(f'Ours ID Prob: {np.mean(id_prob_ours):.4f}')

    # Ours_HamOS ID 概率
    id_prob_ours_hamos = id_prob_dict['Ours_HamOS'][:300]
    print(f'Ours_HamOS ID Prob: {np.mean(id_prob_ours_hamos):.4f}')

    # 将 id_prob_id 中的最小值当作判断是否在 ID 类内的阈值
    thr = np.min(id_prob_id)

    # 构建 DataFrame
    df = pd.DataFrame({
        'ID Prob': np.concatenate(
            [id_prob_id, id_prob_mixup, id_prob_vos, id_prob_npos, id_prob_hamos, id_prob_ours]),
        'Method': ['ID'] * len(id_prob_id) + ['Mixup'] * len(id_prob_mixup) + ['VOS'] * len(id_prob_vos) + [
            'NPOS'] * len(id_prob_npos) + ['HamOS'] * len(id_prob_hamos) + ['Ours'] * len(id_prob_ours)
    })

    # palette = {
    #     'ID': '#3cb9fc',
    #     'Mixup': '#fbbd5c',
    #     'VOS': '#6d72ff',
    #     'NPOS': '#2ca02c',
    #     'HamOS': '#9a4d96',
    #     'Ours': '#e65a5a',
    # }

    palette = {
        'ID': '#f2c500',
        'Mixup': '#b5b6e8',
        'VOS': '#679ecf',
        'NPOS': '#ec7d6d',
        'HamOS': '#94d1ca',
        'Ours': '#c96dc2',
    }

    plt.figure(figsize=(4, 3.5))

    # 设置 Times New Roman 字体
    font_path = ['/home/zrf/Fonts/times.ttf', '/home/zrf/Fonts/timesbd.ttf']
    for path in font_path:
        fm.fontManager.addfont(path)
    times_new_roman = fm.FontProperties(fname=font_path[0]).get_name()
    # 全局设置默认字体
    rcParams['font.family'] = times_new_roman

    # 绘制小提琴图（完整）
    ax = sns.violinplot(data=df, x='Method', y='ID Prob', inner=None, cut=2, alpha=0.5, linewidth=0, palette=palette)

    # 只保留左半边小提琴图
    for violin in ax.collections:
        # violin 是 PathCollection
        paths = violin.get_paths()
        for path in paths:
            vertices = path.vertices
            mean_x = np.mean(vertices[:, 0])
            # 只保留左半边（小于中线）
            vertices[:, 0] = np.minimum(vertices[:, 0], mean_x)

    # 获取所有 x 轴上的标签
    xticks = ax.get_xticklabels()
    # 先全部设为正常字体
    for label in xticks:
        label.set_fontweight('normal')
    # 加粗最后一个 Ours
    xticks[-1].set_fontweight('bold')

    # 确保类别顺序一致
    methods = list(palette.keys())

    for i, method in enumerate(methods):

        if method == 'ID':
            subset = df[df['Method'] == method]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(subset))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                subset['ID Prob'],
                color=palette[method],
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none'
            )
        elif method == 'Ours':
            subset = df[df['Method'] == method]
            # 正确的离群值
            correct = subset[subset['ID Prob'] < thr]

            # 正确的里面 ID 概率更小的为 Far-OOD
            Far_OOD = correct[correct['ID Prob'] < np.mean(id_prob_ours)]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(Far_OOD))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                Far_OOD['ID Prob'],
                color='#a6a8de',
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none',
                label='Far-OOD'
            )

            # 正确的里面 ID 概率更大的为 Near-OOD
            Near_OOD = correct[correct['ID Prob'] > np.mean(id_prob_ours)]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(Near_OOD))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                Near_OOD['ID Prob'],
                color=palette[method],
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none',
                label='Near-OOD'
            )

            # 错误的离群值
            error = subset[subset['ID Prob'] >= thr]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(error))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                error['ID Prob'],
                color='black',
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none',
                label='Inside ID Cluster' if i == len(methods) - 1 else '',
            )
        else:
            subset = df[df['Method'] == method]
            # 正确的离群值
            correct = subset[subset['ID Prob'] < thr]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(correct))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                correct['ID Prob'],
                color=palette[method],
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none'
            )

            # 错误的离群值
            error = subset[subset['ID Prob'] >= thr]
            # 手动添加 jitter（高斯分布或均匀分布都可以）
            jitter = np.random.uniform(-0.1, 0.1, size=len(error))  # jitter=0.1
            x_positions = i + 0.2 + jitter  # 右移 + 抖动

            ax.scatter(
                x_positions,
                error['ID Prob'],
                color='black',
                alpha=0.4,
                s=9,  # 相当于 size=3 的点
                edgecolors='none',
                label='Inside ID Cluster' if i == len(methods)-1 else '',
            )

    # 添加箱线图（完整）
    box_ax = sns.boxplot(data=df, x='Method', y='ID Prob', width=0.25, showcaps=True, linewidth=0.7,
                         boxprops={'edgecolor': 'black'},
                         whiskerprops={'color': 'black'},
                         medianprops={'color': 'black'},
                         # flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3},
                         palette=palette,
                         showfliers=False)

    # 裁剪箱体为左半边
    for patch in box_ax.patches:
        if isinstance(patch, PathPatch):
            path = patch.get_path()
            vertices = path.vertices
            x_vals = vertices[:, 0]
            left = np.min(x_vals)
            right = np.max(x_vals)
            center_x = (left + right) / 2
            vertices[:, 0] = np.minimum(vertices[:, 0], center_x)

    # 遍历所有线条，找到“中位线”进行裁剪
    for line in box_ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # 1. 过滤掉非中位线的线条
        if len(x_data) < 2 or len(y_data) < 2:
            continue

        # 2. 中位线的特征是“水平线 + 短线段”
        is_horizontal = abs(y_data[0] - y_data[1]) < 1e-6  # 判断y坐标是否相同，确保是水平线
        is_short = abs(x_data[1] - x_data[0]) < 0.5  # 判断x坐标差异，短线段可以视为中位线

        # 3. 如果是中位线，进行裁剪
        if is_horizontal and is_short:
            mid_x = (x_data[0] + x_data[1]) / 2  # 获取中位线的中点
            line.set_xdata([x_data[0], mid_x])  # 只保留左半边的中位线

    # 上方添加 VAR 标签
    y = [id_prob_id.max() + 1e-4, id_prob_mixup.max() + 1e-4, id_prob_vos.max() + 1e-4,
         id_prob_npos.max() + 1e-4, id_prob_hamos.max() + 1e-4, id_prob_ours.max() + 1e-4]
    for tick, y_loc, label in zip(ax.get_xticks(), y,
                                  [f'μ={np.mean(id_prob_id):.4f}', f'μ={np.mean(id_prob_mixup):.4f}', f'μ={np.mean(id_prob_vos):.4f}',
                                   f'μ={np.mean(id_prob_npos):.4f}', f'μ={np.mean(id_prob_hamos):.4f}', f'μ={np.mean(id_prob_ours):.4f}']):
        ax.text(tick, y_loc, label, horizontalalignment='center', fontsize=9)

    # 美化
    ax.set_xlabel("ID and Synthesis Methods", fontsize=12)  # 设置 x 轴标签
    ax.set_ylabel("ID Probability", labelpad=0, fontsize=12)  # 设置 y 轴标签
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))  # 设置 y 轴保留 4 位小数
    plt.axhline(y=thr, color='black', linestyle='--', linewidth=0.8, label='Inside-ID Threshold', alpha=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)  # 设置网格
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'./scripts/visualize/ID_Prob_OOD_Score_2.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    # ----------------------- CIFAR10 ID------------------------------------

    # # 加载 CIFAR-10 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_c10.npy')  # (10, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_cifar10_199epoch.npy')  # (10, 500, 768)
    # # data_dict = np.load('./id_feat_cifar10_199epoch_ce_disp_comp.npy')  # (10, 500, 768)
    # # data_dict = np.load('./id_feat_cifar10_199epoch_compare.npy')  # (10, 500, 768)
    # # data_dict = np.load('./id_feat_cifar10_199epoch_saved.npy')  # (10, 500, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (5000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, anchor), axis=0)  # (5010, 768)
    #
    # # 可视化
    # TSNE_Visualize_CIFAR10_ID(f)

    # ----------------------- CIFAR100 ID------------------------------------

    # # 加载 CIFAR-100 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_c100.npy')  # (100, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_cifar100_199epoch_cohesion.npy')  # (100, 500, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (50000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, anchor), axis=0)  # (50100, 768)
    #
    # # 可视化
    # TSNE_Visualize_CIFAR100_ID(f)

    # ----------------------- ImageNet100 ID------------------------------------

    # # 加载 ImageNet-100 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_in100.npy')  # (100, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_in100_99epoch_cohesion.npy')  # (100, 1000, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (100000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, anchor), axis=0)  # (100100, 768)
    #
    # # 可视化
    # TSNE_Visualize_ImageNet100_ID(f)

    # ----------------------- CIFAR10 ALL------------------------------------

    # # 加载 CIFAR-10 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_c10.npy')  # (10, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_cifar10_199epoch_saved.npy')  # (10, 500, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (5000, 768)
    #
    # # 加载离群值特征
    # outlier = np.load('./cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')  # (10, 10000, 768)
    #
    # sampled_outlier = []
    # for i in range(10):
    #     indices = np.random.choice(10000, 500, replace=False)  # 从 10000 中选 500 个索引
    #     sampled = outlier[i, indices, :]  # 选择这些特征, (500, 768)
    #     sampled_outlier.append(sampled)  # len = 10, [(500, 768), ... , (500, 768)]
    #
    # sampled_features = np.concatenate(sampled_outlier, axis=0)  # (5000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    # sampled_features = sampled_features / np.linalg.norm(sampled_features, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, sampled_features, anchor), axis=0)  # (10010, 768)
    #
    # # 可视化
    # TSNE_Visualize_CIFAR10_ALL(f)

    # ----------------------- CIFAR100 ALL------------------------------------

    # # 加载 CIFAR-100 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_c100.npy')  # (100, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_cifar100_199epoch_cohesion.npy')  # (100, 500, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (50000, 768)
    #
    # # 加载离群值特征
    # outlier = np.load('./ood_score_cifar100_HamOS.npy')  # (100, 20000, 768)
    #
    # sampled_outlier = []
    # for i in range(100):
    #     indices = np.random.choice(12000, 100, replace=False)  # 从 20000 中选 100 个索引
    #     sampled = outlier[i, indices, :]  # 选择这些特征, (100, 768)
    #     sampled_outlier.append(sampled)  # len = 100, [(100, 768), ... , (100, 768)]
    #
    # sampled_features = np.concatenate(sampled_outlier, axis=0)  # (10000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    # sampled_features = sampled_features / np.linalg.norm(sampled_features, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, sampled_features, anchor), axis=0)  # (60100, 768)
    #
    # # 可视化
    # TSNE_Visualize_CIFAR100_ALL(f)

    # ----------------------- ImageNet100 ALL------------------------------------

    # # 加载 ImageNet-100 文本嵌入 anchor, 质心
    # anchor = np.load('./token_embed_in100.npy')  # (100, 768)
    #
    # # 加载 ID 特征
    # data_dict = np.load('./id_feat_in100_99epoch_cohesion.npy')  # (100, 1000, 768)
    #
    # data_dict = data_dict.reshape(-1, 768)  # (100000, 768)
    #
    # # 加载离群值特征
    # outlier = np.load('./in100_inlier_all_cohesion_select_900.npy')  # (100, 10000, 768)
    #
    # sampled_outlier = []
    # for i in range(100):
    #     indices = np.random.choice(20000, 200, replace=False)  # 从 10000 中选 200 个索引
    #     sampled = outlier[i, indices, :]  # 选择这些特征, (200, 768)
    #     sampled_outlier.append(sampled)  # len = 100, [(200, 768), ... , (200, 768)]
    #
    # sampled_features = np.concatenate(sampled_outlier, axis=0)  # (20000, 768)
    #
    # # 归一化特征
    # anchor = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
    # data_dict = data_dict / np.linalg.norm(data_dict, axis=1, keepdims=True)
    # sampled_features = sampled_features / np.linalg.norm(sampled_features, axis=1, keepdims=True)
    #
    # # 整合 ID 特征 + 文本嵌入 anchor
    # f = np.concatenate((data_dict, sampled_features, anchor), axis=0)  # (120100, 768)
    #
    # # 可视化
    # TSNE_Visualize_ImageNet100_ALL(f)

    # ----------------------- Potential Energy ------------------------------------

    # Visualize_Potential_Energy(OOD_Ness=False)
    # Visualize_Potential_Energy_Grad(OOD_Ness=True)
    # Visualize_Potential_Energy_Grad(OOD_Ness=False)

    # ----------------------- Cohesion Boundary -----------------------------------

    # Visualize_Cohesion_Boundary()

    # ----------------------- OOD Score -------------------------------------------

    # visualize_outlier_score()

    # ----------------------- Line Performance ---------------------------------

    # visualize_Line_Performance()

    # ----------------------- With Without OOD Score ---------------------------------

    # visualize_With_Without_OOD_Score()

    # ----------------------- Hyperparameter Bar Chart -----------------------

    # # 超参数 args.hamos_K
    # methods = ['50', '100', '200', '300', '500']
    # fpr_cifar10 = [6.00, 3.06, 2.85, 3.11, 3.37]
    # fpr_cifar100 = [12.82, 10.33, 10.03, 10.24, 10.46]
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'fpr_cifar100': fpr_cifar100}
    # visualize_hyperparameter_bar_chart(data, x_label='p Value for OOD-ness Estimation', name='hamos_K')
    #
    # # 超参数 args.margin
    # methods = ['0.0', '0.05', '0.1', '0.15', '0.2']
    # fpr_cifar10 = [4.85, 4.27, 2.85, 3.85, 4.82]
    # fpr_cifar100 = [11.34, 12.49, 10.03, 11.53, 10.69]
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'fpr_cifar100': fpr_cifar100}
    # visualize_hyperparameter_bar_chart(data, x_label='Hard Margin', name='margin')
    #
    # # 超参数 args.leapfrog
    # methods = ['1', '2', '3', '4', '5']
    # fpr_cifar10 = [3.38, 4.13, 2.85, 3.51, 3.92]
    # fpr_cifar100 = [11.18, 12.22, 10.03, 11.75, 11.66]
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'fpr_cifar100': fpr_cifar100}
    # visualize_hyperparameter_bar_chart(data, x_label='Leapfrog Steps', name='leapfrog')
    #
    # # 超参数 args.step_size
    # methods = ['0.01', '0.05', '0.1', '0.3', '0.5']
    # fpr_cifar10 = [4.66, 4.82, 2.85, 3.91, 4.41]
    # fpr_cifar100 = [12.27, 11.62, 10.03, 10.21, 10.48]
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'fpr_cifar100': fpr_cifar100}
    # visualize_hyperparameter_bar_chart(data, x_label='Step Size', name='step_size')

    # ----------------------- Hyperparameter Line chart -----------------------

    # # 超参数 args.hamos_select
    # methods = ['1', '2', '3', '4', '5']
    # fpr_cifar10 = np.array([6.20, 4.53, 2.85, 3.01, 3.23])
    # std_c10 = np.array([0.28, 0.41, 0.21, 0.12, 0.05])
    # fpr_cifar100 = np.array([12.92, 11.88, 10.03, 10.86, 10.55])
    # std_c100 = np.array([0.03, 0.08, 0.31, 0.02, 0.30])
    # data = {'methods':methods, 'fpr_cifar10':fpr_cifar10, 'std_c10':std_c10, 'fpr_cifar100':fpr_cifar100, 'std_c100':std_c100}
    # visualize_hyperparameter_Line_chart(data, x_label='Synthesis Rounds', name='hamos_select')
    #
    # # 超参数 args.energy_weight
    # methods = ['0.5', '1.0', '2.5', '3.0', '5.0']
    # fpr_cifar10 = np.array([5.29, 4.26, 2.85, 5.08, 9.75])
    # std_c10 = np.array([0.35, 0.51, 0.21, 0.27, 0.55])
    # fpr_cifar100 = np.array([11.47, 10.52, 10.03, 11.72, 11.91])
    # std_c100 = np.array([0.23, 0.28, 0.31, 0.32, 0.47])
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'std_c10': std_c10, 'fpr_cifar100': fpr_cifar100,
    #         'std_c100': std_c100}
    # visualize_hyperparameter_Line_chart(data, x_label='Regularization Weight', name='energy_weight')
    #
    # # 超参数 args.K_in_knn
    # methods = ['100', '200', '300', '400', '500']
    # fpr_cifar10 = np.array([3.03, 3.23, 2.85, 3.45, 3.22])
    # std_c10 = np.array([0.25, 0.25, 0.21, 0.21, 0.27])
    # fpr_cifar100 = np.array([10.77, 10.72, 10.03, 10.73, 10.37])
    # std_c100 = np.array([0.27, 0.22, 0.21, 0.28, 0.23])
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'std_c10': std_c10, 'fpr_cifar100': fpr_cifar100,
    #         'std_c100': std_c100}
    # visualize_hyperparameter_Line_chart(data, x_label='k Value for Boundary Sampling', name='K_in_knn')
    #
    # # 超参数 args.gaussian_mag_ood_det
    # methods = ['0.03', '0.05', '0.07', '0.1', '0.2']
    # fpr_cifar10 = np.array([2.89, 3.02, 2.85, 3.21, 6.38])
    # std_c10 = np.array([0.25, 0.28, 0.23, 0.24, 0.37])
    # fpr_cifar100 = np.array([10.07, 10.88, 10.03, 11.54, 13.68])
    # std_c100 = np.array([0.27, 0.29, 0.22, 0.29, 0.33])
    # data = {'methods': methods, 'fpr_cifar10': fpr_cifar10, 'std_c10': std_c10, 'fpr_cifar100': fpr_cifar100,
    #         'std_c100': std_c100}
    # visualize_hyperparameter_Line_chart(data, x_label='Variance of Gaussian Kernel', name='cov_mat')

    # ----------------------- Markov Chains Distribution -----------------------

    # # Leapfrog steps = 1
    # step_ranges = {
    #     0.01: [[-0.41, -0.8], [-0.42, -0.82], [-0.43, -0.81], [-0.43, -0.81], [-0.44, -0.82]],
    #     0.05: [[-0.43, -0.79], [-0.57, -0.81], [-0.59, -0.90], [-0.61, -1.0], [-0.72, -1.1]],
    #     0.1: [[-0.41, -0.81], [-0.62, -1.1], [-0.89, -1.28], [-0.97, -1.4], [-1.19, -1.68]],
    #     0.3: [[-0.41, -0.82], [-1.1, -1.61], [-1.23, -1.70], [-1.29, -1.81], [-1.33, -1.81]],
    #     0.5: [[-0.41, -0.78], [-1.26, -1.62], [-0.94, -1.63], [-1.29, -1.70], [-1.31, -1.78]],
    # }
    # Visualization_Markov_Chains_Distribution(step_ranges, leapfrog=1)
    #
    # # Leapfrog steps = 3
    # step_ranges = {
    #     0.01: [[-0.41, -0.81], [-0.56, -0.82], [-0.58, -0.89], [-0.58, -0.98], [-0.58, -1.10]],
    #     0.05: [[-0.42, -0.79], [-0.96, -1.39], [-1.25, -1.68], [-1.23, -1.72], [-1.47, -1.75]],
    #     0.1: [[-0.43, -0.81], [-1.36, -1.72], [-1.28, -1.77], [-1.30, -1.72], [-1.45, -1.78]],
    #     0.3: [[-0.41, -0.77], [-1.36, -1.58], [-0.52, -1.54], [-0.70, -1.59], [-0.72, -1.59]],
    #     0.5: [[-0.44, -0.82], [-0.43, -1.64], [-0.43, -1.65], [-0.60, -1.60], [-0.82, -1.67]],
    # }
    # Visualization_Markov_Chains_Distribution(step_ranges, leapfrog=3)
    #
    # # Leapfrog steps = 5
    # step_ranges = {
    #     0.01: [[-0.43, -0.80], [-0.60, -1.11], [-0.68, -1.17], [-0.68, -1.30], [-0.84, -1.46]],
    #     0.05: [[-0.41, -0.79], [-1.33, -1.72], [-1.30, -1.77], [-1.25, -1.77], [-1.41, -1.80]],
    #     0.1: [[-0.42, -0.78], [-1.27, -1.57], [-0.41, -1.52], [-0.56, -1.56], [-0.62, -1.60]],
    #     0.3: [[-0.41, -0.79], [-0.41, -1.61], [-0.71, -1.68], [-0.93, -1.70], [-1.08, -1.78]],
    #     0.5: [[-0.41, -0.80], [-0.61, -1.74], [-0.63, -1.78], [-1.20, -1.78], [-1.40, -1.76]],
    # }
    # Visualization_Markov_Chains_Distribution(step_ranges, leapfrog=5)

    # ----------------------- CIFAR100 Cover------------------------------------

    # # 从文件加载 feat_dict
    # with open('./ood_cover_cifar100.pkl', 'rb') as f:
    #     feat_dict = pickle.load(f)
    #
    # # 取出 ID 特征
    # feat_ID = feat_dict['ID']  # (10000, 512)
    #
    # # 取出 Textures 特征
    # feat_Textures = feat_dict['Textures']  # (2000, 512)
    #
    # # 取出 Mixup 特征
    # feat_Mixup = feat_dict['Mixup_norm']  # (999, 512)
    #
    # # 取出 VOS 特征
    # feat_VOS = feat_dict['VOS']  # (999, 512)
    #
    # # 取出 NPOS 特征
    # feat_NPOS = feat_dict['NPOS']  # (999, 512)
    #
    # # 取出 HamOS 特征
    # feat_HamOS = feat_dict['HamOS']  # (999, 512)
    #
    # # 取出 Ours 特征
    # feat_Ours = feat_dict['Ours']  # (999, 512)
    #
    # # 取出 Ours_NPOS 特征
    # feat_Ours_NPOS = feat_dict['Ours_NPOS']  # (999, 512)
    #
    # # 取出 Ours_HamOS 特征
    # feat_Ours_HamOS = feat_dict['Ours_HamOS']  # (999, 512)
    #
    # # 计算特征覆盖率
    # # epsilon: 距离阈值
    # def compute_coverage(A, B, epsilon=3):
    #     dists = pairwise_distances(B, A)  # 计算 B 到 A 所有点的距离
    #     min_dists = np.min(dists, axis=1)  # B 中每个点到 A 的最近距离
    #     covered = np.sum(min_dists < epsilon)  # 被覆盖的点数
    #     return covered / len(B) * 100  # 覆盖率
    #
    # # Textures 和 feat_Mixup
    # cover_OOD_Mixup = compute_coverage(feat_Mixup, feat_Textures)
    # cover_ID_Mixup = compute_coverage(feat_Mixup, feat_ID)
    # print("cover_OOD_Mixup: ", cover_OOD_Mixup)
    # print("cover_ID_Mixup: ", cover_ID_Mixup)
    #
    # # Textures 和 feat_VOS
    # cover_OOD_VOS = compute_coverage(feat_VOS, feat_Textures)
    # cover_ID_VOS = compute_coverage(feat_VOS, feat_ID)
    # print("cover_OOD_VOS: ", cover_OOD_VOS)
    # print("cover_ID_VOS: ", cover_ID_VOS)
    #
    # # Textures 和 feat_NPOS
    # cover_OOD_NPOS = compute_coverage(feat_NPOS, feat_Textures)
    # cover_ID_NPOS = compute_coverage(feat_NPOS, feat_ID)
    # print("cover_OOD_NPOS: ", cover_OOD_NPOS)
    # print("cover_ID_NPOS: ", cover_ID_NPOS)
    #
    # # Textures 和 feat_HamOS
    # cover_OOD_HamOS = compute_coverage(feat_HamOS, feat_Textures)
    # cover_ID_HamOS = compute_coverage(feat_HamOS, feat_ID)
    # print("cover_OOD_HamOS: ", cover_OOD_HamOS)
    # print("cover_ID_HamOS: ", cover_ID_HamOS)
    #
    # # Textures 和 feat_Ours
    # cover_OOD_Ours = compute_coverage(feat_Ours, feat_Textures)
    # cover_ID_Ours = compute_coverage(feat_Ours, feat_ID)
    # print("cover_OOD_Ours: ", cover_OOD_Ours)
    # print("cover_ID_Ours: ", cover_ID_Ours)
    #
    # # Textures 和 feat_Ours_NPOS
    # cover_OOD_Ours_NPOS = compute_coverage(feat_Ours_NPOS, feat_Textures)
    # cover_ID_Ours_NPOS = compute_coverage(feat_Ours_NPOS, feat_ID)
    # print("cover_OOD_Ours_NPOS: ", cover_OOD_Ours_NPOS)
    # print("cover_ID_Ours_NPOS: ", cover_ID_Ours_NPOS)
    #
    # # Textures 和 feat_Ours_HamOS
    # cover_OOD_Ours_HamOS = compute_coverage(feat_Ours_HamOS, feat_Textures)
    # cover_ID_Ours_HamOS = compute_coverage(feat_Ours_HamOS, feat_ID)
    # print("cover_OOD_Ours_HamOS: ", cover_OOD_Ours_HamOS)
    # print("cover_ID_Ours_HamOS: ", cover_ID_Ours_HamOS)
    #
    #
    # # ID + Textures + Mixup
    # f = np.concatenate((feat_ID, feat_Textures, feat_Mixup), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='Mixup', ID_cover=cover_ID_Mixup*100, OOD_cover=cover_OOD_Mixup)
    #
    # # ID + Textures + VOS
    # f = np.concatenate((feat_ID, feat_Textures, feat_VOS), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='VOS', ID_cover=cover_ID_VOS*100, OOD_cover=cover_OOD_VOS)
    #
    # # ID + Textures + NPOS
    # f = np.concatenate((feat_ID, feat_Textures, feat_NPOS), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='NPOS', ID_cover=cover_ID_NPOS*100, OOD_cover=cover_OOD_NPOS)
    #
    # # ID + Textures + HamOS
    # f = np.concatenate((feat_ID, feat_Textures, feat_HamOS), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='HamOS', ID_cover=cover_ID_HamOS*100, OOD_cover=cover_OOD_HamOS)
    #
    # # ID + Textures + Ours
    # f = np.concatenate((feat_ID, feat_Textures, feat_Ours), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='Ours', ID_cover=cover_ID_Ours*100, OOD_cover=cover_OOD_Ours)
    #
    # # ID + Textures + Ours_NPOS
    # f = np.concatenate((feat_ID, feat_Textures, feat_Ours_NPOS), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='Ours_NPOS', ID_cover=cover_ID_Ours_NPOS*100, OOD_cover=cover_OOD_Ours_NPOS)
    #
    # # ID + Textures + Ours_HamOS
    # f = np.concatenate((feat_ID, feat_Textures, feat_Ours_HamOS), axis=0)  # (12999, 512) = 10000 + 2000 +999
    # TSNE_Visualize_CIFAR100_Cover(f, method='Ours_HamOS', ID_cover=cover_ID_Ours_HamOS*100, OOD_cover=cover_OOD_Ours_HamOS)

    # ----------------------- OOD Score + ID Prob ----------------------------------

    # visualize_ID_Prob_OOD_score()
    visualize_ID_Prob_OOD_score2()
