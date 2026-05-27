import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s","--seed", type=int, required=True)
ap.add_argument("-t","--type", type=int, required=True)
ap.add_argument("-e", "--epochs", type=int, default=500)

args = ap.parse_args()

seed = args.seed  # 固定种子
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# 保证 cuDNN 行为确定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===== 选择现在要用哪种 kernel 做初始化 =====
type_ = ['Random', 'ClassMean', 'PCA', 'KMeans', 'Gabor']
kernel_init_type = type_[args.type]

# ========== 初始化 kernel 可视化相关参数 ==========
visualize_init_kernels = True      # 是否生成三种初始化 kernel 的可视化图片

pca_components_per_class = 1       # 每个类别取多少个 PCA 主成分（1 个刚好 10 个 filter）
kmeans_global_k = 10               # k-means 的聚类个数（建议 = conv1 输出通道数）

gabor_num_orientations = 5         # Gabor 方向数
gabor_num_scales = 2               # 每个方向的尺度数，5*2 = 10 个 kernel
# 注意：gabor_num_orientations * gabor_num_scales 最好等于 conv1 的 out_channels (=10)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_random_init = False
epoch_num = args.epochs
path_name = f'output_img/seed={seed}/MNIST_2layer/{kernel_init_type}/'
import os
os.makedirs(path_name, exist_ok=True)

loss_list = []
test_acc_list = []

# functions to show an image
target_size = 20
transform = transforms.Compose(
    [torchvision.transforms.Resize(target_size),
     transforms.ToTensor(),
     # torchvision.transforms.Lambda(
     #     lambda samples: (samples - 0.5) * 2),
     torchvision.transforms.Lambda(
         lambda samples: samples.view(1, target_size, target_size))])

batch_size = 64

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('Digit 0', 'Digit 1', 'Digit 2', 'Digit 3',
           'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path_name + 'image_grid.pdf')


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
conv1_k = 10
conv1_s = 5
conv2_k = 3
cord_cnt = [0, 1, 2]


# ============================================================
# 三种初始化方式 + 通用可视化函数
# ============================================================
def align_templates_to_class_means(kmeans_centers, class_means):
    """
    输入:
        kmeans_centers: [K, D] (无序的 K-Means 中心)
        class_means:    [K, D] (有序的 ClassMean，即 Ground Truth)
    输出:
        ordered_centers: [K, D] (重新排序后的 K-Means 中心，使其第 i 行对应第 i 类)
    """
    K = kmeans_centers.shape[0]
    # 1. 计算距离矩阵 [K, K]
    dists = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # 欧氏距离
            dists[i, j] = np.linalg.norm(kmeans_centers[i] - class_means[j])

    # 2. 贪心匹配：每次找全局最小距离，锁定配对
    ordered_centers = np.zeros_like(kmeans_centers)
    matched_kmeans_idx = set()
    matched_class_idx = set()
    
    # 映射表：class_idx -> kmeans_idx
    mapping = {} 
    
    for _ in range(K):
        min_val = np.inf
        best_k = -1 # K-Means 的索引
        best_c = -1 # Class 的索引
        
        # 遍历所有未匹配的组合
        for k in range(K):
            if k in matched_kmeans_idx: continue
            for c in range(K):
                if c in matched_class_idx: continue
                if dists[k, c] < min_val:
                    min_val = dists[k, c]
                    best_k = k
                    best_c = c
        
        # 锁定这一对
        matched_kmeans_idx.add(best_k)
        matched_class_idx.add(best_c)
        mapping[best_c] = best_k

    # 3. 根据映射重组
    for c in range(K):
        k_idx = mapping[c]
        ordered_centers[c] = kmeans_centers[k_idx]
        
    return ordered_centers

def collect_patches_by_class_and_pos(trainloader, num_classes, patch_size, stride, grid_k=3):
    """
    收集每个(类c, 网格位置x,y)的 10x10 patch，返回 dict[(c,x,y)] = np.array [N, D]
    其中 D = patch_size*patch_size
    """
    D = patch_size * patch_size
    buckets = {(c,x,y): [] for c in range(num_classes)
                          for x in range(grid_k) for y in range(grid_k)}
    cord_list = [i*stride for i in range(grid_k)]  # [0,5,10]
    for imgs, labels in trainloader:
        # imgs: [B,1,20,20]
        B = imgs.size(0)
        for i in range(B):
            img = imgs[i,0]  # [20,20], torch
            lab = int(labels[i].item())
            for x in range(grid_k):
                for y in range(grid_k):
                    xs, ys = cord_list[x], cord_list[y]
                    patch = img[xs:xs+patch_size, ys:ys+patch_size].contiguous()  # [10,10]
                    buckets[(lab,x,y)].append(patch.view(-1).cpu().numpy())
    # 堆叠
    for k in buckets.keys():
        if len(buckets[k]) == 0:
            buckets[k] = np.zeros((1, D), dtype=np.float32)
        else:
            buckets[k] = np.stack(buckets[k], axis=0).astype(np.float32)
    return buckets  # dict[(c,x,y)] -> [N, D]


def build_pca_templates_layer1_L2(patch_dict, num_classes, grid_k, patch_size,
                                  pcs_per_bucket=1, seed=0):
    """
    对于每个 (类 c, 位置 x, y)：
      - 做一次 PCA，取前 pcs_per_bucket 个主成分
      - 把这几个主成分“叠加”（这里用求和）成 1 个模板

    最终仍然返回 [num_classes * grid_k * grid_k, 1, patch_size, patch_size]
    对当前网络就是 [90, 1, 10, 10]，conv1 的通道数不变。
    """
    templates = []
    D = patch_size * patch_size

    for c in range(num_classes):
        for x in range(grid_k):
            for y in range(grid_k):
                X = patch_dict[(c, x, y)]          # [N, D]
                if X.shape[0] < 2:
                    # 样本太少，就退化成“均值模板”
                    mu = X.mean(axis=0, keepdims=True) if X.shape[0] > 0 else np.zeros((1, D), dtype=np.float32)
                    merged_pc = mu.reshape(-1)
                else:
                    mu = X.mean(axis=0, keepdims=True)
                    Xc = X - mu

                    # 实际可以用的主成分数不能超过样本数和维度
                    n_comp = min(pcs_per_bucket, Xc.shape[0], D)

                    pca = PCA(n_components=n_comp, random_state=seed)
                    pca.fit(Xc)
                    pcs = pca.components_.copy()   # [n_comp, D]

                    # 可选：每个主成分和均值对齐符号，避免方向随机翻转
                    mu_flat = mu.reshape(-1)
                    for k in range(n_comp):
                        if pcs[k].dot(mu_flat) < 0:
                            pcs[k] = -pcs[k]

                    # 叠加前 n_comp 个主成分，这里用“求和”
                    merged_pc = pcs.sum(axis=0)    # 你也可以改成 pcs.mean(axis=0)

                    # 避免极端情况下全 0（理论上几乎不会发生）
                    if np.allclose(merged_pc, 0):
                        merged_pc = mu_flat

                templates.append(merged_pc.reshape(1, patch_size, patch_size))

    W = torch.tensor(np.stack(templates, axis=0), dtype=torch.float32)  # [90,1,10,10]

    # 零均值 + L2 归一化（和原来一样）
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)

    return W

def collect_flattened_by_class(trainloader, num_classes):
    """
    收集完整的图像数据，用于全局 K-Means。
    """
    class_data = [[] for _ in range(num_classes)]
    for imgs, labels in trainloader:
        # imgs: [B, C, H, W]
        B = imgs.size(0)
        x_flat = imgs.view(B, -1).cpu().numpy()  # [B, D]
        y_np = labels.cpu().numpy()
        for i in range(B):
            class_data[y_np[i]].append(x_flat[i])
            
    # 变成 numpy 数组
    class_data = [np.stack(v, axis=0).astype(np.float32) for v in class_data if len(v)>0]
    # 处理可能的空类别（虽然一般不会有）
    if len(class_data) < num_classes:
        class_data = class_data + [np.zeros((1, x_flat.shape[1]), dtype=np.float32)] * (num_classes - len(class_data))
        
    return class_data


def build_kmeans_templates_sliced_layer1(class_data, num_classes, grid_k, patch_size, stride, full_img_size,
                                         channels=1):
    """
    新逻辑：
    1. 全局 K-Means (K=num_classes) + ClassMean 对齐 -> 得到 10 张完整的“原型图”。
    2. 像处理 ClassMean 那样，把这 10 张原型图切分成 patch，赋值给对应的核。
    """
    # --- Part 1: 全局 K-Means + 对齐 (和 1-Layer 逻辑完全一样) ---

    # 计算 Class Means
    class_means = []
    for Xc in class_data:
        if Xc.shape[0] > 0:
            class_means.append(Xc.mean(axis=0))
        else:
            class_means.append(np.zeros(Xc.shape[1]))
    class_means = np.stack(class_means, axis=0)  # [10, D_total]

    # 全局 K-Means
    X_all = np.concatenate(class_data, axis=0)
    km = KMeans(n_clusters=num_classes, n_init=10, random_state=seed)
    km.fit(X_all)
    kmeans_centers = km.cluster_centers_

    # 对齐
    print("正在对齐 K-Means 中心到对应的类别...")
    ordered_centers = align_templates_to_class_means(kmeans_centers, class_means)  # [10, D_total]

    # 恢复成图片形状 [10, C, H, W]
    prototypes = ordered_centers.reshape(num_classes, channels, full_img_size, full_img_size)
    prototypes = torch.tensor(prototypes, dtype=torch.float32)

    # --- Part 2: Sliding Window 切分 (初始化二层网络的第一层) ---
    # 目标形状: [num_classes * grid_k * grid_k, C, patch_size, patch_size]
    # 也就是 [90, 1, 10, 10] (MNIST) 或 [90, 3, 32, 32] (Sign)

    templates = []

    # 计算切分坐标 (和你原来的 logic 一致)
    cord_list = [i * stride for i in range(grid_k)]

    for c in range(num_classes):
        # 取出第 c 类的 K-Means 原型图
        proto = prototypes[c]  # [C, H, W]

        for x in range(grid_k):
            for y in range(grid_k):
                xs, ys = cord_list[x], cord_list[y]
                # 切片
                patch = proto[:, xs:xs + patch_size, ys:ys + patch_size]  # [C, patch_h, patch_w]
                templates.append(patch)

    W = torch.stack(templates, dim=0)  # [90, C, patch_h, patch_w]

    # 零均值 + L2 归一化
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)

    return W

def gabor_kernel(size, theta, lambd, sigma, gamma=0.6, psi=0.0):
    """ 生成单个 2D Gabor 核 """
    y, x = np.mgrid[0:size, 0:size]
    # 居中坐标
    x = x - (size - 1) / 2
    y = y - (size - 1) / 2

    ct, st = np.cos(theta), np.sin(theta)
    x_prime =  x * ct + y * st
    y_prime = -x * st + y * ct

    gauss = np.exp(-(x_prime**2 + (gamma**2) * y_prime**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * x_prime / lambd + psi)
    g = gauss * sinusoid
    return g.astype(np.float32)

def build_diverse_gabor_bank(num_filters, kernel_size, channels=1):
    """
    生成一个包含 num_filters 个不同参数的 Gabor 核的 Bank。
    参数（theta, lambda, psi）会在一定范围内均匀分布或随机，以保证多样性。
    """
    filters = []
    
    # 为了保证多样性，我们让 theta (方向) 均匀覆盖 [0, pi]
    # 让 lambda (波长) 在一定范围内变化
    for i in range(num_filters):
        # 1. 方向：均匀分布
        theta = (i / num_filters) * np.pi 
        
        # 2. 波长：在 kernel_size 的 20% 到 80% 之间随机，或者是固定的几个档位
        # 这里加入一点随机扰动，让它不那么死板
        lambd = (0.2 + 0.6 * ((i % 5)/5)) * kernel_size 
        
        # 3. 相位：也可以随机一下
        psi = (i % 2) * (np.pi / 2) # 0 或 90度
        
        sigma = kernel_size * 0.15 # 高斯窗宽度
        gamma = 0.6 # 纵横比
        
        # 生成 2D 核
        g = gabor_kernel(kernel_size, theta, lambd, sigma, gamma, psi)
        
        # 扩展通道维度 [C, K, K]
        # 如果是 RGB，通常三个通道用一样的 Gabor 形状（检测亮度边缘）
        g_multichannel = np.stack([g] * channels, axis=0)
        filters.append(g_multichannel)

    W = torch.tensor(np.stack(filters, axis=0), dtype=torch.float32) # [N, C, K, K]
    
    # 归一化
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)
    
    return W

def build_gabor_layer1_diverse(num_classes, grid_k, patch_size, channels=1):
    """
    为第一层生成 [num_classes * grid^2, C, patch_k, patch_k] 的 Gabor 核。
    所有的核都是从 Gabor 分布中采样的，彼此不同。
    """
    total_filters = num_classes * grid_k * grid_k
    W = build_diverse_gabor_bank(num_filters=total_filters, 
                                 kernel_size=patch_size, 
                                 channels=channels)
    return W

def build_gabor_layer2_spatial(in_channels, out_classes, kernel_size):
    """
    为第二层生成 [Out, In, K2, K2] 的 Gabor 核。
    这里我们将 kernel_size (通常是3) 视为空间域。
    生成 Out 个不同的 3x3 Gabor 模式，并将其广播到 In 个输入通道上。
    """
    # 1. 生成 out_classes 个基础的空间 Gabor 模式 [Out, 1, K2, K2]
    # 因为 3x3 很小，theta 的变化主要体现为 横/竖/斜 边缘
    spatial_patterns = build_diverse_gabor_bank(num_filters=out_classes, 
                                                kernel_size=kernel_size, 
                                                channels=1)
    
    # 2. 扩展到输入通道 [Out, In, K2, K2]
    # 我们假设所有输入通道共享同一个空间聚合模式（类似于 Depthwise distinct）
    W = spatial_patterns.repeat(1, in_channels, 1, 1)
    
    # 再次归一化，因为通道数变多了，数值范围可能会变
    # 注意：这里按 (In, K, K) 整体归一化
    W = W - W.mean(dim=(1, 2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)
    
    return W


def visualize_filter_bank(W, filename):
    """
    用和你原来类似的方式，把一组 [N,1,H,W] 的 kernel 摆成网格，画成 heatmap 并保存。
    """
    if W.dim() == 3:         # [N,H,W] -> [N,1,H,W]
        W = W.unsqueeze(1)
    # 和你原来的做法一致：make_grid 然后对 channel 求平均
    grid = torchvision.utils.make_grid(W, normalize=True)  # [C, H_grid, W_grid]
    grid = grid.mean(dim=0)                               # [H_grid, W_grid]

    plt.figure(figsize=(4, 4))
    sns.heatmap(torch.exp(grid), square=True, cmap="YlGnBu", cbar=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path_name + filename)
    plt.close()



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv2_k * conv2_k * 10, kernel_size=conv1_k, padding=0, stride=conv1_s, bias=True)
        # self.linear1 = nn.Linear(64*6*6, 10)
        self.conv2 = nn.Conv2d(conv2_k * conv2_k * 10, 10, kernel_size=conv2_k, padding=0, stride=1)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1)

    def forward(self, x):
        # with torch.no_grad():
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = torch.flatten(x, 1)
        # x = self.linear1(x.squeeze())
        return x.squeeze()


record_mean = torch.zeros(len(classes) * conv2_k * conv2_k, conv1_k, conv1_k)
record_cnt = torch.zeros(len(classes) * conv2_k * conv2_k)
cord_list = [i*conv1_s for i in cord_cnt]
for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    for i in range(len(labels)):
        for x_index in range(conv2_k):
            for y_index in range(conv2_k):
                kernel_idx = labels[i] * conv2_k * conv2_k + x_index * conv2_k + y_index
                record_mean[kernel_idx, :, :] = record_mean[kernel_idx, :, :] \
                                                + inputs[i, 0, cord_list[x_index]:cord_list[x_index] + conv1_k,
                                                  cord_list[y_index]:cord_list[y_index] + conv1_k]
                record_cnt[kernel_idx] = record_cnt[kernel_idx] + 1

for i in range(len(classes) * conv2_k * conv2_k):
    record_mean[i, :, :] = record_mean[i, :, :] / record_cnt[i]
    # record_mean[i, :, :] = record_mean[i, :, :] / record_mean[i, :, :].sum()
    # record_mean[i, :, :] = record_mean[i, :, :] / record_mean[i, :, :].max()

record_mean_layer2 = torch.zeros(10, len(classes) * conv2_k * conv2_k, conv2_k, conv2_k)
for i in range(10):
    for x_index in range(conv2_k):
        for y_index in range(conv2_k):
            kernel_idx = i * conv2_k * conv2_k + x_index * conv2_k + y_index
            record_mean_layer2[i, kernel_idx, x_index, y_index] = 1

record_mean = record_mean.unsqueeze(dim=1)
net_weights = torchvision.utils.make_grid(record_mean, normalize=True)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
if kernel_init_type == 'ClassMean':
    plt.savefig(path_name + 'heatmap_filter_1_init.pdf')

net_weights = torchvision.utils.make_grid(record_mean_layer2, normalize=True)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_2_init.pdf')


# ============================================================
# 额外：基于 MNIST 样本生成三组初始化 kernel 并可视化
# ============================================================
if visualize_init_kernels:
    num_classes = len(classes)
    size = target_size

    # 1) 准备按类分好的展平数据（PCA / k-means 都要用）
    print("收集各类别样本并展平...")
    # 先按(类,位置)收集 10x10 patch
    patch_dict = collect_patches_by_class_and_pos(trainloader,
                                                  num_classes=len(classes), patch_size=conv1_k,
                                                  stride=conv1_s, grid_k=conv2_k)

    # 2. 【新增】收集全图数据用于 K-Means
    class_data_full = collect_flattened_by_class(trainloader, num_classes=len(classes))

    # 对应三种初始化（第1层）
    W_pca = build_pca_templates_layer1_L2(
        patch_dict,
        num_classes=len(classes),
        grid_k=conv2_k,
        patch_size=conv1_k,
        pcs_per_bucket=pca_components_per_class,  # 比如 3
        seed=seed
    )
    # [90,1,10,10]
    input_channels = 3 if 'SIGN' in path_name else 1 # 简单的自动判断，或者你手动写死
    
    W_kmeans = build_kmeans_templates_sliced_layer1(
        class_data=class_data_full,
        num_classes=len(classes),
        grid_k=conv2_k,
        patch_size=conv1_k,
        stride=conv1_s,
        full_img_size=target_size,
        channels=input_channels
    )
    input_channels = 3 if 'SIGN' in path_name or 'Sign' in path_name else 1

    # 【修改 Layer 1 Gabor 生成】
    # 不再是简单的重复，而是生成 90 个完全不同的核
    W_gabor = build_gabor_layer1_diverse(num_classes=len(classes),
                                         grid_k=conv2_k,
                                         patch_size=conv1_k,
                                         channels=input_channels)

    # 【新增 Layer 2 Gabor 生成】
    # 输入通道数 = conv2_k * conv2_k * 10
    layer2_in_channels = conv2_k * conv2_k * len(classes)
    W_gabor_L2 = build_gabor_layer2_spatial(in_channels=layer2_in_channels,
                                            out_classes=len(classes),
                                            kernel_size=conv2_k)  # 通常是 3

    # ... 可视化部分 ...
    if kernel_init_type == 'Gabor':
        visualize_filter_bank(W_gabor, 'heatmap_filter_gabor_init.pdf')
        # 如果你想看第二层的 Gabor (选前10个通道看看)
        visualize_filter_bank(W_gabor_L2[:, 0:1, :, :], 'heatmap_filter_gabor_L2_init.pdf')

    # ---------- (1) PCA / SVD of class subspaces ----------
    if kernel_init_type == 'PCA':
        visualize_filter_bank(W_pca, 'heatmap_filter_PCA_init.pdf')

    # ---------- (2) k-means templates ----------
    if kernel_init_type == 'KMeans':
        visualize_filter_bank(W_kmeans, 'heatmap_filter_kmeans_init.pdf')


    print("三种初始化 kernel 的可视化已保存到：", path_name)



tbwriter = SummaryWriter(f'logs/conv1conv2_{kernel_init_type}')
net = Net().to(device)


# 把几组 kernel 都搬到当前 device 上
record_mean = record_mean.to(device)      # 类平均 [10,1,20,20]
if visualize_init_kernels:
    W_pca    = W_pca.to(device)           # PCA [10,1,20,20]
    W_kmeans = W_kmeans.to(device)        # k-means [10,1,20,20]
    W_gabor  = W_gabor.to(device)
    W_gabor_L2 = W_gabor_L2.to(device) # 【新增】

# ===== 根据 kernel_init_type 选择初始化方式 =====
if kernel_init_type == 'Random':
    pass
elif kernel_init_type == 'ClassMean':
    net.conv1.weight.data = record_mean.to(device)            # [90,1,10,10]
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'PCA':
    net.conv1.weight.data = W_pca.to(device)                  # [90,1,10,10]
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'KMeans':
    net.conv1.weight.data = W_kmeans.to(device)               # [90,1,10,10]
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'Gabor':
    # 【修改】第一层赋值
    net.conv1.weight.data = W_gabor.to(device)
    net.conv1.bias.data.zero_()
    
    # 【新增】第二层赋值：不再使用 record_mean_layer2，而是使用 Gabor L2
    net.conv2.weight.data = W_gabor_L2.to(device)
    # net.conv2.bias.data.zero_() # 可选
else:
    raise ValueError(f"未知的 kernel_init_type: {kernel_init_type}")

# 接下来照旧
record_mean_layer2 = record_mean_layer2.to(device)
if not (kernel_init_type == 'Random' or kernel_init_type == 'Gabor') :
    net.conv2.weight.data = record_mean_layer2
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)



cnt = 0
for epoch in range(epoch_num):  # loop over the dataset multiple times
    running_loss = 0.0
    # 计算batch数量
    count_batches = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        tbwriter.add_scalar('Training_loss', loss.item(), cnt)
        cnt = cnt + 1
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

        # 计算batch数量
        count_batches += 1

    # 计算该epoch的平均训练loss
    avg_loss = running_loss / count_batches
    loss_list.append(avg_loss)

    # 在每个epoch结束后计算测试准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbs in testloader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            _, pred = torch.max(net(imgs), 1)
            total += lbs.size(0)
            correct += (pred == lbs).sum().item()
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    print(f'Epoch {epoch + 1}: test acc = {test_acc:.2f}%')
    net.train()

print('Finished Training')
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# print images
# imshow(torchvision.utils.make_grid(images))
net_weights = torchvision.utils.make_grid(net.conv1.weight.data, normalize=True, nrow=15)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights.cpu()), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_1_finish.pdf')
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(64)))

net_weights = torchvision.utils.make_grid(net.conv2.weight.data, normalize=True, nrow=10)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights.cpu()), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_2_finish.pdf')


outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(64)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# build confusion matrix
y_pred = []
y_true = []

# iterate over test data
for inputs, labels in testloader:
    inputs = inputs.to(device)
    output = net(inputs)  # Feed Network

    output = (torch.max(output, 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
cf_matrix = cf_matrix * 100
df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 8))
plot_args = {'fontsize': 'xx-large'}
sns.heatmap(df_cm, annot=True, fmt=".1f", annot_kws=plot_args)
plt.savefig(path_name + 'confusion_matrix.pdf')



# 训练结束后
import csv
# 保存数据到CSV文件
if not os.path.exists(f"plot_data/seed={seed}/MNIST_2layer/"):
    os.makedirs(f"plot_data/seed={seed}/MNIST_2layer/", exist_ok=True)
csv_filename = f'plot_data/seed={seed}/MNIST_2layer/{kernel_init_type}.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入标题行
    writer.writerow(['Epoch', 'Training Loss', 'Test Accuracy'])
    # 写入每个epoch的数据
    for epoch, (loss, acc) in enumerate(zip(loss_list, test_acc_list), start=1):
        writer.writerow([epoch, loss, acc])

print(f"训练指标已保存到 {csv_filename}")
