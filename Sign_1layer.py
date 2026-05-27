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

epoch_num = args.epochs
path_name = f'output_img/seed={seed}/SIGN_1layer/{kernel_init_type}/'
import os
os.makedirs(path_name, exist_ok=True)

loss_list = []
test_acc_list = []

# functions to show an image
target_size = 64
transform = transforms.Compose(
    [torchvision.transforms.Resize(target_size),
     transforms.ToTensor(),
     # torchvision.transforms.Lambda(
     #     lambda samples: (samples - 0.5) * 2),
     torchvision.transforms.Lambda(
         lambda samples: samples.view(3, target_size, target_size))])

batch_size = 64

train_dataset = torchvision.datasets.folder.ImageFolder(root='../data/sign_language_mnist/train_10', transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
test_dataset = torchvision.datasets.folder.ImageFolder(root='../data/sign_language_mnist/test_10', transform=transform)

testloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=0)

classes = ('C', 'E', 'I', 'K', 'L', 'O', 'P', 'Q', 'X', 'Y')

print('训练集图片总数：', len(train_dataset))
print('测试集图片总数：', len(test_dataset))

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path_name + 'image_grid.pdf')


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

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

def collect_flattened_by_class(trainloader, num_classes, size):
    """
    把每个类别的样本收集起来，展平到 [N_c, D]，返回一个长度为 num_classes 的 list
    每个元素是该类的数据矩阵 X_c: [N_c, D]
    """
    class_data = [[] for _ in range(num_classes)]
    for imgs, labels in trainloader:
        # imgs: [B, 1, H, W]
        B = imgs.size(0)
        x_flat = imgs.view(B, -1).cpu().numpy()  # [B, D]
        y_np = labels.cpu().numpy()
        for i in range(B):
            class_data[y_np[i]].append(x_flat[i])
    # 变成 numpy 数组
    class_data = [np.stack(v, axis=0).astype(np.float32) for v in class_data]
    return class_data  # list, 每个元素 [N_c, D]


def build_pca_templates(class_data, pcs_per_class, size):
    """
    对于每个类别：
      - 做一次 PCA，取前 pcs_per_class 个主成分
      - 把这几个主成分“叠加”（例如求和）成 1 个模板
    最终返回 W: [num_classes, 1, size, size]，也就是 10 个 kernel
    """
    templates = []
    for Xc in class_data:
        # Xc: [N_c, D]
        Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)

        # 这里 n_components = pcs_per_class = 3
        pca = PCA(n_components=pcs_per_class, random_state=seed)
        pca.fit(Xc_centered)

        # 叠加前 pcs_per_class 个主成分
        # shape: [pcs_per_class, D] -> [D]
        merged_pc = pca.components_[:pcs_per_class].sum(axis=0)
        # 如果你更喜欢“平均”，可以改成：
        # merged_pc = pca.components_[:pcs_per_class].mean(axis=0)

        pc2d = merged_pc.reshape(3, size, size)    # [1, H, W]
        templates.append(pc2d)

    # 现在 templates 长度 = num_classes (=10)
    W = torch.tensor(np.stack(templates, axis=0), dtype=torch.float32)  # [10, 1, H, W]

    # 零均值 + L2 归一化（原代码保持不变）
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = W.view(W.size(0), -1).norm(dim=1, keepdim=True)
    W = W / (norms.view(-1, 1, 1, 1) + 1e-8)
    return W




def build_kmeans_templates_global(class_data, k, size):
    """
    改进版：
    1. 计算所有类的平均值 (ClassMean) 作为对齐基准。
    2. 将所有数据混合，做 K=10 的 K-Means。
    3. 将 K-Means 结果与 ClassMean 做 MSE 匹配，重新排序。
    """
    # 1. 计算 ClassMean (作为对齐的目标)
    # class_data 是一个 list，每个元素是 [N_c, D]
    class_means = []
    for Xc in class_data:
        # 计算该类的均值向量 [D]
        if Xc.shape[0] > 0:
            class_means.append(Xc.mean(axis=0))
        else:
            class_means.append(np.zeros(Xc.shape[1]))
    class_means = np.stack(class_means, axis=0) # [10, D]

    # 2. 准备全局数据并聚类
    X_all = np.concatenate(class_data, axis=0)  # [N_total, D]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(X_all)
    kmeans_centers = km.cluster_centers_        # [10, D] (无序)

    # 3. 对齐 / 匹配
    print("正在对齐 K-Means 中心到对应的类别...")
    ordered_centers = align_templates_to_class_means(kmeans_centers, class_means)

    # 4. Reshape 回图片形状
    # 注意：MNIST/Fashion 是 1 通道，Sign 是 3 通道
    # 判断一下维度 D
    D = ordered_centers.shape[1]
    if D == size * size:
        # 单通道 (MNIST/Fashion)
        reshaped_centers = ordered_centers.reshape(k, 1, size, size)
    else:
        # 3通道 (Sign)
        reshaped_centers = ordered_centers.reshape(k, 3, size, size)

    W = torch.tensor(reshaped_centers, dtype=torch.float32)

    # 零均值 + L2 归一化
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = W.view(W.size(0), -1).norm(dim=1, keepdim=True)
    W = W / (norms.view(-1, 1, 1, 1) + 1e-8)
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
        self.conv1 = nn.Conv2d(3, len(classes), kernel_size=64, padding=0, stride=1, bias=True)


    def forward(self, x):
        x = F.relu(self.conv1(x))

        return x.squeeze()


record_mean = torch.zeros(len(classes), 3, target_size, target_size)
record_cnt = torch.zeros(len(classes))
for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    for i in range(len(labels)):
        record_mean[labels[i], :, :, :] = record_mean[labels[i], :, :, :] + inputs[i, :, :, :]
        record_cnt[labels[i]] = record_cnt[labels[i]] + 1

for i in range(len(classes)):
    record_mean[i, :, :, :] = record_mean[i, :, :, :] / record_cnt[i]


# record_mean = record_mean.unsqueeze(dim=1)
net_weights = torchvision.utils.make_grid(record_mean, normalize=True)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
if kernel_init_type == 'ClassMean':
    plt.savefig(path_name + 'heatmap_filter_init.pdf')

# ============================================================
# 额外：基于 MNIST 样本生成三组初始化 kernel 并可视化
# ============================================================
if visualize_init_kernels:
    num_classes = len(classes)
    size = target_size

    # 1) 准备按类分好的展平数据（PCA / k-means 都要用）
    print("收集各类别样本并展平...")
    class_data = collect_flattened_by_class(trainloader, num_classes=num_classes, size=size)

    # ---------- (1) PCA / SVD of class subspaces ----------
    print("构造 PCA 初始化 kernel ...")
    W_pca = build_pca_templates(class_data,
                                pcs_per_class=pca_components_per_class,
                                size=size)       # 形状 [num_filters,1,size,size]
    if kernel_init_type == 'PCA':
        visualize_filter_bank(W_pca, 'heatmap_filter_PCA_init.pdf')

    # ---------- (2) k-means templates ----------
    print("构造 k-means 初始化 kernel ...")
    W_kmeans = build_kmeans_templates_global(class_data,
                                             k=kmeans_global_k,
                                             size=size)
    if kernel_init_type == 'KMeans':
        visualize_filter_bank(W_kmeans, 'heatmap_filter_kmeans_init.pdf')

    # ---------- (3) Gabor banks ----------
# 【修改后】
    # 自动判断通道数
    input_channels = 3 if 'Sign' in path_name or 'SIGN' in path_name else 1
    
    # 直接生成 10 个（或 num_classes 个）各不相同的 Gabor 核
    W_gabor = build_diverse_gabor_bank(num_filters=len(classes), 
                                       kernel_size=target_size, # 比如 28 或 20
                                       channels=input_channels)

    if kernel_init_type == 'Gabor':
        visualize_filter_bank(W_gabor, 'heatmap_filter_gabor_init.pdf')

    print("三种初始化 kernel 的可视化已保存到：", path_name)


# 日志目录加上初始化类型标签
tbwriter = SummaryWriter(f'logs/conv1_{kernel_init_type}')

net = Net().to(device)

# 把几组 kernel 都搬到当前 device 上
record_mean = record_mean.to(device)      # 类平均 [10,1,20,20]
if visualize_init_kernels:
    W_pca    = W_pca.to(device)           # PCA [10,1,20,20]
    W_kmeans = W_kmeans.to(device)        # k-means [10,1,20,20]
    W_gabor  = W_gabor.to(device)         # Gabor [10,1,20,20]

# ===== 根据 kernel_init_type 选择初始化方式 =====
if kernel_init_type == 'Random':
    # 完全随机，不改任何东西
    pass

elif kernel_init_type == 'ClassMean':
    # 用类平均图像初始化 conv1
    net.conv1.weight.data = record_mean
    net.conv1.bias.data.zero_()

elif kernel_init_type == 'PCA':
    net.conv1.weight.data = W_pca
    net.conv1.bias.data.zero_()

elif kernel_init_type == 'KMeans':
    net.conv1.weight.data = W_kmeans
    net.conv1.bias.data.zero_()

elif kernel_init_type == 'Gabor':
    net.conv1.weight.data = W_gabor
    net.conv1.bias.data.zero_()

else:
    raise ValueError(f"未知的 kernel_init_type: {kernel_init_type}")

# 接下来照旧
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


print("模型参数：", (net.parameters()))
for param in net.parameters():
    print("参数类型：", type(param), "参数大小：", param.size())
total_num = sum(p.numel() for p in net.parameters())
print("参数量：", total_num)
print(net)
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
        if i % 5 == 4:
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

print('Finished Training')
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# print images
# imshow(torchvision.utils.make_grid(images))
net_weights = torchvision.utils.make_grid(net.conv1.weight.data, normalize=True, nrow=5)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights.cpu()), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_finish.pdf')
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(64)))



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


########################################
# 3)  t-SNE 可视化网络输出（最后一层，即 conv1->ReLU->squeeze 后的 [N,10]）
from sklearn.manifold import TSNE  # 用于 t-SNE
features_list = []
labels_list = []
net.eval()
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        outputs = net(inputs)     # shape [batch_size,10]
        features_list.append(outputs.cpu().numpy())
        labels_list.append(labels.numpy())

features = np.concatenate(features_list, axis=0)  # [N,10]
labels_all = np.concatenate(labels_list, axis=0)  # [N]

print("Running t-SNE on test-set outputs, shape =", features.shape)
tsne = TSNE(n_components=2, perplexity=30, init='random', learning_rate='auto')
features_2d = tsne.fit_transform(features)  # [N,2]

plt.figure(figsize=(5,5))
for digit in range(10):
    idxs = np.where(labels_all == digit)
    plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1],
                label=f"{classes[digit]}",
                alpha=0.6, s=10)
# plt.title("t-SNE of final layer outputs on MNIST test set")
plt.legend()
plt.savefig(path_name + 't_SNE.pdf')

########################################
# 4)  频域分析（可选）
#    由于 kernel_size=20，可以看看每个卷积核在频域是何种形态
final_weights = net.conv1.weight.data.cpu()
filter0 = final_weights[0, 0, :, :].numpy()  # [20,20]
fshift = np.fft.fftshift(np.fft.fft2(filter0))
magnitude = np.log(1 + np.abs(fshift))

plt.figure(figsize=(5,4))
im = plt.imshow(magnitude, cmap='hot', vmin=0)
plt.colorbar()


# 3) 现在就可以查询当前的上下限：
vmin, vmax = im.get_clim()
print(f"当前的 vmin = {vmin:.4f}, vmax = {vmax:.4f}")

# plt.title("Frequency Magnitude Spectrum of Filter[0]")
plt.savefig(path_name + 'magnitude.pdf')
########################################

# 训练结束后
import csv
# 保存数据到CSV文件
if not os.path.exists(f"plot_data/seed={seed}/SIGN_1layer/"):
    os.makedirs(f"plot_data/seed={seed}/SIGN_1layer/", exist_ok=True)
csv_filename = f'plot_data/seed={seed}/SIGN_1layer/{kernel_init_type}.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入标题行
    writer.writerow(['Epoch', 'Training Loss', 'Test Accuracy'])
    # 写入每个epoch的数据
    for epoch, (loss, acc) in enumerate(zip(loss_list, test_acc_list), start=1):
        writer.writerow([epoch, loss, acc])

print(f"训练指标已保存到 {csv_filename}")
