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
import os
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-s","--seed", type=int, required=True)
ap.add_argument("-t","--type", type=int, required=True)
ap.add_argument("-e", "--epochs", type=int, default=500)

args = ap.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

type_ = ['Random', 'ClassMean', 'PCA', 'KMeans', 'Gabor']
kernel_init_type = type_[args.type]

visualize_init_kernels = True

pca_components_per_class = 1
kmeans_global_k = 10

gabor_num_orientations = 5
gabor_num_scales = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_random_init = False
epoch_num = args.epochs

# ---- CIFAR-10 specific ----
target_size  = 32
in_channels  = 3   # RGB
conv1_k = 8               # floor((32-8)/6)+1 = 5  → 5×5 feature map
conv1_s = 6
conv2_k = 5
cord_cnt = [0, 1, 2, 3, 4]

path_name = f'output_img/seed={seed}/CIFAR10_2layer/{kernel_init_type}/'
os.makedirs(path_name, exist_ok=True)

loss_list = []
test_acc_list = []

# ---- Transform: CIFAR-10 is already RGB 32×32, no Lambda needed ----
transform = transforms.Compose([
    torchvision.transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path_name + 'image_grid.pdf')


dataiter = iter(trainloader)
images, labels = next(dataiter)
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# ============================================================
# 初始化方式 + 通用可视化函数
# ============================================================
def align_templates_to_class_means(kmeans_centers, class_means):
    K = kmeans_centers.shape[0]
    dists = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dists[i, j] = np.linalg.norm(kmeans_centers[i] - class_means[j])

    ordered_centers = np.zeros_like(kmeans_centers)
    matched_kmeans_idx = set()
    matched_class_idx = set()
    mapping = {}

    for _ in range(K):
        min_val = np.inf
        best_k = -1
        best_c = -1
        for k in range(K):
            if k in matched_kmeans_idx: continue
            for c in range(K):
                if c in matched_class_idx: continue
                if dists[k, c] < min_val:
                    min_val = dists[k, c]
                    best_k = k
                    best_c = c
        matched_kmeans_idx.add(best_k)
        matched_class_idx.add(best_c)
        mapping[best_c] = best_k

    for c in range(K):
        ordered_centers[c] = kmeans_centers[mapping[c]]

    return ordered_centers


def collect_patches_by_class_and_pos(trainloader, num_classes, patch_size, stride, grid_k=3, channels=1):
    """
    收集每个(类c, 网格位置x,y)的 patch，返回 dict[(c,x,y)] = np.array [N, D]
    D = channels * patch_size * patch_size
    """
    D = channels * patch_size * patch_size
    buckets = {(c,x,y): [] for c in range(num_classes)
                          for x in range(grid_k) for y in range(grid_k)}
    cord_list = [i*stride for i in range(grid_k)]
    for imgs, labels in trainloader:
        B = imgs.size(0)
        for i in range(B):
            img = imgs[i]           # [C, H, W]
            lab = int(labels[i].item())
            for x in range(grid_k):
                for y in range(grid_k):
                    xs, ys = cord_list[x], cord_list[y]
                    patch = img[:, xs:xs+patch_size, ys:ys+patch_size].contiguous()  # [C, p, p]
                    buckets[(lab,x,y)].append(patch.view(-1).cpu().numpy())
    for k in buckets.keys():
        if len(buckets[k]) == 0:
            buckets[k] = np.zeros((1, D), dtype=np.float32)
        else:
            buckets[k] = np.stack(buckets[k], axis=0).astype(np.float32)
    return buckets


def build_pca_templates_layer1_L2(patch_dict, num_classes, grid_k, patch_size,
                                  pcs_per_bucket=1, seed=0, channels=1):
    templates = []
    D = channels * patch_size * patch_size

    for c in range(num_classes):
        for x in range(grid_k):
            for y in range(grid_k):
                X = patch_dict[(c, x, y)]
                if X.shape[0] < 2:
                    mu = X.mean(axis=0, keepdims=True) if X.shape[0] > 0 else np.zeros((1, D), dtype=np.float32)
                    merged_pc = mu.reshape(-1)
                else:
                    mu = X.mean(axis=0, keepdims=True)
                    Xc = X - mu
                    n_comp = min(pcs_per_bucket, Xc.shape[0], D)
                    pca = PCA(n_components=n_comp, random_state=seed)
                    pca.fit(Xc)
                    pcs = pca.components_.copy()
                    mu_flat = mu.reshape(-1)
                    for k in range(n_comp):
                        if pcs[k].dot(mu_flat) < 0:
                            pcs[k] = -pcs[k]
                    merged_pc = pcs.sum(axis=0)
                    if np.allclose(merged_pc, 0):
                        merged_pc = mu_flat

                # reshape to [C, patch_size, patch_size]
                templates.append(merged_pc.reshape(channels, patch_size, patch_size))

    W = torch.tensor(np.stack(templates, axis=0), dtype=torch.float32)  # [90, C, p, p]

    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)

    return W


def collect_flattened_by_class(trainloader, num_classes):
    class_data = [[] for _ in range(num_classes)]
    for imgs, labels in trainloader:
        B = imgs.size(0)
        x_flat = imgs.view(B, -1).cpu().numpy()
        y_np = labels.cpu().numpy()
        for i in range(B):
            class_data[y_np[i]].append(x_flat[i])
    class_data = [np.stack(v, axis=0).astype(np.float32) for v in class_data if len(v)>0]
    if len(class_data) < num_classes:
        class_data = class_data + [np.zeros((1, x_flat.shape[1]), dtype=np.float32)] * (num_classes - len(class_data))
    return class_data


def build_kmeans_templates_sliced_layer1(class_data, num_classes, grid_k, patch_size, stride, full_img_size,
                                         channels=1):
    class_means = []
    for Xc in class_data:
        if Xc.shape[0] > 0:
            class_means.append(Xc.mean(axis=0))
        else:
            class_means.append(np.zeros(Xc.shape[1]))
    class_means = np.stack(class_means, axis=0)

    X_all = np.concatenate(class_data, axis=0)
    km = KMeans(n_clusters=num_classes, n_init=10, random_state=seed)
    km.fit(X_all)
    kmeans_centers = km.cluster_centers_

    print("正在对齐 K-Means 中心到对应的类别...")
    ordered_centers = align_templates_to_class_means(kmeans_centers, class_means)

    prototypes = ordered_centers.reshape(num_classes, channels, full_img_size, full_img_size)
    prototypes = torch.tensor(prototypes, dtype=torch.float32)

    templates = []
    cord_list = [i * stride for i in range(grid_k)]

    for c in range(num_classes):
        proto = prototypes[c]  # [C, H, W]
        for x in range(grid_k):
            for y in range(grid_k):
                xs, ys = cord_list[x], cord_list[y]
                patch = proto[:, xs:xs + patch_size, ys:ys + patch_size]
                templates.append(patch)

    W = torch.stack(templates, dim=0)  # [90, C, p, p]

    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)

    return W


def gabor_kernel(size, theta, lambd, sigma, gamma=0.6, psi=0.0):
    y, x = np.mgrid[0:size, 0:size]
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
    filters = []
    for i in range(num_filters):
        theta = (i / num_filters) * np.pi
        lambd = (0.2 + 0.6 * ((i % 5)/5)) * kernel_size
        psi   = (i % 2) * (np.pi / 2)
        sigma = kernel_size * 0.15
        gamma = 0.6
        g = gabor_kernel(kernel_size, theta, lambd, sigma, gamma, psi)
        g_multichannel = np.stack([g] * channels, axis=0)
        filters.append(g_multichannel)

    W = torch.tensor(np.stack(filters, axis=0), dtype=torch.float32)
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)
    return W


def build_gabor_layer1_diverse(num_classes, grid_k, patch_size, channels=1):
    total_filters = num_classes * grid_k * grid_k
    W = build_diverse_gabor_bank(num_filters=total_filters,
                                 kernel_size=patch_size,
                                 channels=channels)
    return W


def build_gabor_layer2_spatial(in_channels, out_classes, kernel_size):
    spatial_patterns = build_diverse_gabor_bank(num_filters=out_classes,
                                                kernel_size=kernel_size,
                                                channels=1)
    W = spatial_patterns.repeat(1, in_channels, 1, 1)
    W = W - W.mean(dim=(1, 2, 3), keepdim=True)
    norms = torch.linalg.norm(W.view(W.size(0), -1), dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)
    return W


def visualize_filter_bank(W, filename):
    if W.dim() == 3:
        W = W.unsqueeze(1)
    grid = torchvision.utils.make_grid(W, normalize=True, nrow=15)
    grid = grid.mean(dim=0)
    plt.figure(figsize=(4, 4))
    sns.heatmap(torch.exp(grid), square=True, cmap="YlGnBu", cbar=False)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path_name + filename)
    plt.close()


# ============================================================
# Network
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1: 3 → 250 channels, kernel 8×8, stride 6  →  5×5 output
        self.conv1 = nn.Conv2d(in_channels, conv2_k * conv2_k * 10,
                               kernel_size=conv1_k, padding=0, stride=conv1_s, bias=True)
        # conv2: 250 → 10, kernel 5×5  →  1×1 output
        self.conv2 = nn.Conv2d(conv2_k * conv2_k * 10, 10,
                               kernel_size=conv2_k, padding=0, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)   # 输出层不加 ReLU，保留负值 logits
        return x.squeeze()


# ============================================================
# ClassMean templates (Layer 1) — inline computation
# ============================================================
# record_mean: [90, 3, conv1_k, conv1_k]  (3 channels for RGB)
record_mean = torch.zeros(len(classes) * conv2_k * conv2_k, in_channels, conv1_k, conv1_k)
record_cnt  = torch.zeros(len(classes) * conv2_k * conv2_k)
cord_list   = [i * conv1_s for i in cord_cnt]

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    for i in range(len(labels)):
        for x_index in range(conv2_k):
            for y_index in range(conv2_k):
                kernel_idx = labels[i] * conv2_k * conv2_k + x_index * conv2_k + y_index
                record_mean[kernel_idx] += inputs[i, :,
                                            cord_list[x_index]:cord_list[x_index] + conv1_k,
                                            cord_list[y_index]:cord_list[y_index] + conv1_k]
                record_cnt[kernel_idx] += 1

for i in range(len(classes) * conv2_k * conv2_k):
    record_mean[i] = record_mean[i] / record_cnt[i]

# Layer-2 identity init (unchanged)
record_mean_layer2 = torch.zeros(10, len(classes) * conv2_k * conv2_k, conv2_k, conv2_k)
for i in range(10):
    for x_index in range(conv2_k):
        for y_index in range(conv2_k):
            kernel_idx = i * conv2_k * conv2_k + x_index * conv2_k + y_index
            record_mean_layer2[i, kernel_idx, x_index, y_index] = 1

# Visualise ClassMean init
net_weights = torchvision.utils.make_grid(record_mean, normalize=True)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
if kernel_init_type == 'ClassMean':
    plt.savefig(path_name + 'heatmap_filter_1_init.pdf')
plt.close()

net_weights = torchvision.utils.make_grid(record_mean_layer2, normalize=True)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_2_init.pdf')
plt.close()


# ============================================================
# 其他初始化方式 (PCA / KMeans / Gabor)
# ============================================================
if visualize_init_kernels:
    num_classes = len(classes)

    print("收集各类别样本并展平...")
    patch_dict = collect_patches_by_class_and_pos(trainloader,
                                                  num_classes=num_classes,
                                                  patch_size=conv1_k,
                                                  stride=conv1_s,
                                                  grid_k=conv2_k,
                                                  channels=in_channels)  # 3 channels

    class_data_full = collect_flattened_by_class(trainloader, num_classes=num_classes)

    W_pca = build_pca_templates_layer1_L2(
        patch_dict,
        num_classes=num_classes,
        grid_k=conv2_k,
        patch_size=conv1_k,
        pcs_per_bucket=pca_components_per_class,
        seed=seed,
        channels=in_channels
    )

    W_kmeans = build_kmeans_templates_sliced_layer1(
        class_data=class_data_full,
        num_classes=num_classes,
        grid_k=conv2_k,
        patch_size=conv1_k,
        stride=conv1_s,
        full_img_size=target_size,
        channels=in_channels
    )

    W_gabor = build_gabor_layer1_diverse(num_classes=num_classes,
                                          grid_k=conv2_k,
                                          patch_size=conv1_k,
                                          channels=in_channels)

    layer2_in_channels = conv2_k * conv2_k * len(classes)
    W_gabor_L2 = build_gabor_layer2_spatial(in_channels=layer2_in_channels,
                                             out_classes=len(classes),
                                             kernel_size=conv2_k)

    if kernel_init_type == 'Gabor':
        visualize_filter_bank(W_gabor, 'heatmap_filter_gabor_init.pdf')
        visualize_filter_bank(W_gabor_L2[:, 0:1, :, :], 'heatmap_filter_gabor_L2_init.pdf')

    if kernel_init_type == 'PCA':
        visualize_filter_bank(W_pca, 'heatmap_filter_PCA_init.pdf')

    if kernel_init_type == 'KMeans':
        visualize_filter_bank(W_kmeans, 'heatmap_filter_kmeans_init.pdf')

    print("初始化 kernel 可视化已保存到：", path_name)


# ============================================================
# 构建网络并赋值初始权重
# ============================================================
tbwriter = SummaryWriter(f'logs/CIFAR10_conv1conv2_{kernel_init_type}_seed{seed}')
net = Net().to(device)

record_mean = record_mean.to(device)
if visualize_init_kernels:
    W_pca      = W_pca.to(device)
    W_kmeans   = W_kmeans.to(device)
    W_gabor    = W_gabor.to(device)
    W_gabor_L2 = W_gabor_L2.to(device)

if kernel_init_type == 'Random':
    pass
elif kernel_init_type == 'ClassMean':
    net.conv1.weight.data = record_mean.to(device)
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'PCA':
    net.conv1.weight.data = W_pca.to(device)
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'KMeans':
    net.conv1.weight.data = W_kmeans.to(device)
    net.conv1.bias.data.zero_()
elif kernel_init_type == 'Gabor':
    net.conv1.weight.data = W_gabor.to(device)
    net.conv1.bias.data.zero_()
    net.conv2.weight.data = W_gabor_L2.to(device)
else:
    raise ValueError(f"未知的 kernel_init_type: {kernel_init_type}")

record_mean_layer2 = record_mean_layer2.to(device)
if not (kernel_init_type == 'Random' or kernel_init_type == 'Gabor'):
    net.conv2.weight.data = record_mean_layer2

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# ============================================================
# Training
# ============================================================
cnt = 0
for epoch in range(epoch_num):
    net.train()
    running_loss = 0.0
    count_batches = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tbwriter.add_scalar('Training_loss', loss.item(), cnt)
        cnt += 1
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
        count_batches += 1

    avg_loss = running_loss / max(count_batches, 1)
    loss_list.append(avg_loss)

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    print(f'Epoch {epoch + 1}: test acc = {test_acc:.2f}%')
    net.train()

print('Finished Training')

# ============================================================
# Evaluation
# ============================================================
net.eval()

dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

net_weights = torchvision.utils.make_grid(net.conv1.weight.data.cpu(), normalize=True, nrow=15)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_1_finish.pdf')
plt.close()
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(64)))

net_weights = torchvision.utils.make_grid(net.conv2.weight.data.cpu(), normalize=True, nrow=10)
net_weights = net_weights.mean(dim=0)
sns.heatmap(torch.exp(net_weights), square=True, cmap="YlGnBu", cbar=False)
plt.axis('off')
plt.savefig(path_name + 'heatmap_filter_2_finish.pdf')
plt.close()

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(64)))

correct = total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred   = {classname: 0 for classname in classes}
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Confusion matrix
y_pred, y_true = [], []
for inputs, labels in testloader:
    inputs = inputs.to(device)
    output = (torch.max(net(inputs), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)
    y_true.extend(labels.data.cpu().numpy())

cf_matrix = confusion_matrix(y_true, y_pred, normalize='true') * 100
df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
plt.figure(figsize=(12, 8))
sns.heatmap(df_cm, annot=True, fmt=".1f", annot_kws={'fontsize': 'xx-large'})
plt.savefig(path_name + 'confusion_matrix.pdf')
plt.close()

# ============================================================
# Save CSV
# ============================================================
csv_dir = f"plot_data/seed={seed}/CIFAR10_2layer/"
os.makedirs(csv_dir, exist_ok=True)
csv_filename = csv_dir + f'{kernel_init_type}.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Training Loss', 'Test Accuracy'])
    for epoch, (loss, acc) in enumerate(zip(loss_list, test_acc_list), start=1):
        writer.writerow([epoch, loss, acc])

print(f"训练指标已保存到 {csv_filename}")
