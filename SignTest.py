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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import csv

# ============================================================
# Args & Seed
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--seed", type=int, required=True)
ap.add_argument("-t", "--type",  type=int, required=True)
args = ap.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

type_ = ['Random', 'ClassMean', 'PCA', 'KMeans']
kernel_init_type = type_[args.type]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_num   = 500
target_size = 64
batch_size  = 64
in_channels = 3          # RGB
conv1_k = 32   # kernel size  → output spatial: (64-32)/16+1 = 3
conv1_s = 16   # stride
conv2_k = 3    # conv2 kernel, matches the 3×3 feature map

path_name = f'output_img/seed={seed}/SIGN_2layer_FMinit/{kernel_init_type}/'
os.makedirs(path_name, exist_ok=True)

loss_list     = []
test_acc_list = []

# ============================================================
# Data  (ImageFolder, same paths as original)
# ============================================================
transform = transforms.Compose([
    torchvision.transforms.Resize(target_size),
    transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda s: s.view(in_channels, target_size, target_size))
])

train_dataset = torchvision.datasets.folder.ImageFolder(
    root='../data/sign_language_mnist/train_10', transform=transform)
test_dataset  = torchvision.datasets.folder.ImageFolder(
    root='../data/sign_language_mnist/test_10',  transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True,  num_workers=0)
testloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size,
                                           shuffle=False, num_workers=0)

classes     = ('C', 'E', 'I', 'K', 'L', 'O', 'P', 'Q', 'X', 'Y')
num_classes = len(classes)

print('训练集图片总数：', len(train_dataset))
print('测试集图片总数：', len(test_dataset))

# ============================================================
# Network
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1: 3 → 90 channels, kernel 32×32, stride 16
        # input 64×64 → output 3×3
        self.conv1 = nn.Conv2d(in_channels, conv2_k * conv2_k * num_classes,
                               kernel_size=conv1_k, stride=conv1_s, bias=True)
        # conv2: 90 → 10, kernel 3×3
        # input 3×3 → output 1×1
        self.conv2 = nn.Conv2d(conv2_k * conv2_k * num_classes, num_classes,
                               kernel_size=conv2_k, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, 90, 3, 3]
        x = self.conv2(x)           # [B, 10, 1, 1]
        return x.squeeze()          # [B, 10]


# ============================================================
# Layer-1 initialisation  (RGB version)
# ============================================================
def align_templates_to_class_means(kmeans_centers, class_means):
    K = kmeans_centers.shape[0]
    dists = np.array([[np.linalg.norm(kmeans_centers[i] - class_means[j])
                       for j in range(K)] for i in range(K)])
    used_k, used_c = set(), set()
    mapping = {}
    for _ in range(K):
        tmp = dists.copy()
        tmp[list(used_k), :] = np.inf
        tmp[:, list(used_c)] = np.inf
        k, c = np.unravel_index(tmp.argmin(), tmp.shape)
        used_k.add(int(k)); used_c.add(int(c)); mapping[int(c)] = int(k)
    ordered = np.zeros_like(kmeans_centers)
    for c in range(K):
        ordered[c] = kmeans_centers[mapping[c]]
    return ordered


def build_classmean_layer1(trainloader, num_classes, conv2_k, conv1_k, conv1_s, channels=3):
    """
    Per-(class, spatial-position) mean patch → [90, 3, 32, 32]
    RGB 版本：对每个 patch 保留全部 3 个通道。
    """
    cord_list = [i * conv1_s for i in range(conv2_k)]
    total     = num_classes * conv2_k * conv2_k
    W_sum = torch.zeros(total, channels, conv1_k, conv1_k)
    W_cnt = torch.zeros(total)

    for imgs, labels in trainloader:
        # imgs: [B, 3, 64, 64]
        for i in range(len(labels)):
            for x in range(conv2_k):
                for y in range(conv2_k):
                    kid = int(labels[i]) * conv2_k * conv2_k + x * conv2_k + y
                    xs, ys = cord_list[x], cord_list[y]
                    # 保留全部通道 [3, 32, 32]
                    W_sum[kid] += imgs[i, :, xs:xs+conv1_k, ys:ys+conv1_k]
                    W_cnt[kid] += 1

    W = W_sum / W_cnt.view(-1, 1, 1, 1)   # [90, 3, 32, 32]
    # zero-mean + L2 norm (per filter, across spatial dims)
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = W.view(W.size(0), -1).norm(dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)
    return W


# ============================================================
# Layer-2 data-driven initialisation  (core new idea)
# ============================================================
def collect_feature_maps(net, trainloader, device):
    """
    Freeze conv1, pass entire training set through conv1+ReLU.

    Returns
    -------
    feats  : np.ndarray  [N, 90*3*3]  (= [N, 810])
    labels : np.ndarray  [N]
    """
    net.eval()
    for p in net.conv1.parameters():
        p.requires_grad_(False)

    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbs in trainloader:
            imgs = imgs.to(device)
            fm = F.relu(net.conv1(imgs))           # [B, 90, 3, 3]
            feats_list.append(fm.view(fm.size(0), -1).cpu().numpy())
            labels_list.append(lbs.numpy())

    for p in net.conv1.parameters():
        p.requires_grad_(True)
    net.train()

    return (np.concatenate(feats_list,  axis=0),
            np.concatenate(labels_list, axis=0))


def build_layer2_init(net, trainloader, device, method, num_classes, seed):
    """
    Collect feature maps → compute prototypes → reshape to [10, 90, 3, 3].
    """
    print(f"[Layer2 init] collecting feature maps (conv1 frozen) ...")
    feats, labels = collect_feature_maps(net, trainloader, device)

    D = feats.shape[1]   # 90*3*3 = 810

    if method == 'ClassMean':
        print("[Layer2 init] ClassMean on feature maps ...")
        W = np.zeros((num_classes, D), dtype=np.float32)
        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            W[c] = feats[idx].mean(axis=0)

    elif method == 'PCA':
        print("[Layer2 init] PCA (top-1 PC per class) on feature maps ...")
        W = np.zeros((num_classes, D), dtype=np.float32)
        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            Xc = feats[idx]
            mu = Xc.mean(axis=0)
            pca = PCA(n_components=1, random_state=seed)
            pca.fit(Xc - mu)
            pc = pca.components_[0]
            if pc.dot(mu) < 0:
                pc = -pc
            W[c] = pc

    elif method == 'KMeans':
        print("[Layer2 init] KMeans (unsupervised, k=10) on feature maps ...")
        km = KMeans(n_clusters=num_classes, n_init=10, random_state=seed)
        km.fit(feats)
        centers     = km.cluster_centers_
        class_means = np.array([feats[labels == c].mean(axis=0)
                                 for c in range(num_classes)])
        W = align_templates_to_class_means(centers, class_means)

    else:
        raise ValueError(f"Unknown method: {method}")

    # reshape → [10, 90, 3, 3]
    C_in = net.conv2.in_channels
    K    = net.conv2.kernel_size[0]
    W = torch.tensor(W.reshape(num_classes, C_in, K, K), dtype=torch.float32)

    # zero-mean + L2 normalise
    W = W - W.mean(dim=(2, 3), keepdim=True)
    norms = W.view(W.size(0), -1).norm(dim=1, keepdim=True).clamp_min(1e-8)
    W = W / norms.view(-1, 1, 1, 1)

    print(f"[Layer2 init] done. W shape: {W.shape}")
    return W


# ============================================================
# Build layer-1 ClassMean templates & initialise network
# ============================================================
print("Building layer-1 ClassMean templates ...")
W_layer1 = build_classmean_layer1(trainloader, num_classes, conv2_k,
                                   conv1_k, conv1_s, channels=in_channels)

net = Net().to(device)
tbwriter = SummaryWriter(f'logs/SIGN_2layer_FMinit_{kernel_init_type}_seed{seed}')

if kernel_init_type == 'Random':
    print("Using random initialisation for both layers.")

else:
    # Step 1: assign layer-1 weights
    net.conv1.weight.data = W_layer1.to(device)
    net.conv1.bias.data.zero_()

    # Step 2: data-driven layer-2 init
    W_layer2 = build_layer2_init(net, trainloader, device,
                                  method=kernel_init_type,
                                  num_classes=num_classes,
                                  seed=seed)
    net.conv2.weight.data = W_layer2.to(device)
    net.conv2.bias.data.zero_()

print(net)
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

# ============================================================
# Training
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

cnt = 0
for epoch in range(epoch_num):
    net.train()
    running_loss  = 0.0
    count_batches = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tbwriter.add_scalar('Training_loss', loss.item(), cnt)
        cnt += 1
        count_batches += 1
        if i % 5 == 4:
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/5:.3f}')
            running_loss = 0.0

    # epoch-level test accuracy
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbs in testloader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            _, pred = torch.max(net(imgs), 1)
            total   += lbs.size(0)
            correct += (pred == lbs).sum().item()
    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)
    loss_list.append(running_loss / max(count_batches, 1))
    print(f'Epoch {epoch+1}: test acc = {test_acc:.2f}%')

print('Finished Training')

# ============================================================
# Evaluation
# ============================================================
net.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for imgs, lbs in testloader:
        imgs = imgs.to(device)
        out  = torch.max(net(imgs), 1)[1].cpu().numpy()
        y_pred.extend(out)
        y_true.extend(lbs.numpy())

# Per-class accuracy
correct_pred = {c: 0 for c in classes}
total_pred   = {c: 0 for c in classes}
for t, p in zip(y_true, y_pred):
    total_pred[classes[t]] += 1
    if t == p:
        correct_pred[classes[t]] += 1
for c in classes:
    print(f'Accuracy for class: {c:5s} is {100*correct_pred[c]/total_pred[c]:.1f} %')

# Confusion matrix
cf = confusion_matrix(y_true, y_pred, normalize='true') * 100
df_cm = pd.DataFrame(cf, index=classes, columns=classes)
plt.figure(figsize=(12, 8))
sns.heatmap(df_cm, annot=True, fmt='.1f', annot_kws={'fontsize': 'xx-large'})
plt.savefig(path_name + 'confusion_matrix.pdf')
plt.close()

# ============================================================
# Save CSV
# ============================================================
csv_dir = f'plot_data/seed={seed}/SIGN_2layer_FMinit/'
os.makedirs(csv_dir, exist_ok=True)
csv_path = csv_dir + f'{kernel_init_type}.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Epoch', 'Training Loss', 'Test Accuracy'])
    for ep, (l, a) in enumerate(zip(loss_list, test_acc_list), 1):
        w.writerow([ep, l, a])
print(f"Saved to {csv_path}")