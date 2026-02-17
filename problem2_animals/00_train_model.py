"""
00_train_model.py
CIFAR-10 データセットで ResNet18 をファインチューニングする。
- torchvision の ResNet18 (ImageNet事前学習済み) を使用
- 32x32 入力に対応するよう最初の畳み込み層を修正
- 最終全結合層を10クラスに変更
- 10エポック学習後、モデルと評価結果を保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

# ============================================================
# 設定
# ============================================================
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"使用デバイス: {device}")

# ============================================================
# データ前処理
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ============================================================
# データセット読み込み
# ============================================================
print("CIFAR-10 データセットを読み込み中...")
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

print(f"学習データ: {len(train_dataset)} 枚")
print(f"テストデータ: {len(test_dataset)} 枚")

# ============================================================
# モデル構築 (ResNet18 を CIFAR-10 用に修正)
# ============================================================
def build_resnet18_cifar10():
    """
    ResNet18 を CIFAR-10 (32x32) 向けに修正:
    1. conv1: 7x7 stride=2 -> 3x3 stride=1 (小さい入力に対応)
    2. maxpool を削除 (Identity に置換)
    3. fc: 1000 -> 10
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 最初の畳み込み層を 32x32 入力に適した構成に変更
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # MaxPool を削除 (解像度を維持)
    model.maxpool = nn.Identity()
    # 最終全結合層を10クラスに変更
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model

model = build_resnet18_cifar10()
model = model.to(device)
print(f"モデル構築完了: ResNet18 (CIFAR-10 用に修正)")

# ============================================================
# 学習
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\n学習開始 (エポック数: {EPOCHS}, 学習率: {LR})")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"エポック {epoch+1:2d}/{EPOCHS} | "
          f"損失: {avg_loss:.4f} | "
          f"学習精度: {train_acc:.2f}%")

# ============================================================
# テスト評価
# ============================================================
model.eval()
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

test_acc = 100.0 * correct / total
print(f"\nテスト精度: {test_acc:.2f}%")

# ============================================================
# モデル保存
# ============================================================
model_path = os.path.join(RESULTS_DIR, 'resnet18_cifar10.pth')
torch.save(model.state_dict(), model_path)
print(f"モデル保存: {model_path}")

# ============================================================
# 混同行列の保存
# ============================================================
cm = confusion_matrix(all_targets, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('CIFAR-10 混同行列 (ResNet18)', fontsize=14)
ax.set_xlabel('予測ラベル', fontsize=12)
ax.set_ylabel('正解ラベル', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"混同行列保存: {cm_path}")

# ============================================================
# サンプル画像の保存 (各クラス1枚ずつ、計10枚)
# ============================================================
# 正規化なしの生データでテストセットを再読み込み
raw_test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=False,
    transform=transforms.ToTensor()  # 正規化なし [0,1]
)

sample_images = []
sample_labels = []
found_classes = set()

for img, label in raw_test_dataset:
    if label not in found_classes:
        sample_images.append(img)  # Tensor [3, 32, 32], 値域 [0, 1]
        sample_labels.append(label)
        found_classes.add(label)
    if len(found_classes) == NUM_CLASSES:
        break

# ラベル順にソート
sorted_pairs = sorted(zip(sample_labels, sample_images), key=lambda x: x[0])
sample_labels = [p[0] for p in sorted_pairs]
sample_images = [p[1] for p in sorted_pairs]

sample_data = {
    'images': sample_images,       # list of Tensor [3, 32, 32], 値域 [0, 1]
    'labels': sample_labels,       # list of int
    'class_names': CLASS_NAMES,
}

sample_path = os.path.join(RESULTS_DIR, 'sample_images.pkl')
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
print(f"サンプル画像保存: {sample_path} ({len(sample_images)} 枚)")

# サンプル画像の確認用可視化
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = sample_images[i].permute(1, 2, 0).numpy()  # [H, W, 3]
    ax.imshow(img)
    ax.set_title(CLASS_NAMES[sample_labels[i]], fontsize=10)
    ax.axis('off')
fig.suptitle('CIFAR-10 サンプル画像 (各クラス1枚)', fontsize=14)
plt.tight_layout()
sample_vis_path = os.path.join(RESULTS_DIR, 'sample_images_preview.png')
plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"サンプル可視化保存: {sample_vis_path}")

print("\n学習・評価完了!")
