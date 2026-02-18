"""
00_train_model.py
MNIST データセットで CNN を学習する。
- torchvision の MNIST (手書き数字 28x28 グレースケール) を使用
- 3層の畳み込み + 全結合層による分類
- 10エポック学習後、モデルと評価結果を保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
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

CLASS_NAMES = [str(i) for i in range(10)]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"使用デバイス: {device}")

# ============================================================
# モデル定義
# ============================================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 28x28
        x = self.pool(x)           # 14x14
        x = F.relu(self.conv2(x))  # 14x14
        x = self.pool(x)           # 7x7
        x = F.relu(self.conv3(x))  # 7x7
        x = self.pool(x)           # 3x3
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# データ前処理
# ============================================================
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# ============================================================
# データセット読み込み
# ============================================================
print("MNIST データセットを読み込み中...")
train_dataset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

print(f"学習データ: {len(train_dataset)} 枚")
print(f"テストデータ: {len(test_dataset)} 枚")

# ============================================================
# モデル構築
# ============================================================
model = MNISTNet()
model = model.to(device)
print(f"モデル構築完了: MNISTNet (3層CNN)")

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
model_path = os.path.join(RESULTS_DIR, 'mnist_cnn.pth')
torch.save(model.state_dict(), model_path)
print(f"モデル保存: {model_path}")

# ============================================================
# 混同行列の保存
# ============================================================
cm = confusion_matrix(all_targets, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('MNIST 混同行列 (CNN)', fontsize=14)
ax.set_xlabel('予測ラベル', fontsize=12)
ax.set_ylabel('正解ラベル', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"混同行列保存: {cm_path}")

# ============================================================
# サンプル画像の保存 (各クラス2枚ずつ、計20枚)
# ============================================================
# 正規化なしの生データでテストセットを再読み込み
raw_test_dataset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=False,
    transform=transforms.ToTensor()  # 正規化なし [0,1]
)

# 各クラス2枚ずつ収集
samples_per_class = 2
sample_images = {i: [] for i in range(NUM_CLASSES)}
sample_labels_dict = {i: [] for i in range(NUM_CLASSES)}

for img, label in raw_test_dataset:
    if len(sample_images[label]) < samples_per_class:
        sample_images[label].append(img)
        sample_labels_dict[label].append(label)
    if all(len(v) == samples_per_class for v in sample_images.values()):
        break

# フラットなリストに変換 (クラス順)
flat_images = []
flat_labels = []
for cls in range(NUM_CLASSES):
    for img, lbl in zip(sample_images[cls], sample_labels_dict[cls]):
        flat_images.append(img)
        flat_labels.append(lbl)

# XAI用サンプル (各クラス1枚目のみ、計10枚)
xai_images = []
xai_labels = []
for cls in range(NUM_CLASSES):
    xai_images.append(sample_images[cls][0])
    xai_labels.append(cls)

sample_data = {
    'images': xai_images,          # list of Tensor [1, 28, 28], 値域 [0, 1]
    'labels': xai_labels,          # list of int
    'class_names': CLASS_NAMES,
}

sample_path = os.path.join(RESULTS_DIR, 'sample_images.pkl')
with open(sample_path, 'wb') as f:
    pickle.dump(sample_data, f)
print(f"サンプル画像保存: {sample_path} ({len(xai_images)} 枚)")

# サンプル画像の確認用可視化 (各クラス2枚、4行x5列)
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for cls in range(NUM_CLASSES):
    for j in range(samples_per_class):
        row = (cls // 5) * 2 + j
        col = cls % 5
        img = sample_images[cls][j].squeeze().numpy()  # [28, 28]
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'数字: {cls}', fontsize=10)
        axes[row, col].axis('off')

fig.suptitle('MNIST サンプル画像 (各数字2枚)', fontsize=14)
plt.tight_layout()
sample_vis_path = os.path.join(RESULTS_DIR, 'sample_digits.png')
plt.savefig(sample_vis_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"サンプル可視化保存: {sample_vis_path}")

print("\n学習・評価完了!")
