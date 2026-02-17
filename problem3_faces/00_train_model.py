"""
Problem 3: 顔認識モデル (LFW dataset + CNN) の訓練
====================================================
LFW (Labeled Faces in the Wild) データセットを用いて
顔識別 CNN を訓練し、テストデータ・サンプル画像を保存する。
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"デバイス: {device}")

# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------
print("LFW データセットをダウンロード中 (初回は時間がかかります)...")
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw.images  # (n_samples, h, w) - グレースケール
y = lfw.target
target_names = lfw.target_names
n_classes = len(target_names)
h, w = X.shape[1], X.shape[2]

print(f"画像サイズ: {h} x {w}")
print(f"クラス数: {n_classes}")
print(f"サンプル数: {len(X)}")
for i, name in enumerate(target_names):
    print(f"  {name}: {np.sum(y == i)} 枚")

# ---------------------------------------------------------------------------
# 前処理 & train/test 分割
# ---------------------------------------------------------------------------
# 0-1 正規化
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"訓練: {len(X_train)}, テスト: {len(X_test)}")

# PyTorch テンソル化 (N, 1, H, W)
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
y_test_t = torch.LongTensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------------
# CNN モデル定義
# ---------------------------------------------------------------------------
class FaceCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


model = FaceCNN(n_classes).to(device)
print(model)

# クラス重み（不均衡データ対策）
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights_t = torch.FloatTensor(class_weights).to(device)
print(f"クラス重み: {dict(zip(target_names, class_weights.round(2)))}")

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# 訓練ループ
# ---------------------------------------------------------------------------
print("\n訓練開始...")
train_losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

# ---------------------------------------------------------------------------
# テスト評価
# ---------------------------------------------------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = np.mean(all_preds == all_labels)
print(f"\nテスト精度: {accuracy:.4f}")
print("\n分類レポート:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# ---------------------------------------------------------------------------
# 混同行列の保存
# ---------------------------------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
# 短縮名を使用
short_names = [n.split()[-1] for n in target_names]
disp = ConfusionMatrixDisplay(cm, display_labels=short_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('顔認識 CNN - 混同行列', fontsize=16)
ax.set_xlabel('予測ラベル', fontsize=12)
ax.set_ylabel('真のラベル', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("混同行列を保存: results/confusion_matrix.png")

# ---------------------------------------------------------------------------
# クラスごとのサンプル画像保存
# ---------------------------------------------------------------------------
sample_images = {}
for cls_idx in range(n_classes):
    mask = y_test == cls_idx
    cls_images = X_test[mask]
    n_take = min(SAMPLES_PER_CLASS, len(cls_images))
    sample_images[cls_idx] = cls_images[:n_take]

with open(os.path.join(RESULTS_DIR, 'sample_images.pkl'), 'wb') as f:
    pickle.dump({
        'images': sample_images,
        'target_names': target_names,
        'n_classes': n_classes,
    }, f)
print("サンプル画像を保存: results/sample_images.pkl")

# ---------------------------------------------------------------------------
# サンプル顔画像グリッドの保存
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(n_classes, SAMPLES_PER_CLASS, figsize=(3 * SAMPLES_PER_CLASS, 3 * n_classes), dpi=150)
if n_classes == 1:
    axes = axes[np.newaxis, :]

for cls_idx in range(n_classes):
    for s in range(SAMPLES_PER_CLASS):
        ax = axes[cls_idx, s]
        if s < len(sample_images[cls_idx]):
            ax.imshow(sample_images[cls_idx][s], cmap='gray')
            if s == 0:
                ax.set_ylabel(target_names[cls_idx].split()[-1], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle('LFW データセット - 各人物のサンプル顔画像', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sample_faces.png'), dpi=150, bbox_inches='tight')
plt.close()
print("サンプル顔グリッドを保存: results/sample_faces.png")

# ---------------------------------------------------------------------------
# モデル・テストデータの保存
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'cnn_lfw.pth'))
print("モデルを保存: results/cnn_lfw.pth")

with open(os.path.join(RESULTS_DIR, 'test_data.pkl'), 'wb') as f:
    pickle.dump({
        'X_test': X_test,
        'y_test': y_test,
        'target_names': target_names,
        'n_classes': n_classes,
        'image_shape': (h, w),
    }, f)
print("テストデータを保存: results/test_data.pkl")

print("\n完了! すべての結果は results/ に保存されました。")
