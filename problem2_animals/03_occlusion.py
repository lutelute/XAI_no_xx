"""
03_occlusion.py
Occlusion Sensitivity による可視化
- 学習済み ResNet18 と保存済みサンプル画像を使用
- captum.attr.Occlusion を使用
- スライディングウィンドウ: (3, 4, 4), ストライド: (3, 2, 2)
- 10枚のサンプル画像それぞれに対して感度マップを生成
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from captum.attr import Occlusion

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

# ============================================================
# 設定
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

# Captum Occlusion は MPS の float64 非対応のため CPU を使用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"使用デバイス: {device}")

# ============================================================
# モデル読み込み
# ============================================================
def build_resnet18_cifar10():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

model = build_resnet18_cifar10()
model_path = os.path.join(RESULTS_DIR, 'resnet18_cifar10.pth')
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("モデル読み込み完了")

# ============================================================
# サンプル画像読み込み
# ============================================================
sample_path = os.path.join(RESULTS_DIR, 'sample_images.pkl')
with open(sample_path, 'rb') as f:
    sample_data = pickle.load(f)

raw_images = sample_data['images']     # list of Tensor [3, 32, 32], [0, 1]
labels = sample_data['labels']         # list of int
print(f"サンプル画像読み込み完了: {len(raw_images)} 枚")

# ============================================================
# 正規化関数
# ============================================================
def normalize_image(img_tensor):
    """[0,1] のテンソルを CIFAR-10 用に正規化"""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img_tensor - mean) / std

# ============================================================
# Occlusion 計算
# ============================================================
occlusion = Occlusion(model)

occ_results = []

for i, (raw_img, label) in enumerate(zip(raw_images, labels)):
    # 正規化してバッチ次元追加
    input_tensor = normalize_image(raw_img).unsqueeze(0).to(device)

    # Occlusion 計算
    attr = occlusion.attribute(
        input_tensor,
        target=label,
        sliding_window_shapes=(3, 4, 4),
        strides=(3, 2, 2),
        baselines=0,
    )

    # attr shape: [1, 3, 32, 32]
    attr_np = attr.squeeze().detach().cpu().numpy()  # [3, 32, 32]

    # チャネル方向に絶対値の合計を取って2Dマップに変換
    attr_2d = np.sum(np.abs(attr_np), axis=0)  # [32, 32]

    # 正規化 [0, 1]
    if attr_2d.max() > 0:
        attr_2d = attr_2d / attr_2d.max()

    occ_results.append(attr_2d)

    class_name = CLASS_NAMES[label]
    print(f"  [{i+1}/10] {class_name}: Occlusion 計算完了")

print("全サンプルの Occlusion 計算完了")

# ============================================================
# 重畳関数
# ============================================================
def create_occ_overlay(raw_img_tensor, attr_2d, alpha=0.5):
    """
    元画像に Occlusion 感度マップを重畳する
    """
    original = raw_img_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]
    colored_attr = cm.inferno(attr_2d)[:, :, :3]         # [H, W, 3]
    overlay = alpha * colored_attr + (1 - alpha) * original
    overlay = np.clip(overlay, 0, 1)
    return original, colored_attr, overlay

# ============================================================
# 可視化 1: 2x5 グリッド (全10サンプル)
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for i, ax in enumerate(axes.flat):
    original, attr_colored, overlay = create_occ_overlay(
        raw_images[i], occ_results[i])
    ax.imshow(overlay)
    ax.set_title(f"{CLASS_NAMES[labels[i]]}", fontsize=11)
    ax.axis('off')

fig.suptitle('Occlusion Sensitivity マップ (ResNet18 / CIFAR-10)', fontsize=16, y=0.98)
plt.tight_layout()
grid_path = os.path.join(RESULTS_DIR, 'occlusion_grid.png')
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"グリッド画像保存: {grid_path}")

# ============================================================
# 可視化 2: 個別詳細画像 (最初の5枚)
# ============================================================
for i in range(5):
    original, attr_colored, overlay = create_occ_overlay(
        raw_images[i], occ_results[i])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(original)
    axes[0].set_title('元画像', fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(occ_results[i], cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title('Occlusion 感度マップ', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title('重畳表示', fontsize=12)
    axes[2].axis('off')

    class_name = CLASS_NAMES[labels[i]]
    fig.suptitle(f'Occlusion Sensitivity 詳細: {class_name} (サンプル {i})',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    detail_path = os.path.join(RESULTS_DIR, f'occlusion_sample{i}.png')
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"詳細画像保存: {detail_path}")

# ============================================================
# Occlusion結果の保存 (比較用)
# ============================================================
occ_save_data = {
    'attr_2d': occ_results,
    'labels': labels,
}
occ_save_path = os.path.join(RESULTS_DIR, 'occlusion_results.pkl')
with open(occ_save_path, 'wb') as f:
    pickle.dump(occ_save_data, f)
print(f"Occlusion結果保存: {occ_save_path}")

print("\nOcclusion Sensitivity 可視化完了!")
