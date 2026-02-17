"""
Problem 3: Occlusion Sensitivity による顔認識モデルの可視化
==========================================================
Captum の Occlusion を用いて、顔のどの領域を隠すと分類結果に
最も大きな影響を与えるかを可視化する。
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn

from captum.attr import Occlusion

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"デバイス: {device}")

# ---------------------------------------------------------------------------
# CNN モデル定義 (00_train_model.py と同一)
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


# ---------------------------------------------------------------------------
# データ・モデル読み込み
# ---------------------------------------------------------------------------
print("モデルとサンプル画像を読み込み中...")

with open(os.path.join(RESULTS_DIR, 'sample_images.pkl'), 'rb') as f:
    sample_data = pickle.load(f)

with open(os.path.join(RESULTS_DIR, 'test_data.pkl'), 'rb') as f:
    test_data = pickle.load(f)

target_names = sample_data['target_names']
n_classes = sample_data['n_classes']
sample_images = sample_data['images']

model = FaceCNN(n_classes).to(device)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'cnn_lfw.pth'),
                                  map_location=device, weights_only=True))
model.eval()

# ---------------------------------------------------------------------------
# Occlusion の準備
# ---------------------------------------------------------------------------
occlusion = Occlusion(model)

# ---------------------------------------------------------------------------
# サンプル準備: 最初5人 x 2サンプル = 最大10枚
# ---------------------------------------------------------------------------
n_people = min(5, n_classes)
samples_per_person = 2
all_samples = []  # (image, class_idx)

for cls_idx in range(n_people):
    imgs = sample_images[cls_idx]
    n_take = min(samples_per_person, len(imgs))
    for s in range(n_take):
        all_samples.append((imgs[s], cls_idx))

print(f"Occlusion 生成対象: {len(all_samples)} 枚")
print("(各サンプルについてスライディングウィンドウで計算するため時間がかかります)")

# ---------------------------------------------------------------------------
# Occlusion attribution を計算
# ---------------------------------------------------------------------------
results = []

for idx, (img, cls_idx) in enumerate(all_samples):
    print(f"  処理中: {idx + 1}/{len(all_samples)} ({target_names[cls_idx]})")

    input_t = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Occlusion 計算
    # sliding_window_shapes: (channels, height, width)
    # strides: (channels, height, width)
    attr = occlusion.attribute(
        input_t,
        target=cls_idx,
        sliding_window_shapes=(1, 6, 6),
        strides=(1, 3, 3),
        baselines=0.0,  # 黒で隠す
    )

    attr_np = attr.squeeze().cpu().detach().numpy()  # (H, W)

    # 予測
    with torch.no_grad():
        output = model(input_t)
        pred = output.argmax(dim=1).item()

    results.append({
        'image': img,
        'attribution': attr_np,
        'true_class': cls_idx,
        'pred_class': pred,
        'name': target_names[cls_idx],
    })

# ---------------------------------------------------------------------------
# グリッド画像の保存
# ---------------------------------------------------------------------------
n_samples = len(results)
fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples), dpi=150)
if n_samples == 1:
    axes = axes[np.newaxis, :]

for i, r in enumerate(results):
    attr = r['attribution']
    abs_max = max(abs(attr.min()), abs(attr.max()))
    if abs_max == 0:
        abs_max = 1.0
    norm_attr = attr / abs_max

    # 元画像
    axes[i, 0].imshow(r['image'], cmap='gray')
    axes[i, 0].set_title(f"元画像: {r['name'].split()[-1]}", fontsize=10)
    axes[i, 0].axis('off')

    # Occlusion attribution
    im = axes[i, 1].imshow(norm_attr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[i, 1].set_title('Occlusion Sensitivity', fontsize=10)
    axes[i, 1].axis('off')

    # ヒートマップオーバーレイ
    # 正の attribution = その領域を隠すと予測が低下 = 重要な領域
    pos_attr = np.clip(norm_attr, 0, 1)
    heatmap_color = cm.hot(pos_attr)[:, :, :3]
    img_rgb = np.stack([r['image']] * 3, axis=-1)
    overlay = 0.4 * heatmap_color + 0.6 * img_rgb
    overlay = np.clip(overlay, 0, 1)

    axes[i, 2].imshow(overlay)
    pred_name = target_names[r['pred_class']].split()[-1]
    axes[i, 2].set_title(f'オーバーレイ (予測: {pred_name})', fontsize=10)
    axes[i, 2].axis('off')

fig.suptitle('遮蔽感度分析 - Occlusion Sensitivity\n明るい領域=隠すと分類精度が低下する重要領域',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'occlusion_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("グリッド画像を保存: results/occlusion_grid.png")

# ---------------------------------------------------------------------------
# 個別の詳細画像保存 (最大5枚)
# ---------------------------------------------------------------------------
for i in range(min(5, n_samples)):
    r = results[i]
    attr = r['attribution']
    abs_max = max(abs(attr.min()), abs(attr.max()))
    if abs_max == 0:
        abs_max = 1.0
    norm_attr = attr / abs_max

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    axes[0].imshow(r['image'], cmap='gray')
    axes[0].set_title(f"元画像\n{r['name']}", fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(norm_attr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Occlusion Sensitivity\n(赤=隠すと精度低下)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 正の attribution のみの heat map
    pos_attr = np.clip(attr, 0, None)
    if pos_attr.max() > 0:
        pos_attr = pos_attr / pos_attr.max()
    im2 = axes[2].imshow(pos_attr, cmap='hot')
    axes[2].set_title('重要領域マップ\n(明るい=より重要)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f'Occlusion Sensitivity 詳細 - サンプル {i}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'occlusion_sample{i}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"個別画像を保存: results/occlusion_sample0.png ~ results/occlusion_sample{min(4, n_samples-1)}.png")

# ---------------------------------------------------------------------------
# Occlusion 結果を他のスクリプト用に保存
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, 'occlusion_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("Occlusion 結果を保存: results/occlusion_results.pkl")

print("\nOcclusion 解析完了!")
print("考察: Occlusion は顔の各領域を実際に隠して予測の変化を測定します。")
print("       目の周辺を隠すと最も予測精度が低下することが多く、")
print("       人物識別において目が最も重要な特徴であることを示唆します。")
