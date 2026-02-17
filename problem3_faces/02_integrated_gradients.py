"""
Problem 3: Integrated Gradients による顔認識モデルの可視化
=========================================================
Captum の IntegratedGradients を用いて、顔認識 CNN における
ピクセルレベルの寄与度を可視化する。
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

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
# Integrated Gradients の準備
# ---------------------------------------------------------------------------
ig = IntegratedGradients(model)

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

print(f"Integrated Gradients 生成対象: {len(all_samples)} 枚")

# ---------------------------------------------------------------------------
# IG attribution を計算
# ---------------------------------------------------------------------------
results = []

for idx, (img, cls_idx) in enumerate(all_samples):
    print(f"  処理中: {idx + 1}/{len(all_samples)} ({target_names[cls_idx]})")

    input_t = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    input_t.requires_grad_(True)

    # ベースライン: ゼロ画像 (黒)
    baseline = torch.zeros_like(input_t).to(device)

    # IG 計算
    attr = ig.attribute(
        input_t,
        baselines=baseline,
        target=cls_idx,
        n_steps=50,
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
# 可視化ヘルパー
# ---------------------------------------------------------------------------
def visualize_ig_attribution(image, attribution, title=''):
    """IG attribution を赤-青カラーマップで可視化する。

    正の寄与 (予測を支持) → 赤
    負の寄与 (予測に反する) → 青
    """
    # 絶対値の最大でスケール
    abs_max = max(abs(attribution.min()), abs(attribution.max()))
    if abs_max == 0:
        abs_max = 1.0

    # 正規化: [-1, 1]
    norm_attr = attribution / abs_max

    return norm_attr


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

    # IG attribution (赤-青)
    im = axes[i, 1].imshow(norm_attr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[i, 1].set_title('Integrated Gradients', fontsize=10)
    axes[i, 1].axis('off')

    # オーバーレイ: 元画像 + attribution の絶対値
    abs_attr = np.abs(norm_attr)
    overlay = np.stack([r['image']] * 3, axis=-1)
    # 正の寄与を赤でオーバーレイ
    pos_mask = np.clip(norm_attr, 0, 1)
    neg_mask = np.clip(-norm_attr, 0, 1)
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * pos_mask, 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 0.5 * neg_mask, 0, 1)

    axes[i, 2].imshow(overlay)
    pred_name = target_names[r['pred_class']].split()[-1]
    axes[i, 2].set_title(f'オーバーレイ (予測: {pred_name})', fontsize=10)
    axes[i, 2].axis('off')

fig.suptitle('ピクセル寄与度分析 - Integrated Gradients\n赤=正の寄与 / 青=負の寄与',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ig_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("グリッド画像を保存: results/ig_grid.png")

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
    axes[1].set_title('Integrated Gradients\n(赤=正 / 青=負)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 絶対値ヒートマップ
    abs_attr = np.abs(norm_attr)
    im2 = axes[2].imshow(abs_attr, cmap='hot')
    axes[2].set_title('寄与度の絶対値\n(明るい=重要)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f'Integrated Gradients 詳細 - サンプル {i}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'ig_sample{i}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"個別画像を保存: results/ig_sample0.png ~ results/ig_sample{min(4, n_samples-1)}.png")

# ---------------------------------------------------------------------------
# IG 結果を他のスクリプト用に保存
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, 'ig_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("IG 結果を保存: results/ig_results.pkl")

print("\nIntegrated Gradients 解析完了!")
print("考察: IG はピクセルレベルの寄与度を示し、モデルがどの細部に")
print("       注目しているかを明らかにします。目や眉、口の輪郭などの")
print("       顔の特徴的な部分に高い寄与度が見られることが多いです。")
