"""
04_comparison.py
3つの XAI 手法 (Grad-CAM, Integrated Gradients, Occlusion) の比較可視化
- 各手法を再計算して統一的に比較
- 5枚のサンプル画像に対して 4行 x 5列 の比較画像を生成
- 手法の特性サマリーも生成
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
from captum.attr import LayerGradCam, IntegratedGradients, Occlusion

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

# Captum の一部メソッドは MPS の float64 非対応のため CPU を使用
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
# 3手法の計算 (最初の5枚)
# ============================================================
NUM_SAMPLES = 5

# --- Grad-CAM ---
print("\nGrad-CAM 計算中...")
grad_cam_method = LayerGradCam(model, model.layer4)
gradcam_maps = []

for i in range(NUM_SAMPLES):
    input_tensor = normalize_image(raw_images[i]).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    attr = grad_cam_method.attribute(input_tensor, target=labels[i])
    attr = torch.relu(attr)
    attr_upsampled = torch.nn.functional.interpolate(
        attr, size=(32, 32), mode='bilinear', align_corners=False
    )
    attr_np = attr_upsampled.squeeze().detach().cpu().numpy()
    if attr_np.max() > 0:
        attr_np = attr_np / attr_np.max()
    gradcam_maps.append(attr_np)
    print(f"  [{i+1}/{NUM_SAMPLES}] Grad-CAM 完了")

# --- Integrated Gradients ---
print("\nIntegrated Gradients 計算中...")
ig_method = IntegratedGradients(model)
ig_maps = []

for i in range(NUM_SAMPLES):
    input_tensor = normalize_image(raw_images[i]).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    baseline = torch.zeros_like(input_tensor).to(device)
    attr = ig_method.attribute(
        input_tensor, baselines=baseline, target=labels[i],
        n_steps=50, internal_batch_size=50
    )
    attr_np = attr.squeeze().detach().cpu().numpy()
    attr_2d = np.sum(np.abs(attr_np), axis=0)
    if attr_2d.max() > 0:
        attr_2d = attr_2d / attr_2d.max()
    ig_maps.append(attr_2d)
    print(f"  [{i+1}/{NUM_SAMPLES}] IG 完了")

# --- Occlusion ---
print("\nOcclusion 計算中...")
occ_method = Occlusion(model)
occ_maps = []

for i in range(NUM_SAMPLES):
    input_tensor = normalize_image(raw_images[i]).unsqueeze(0).to(device)
    attr = occ_method.attribute(
        input_tensor, target=labels[i],
        sliding_window_shapes=(3, 4, 4), strides=(3, 2, 2), baselines=0
    )
    attr_np = attr.squeeze().detach().cpu().numpy()
    attr_2d = np.sum(np.abs(attr_np), axis=0)
    if attr_2d.max() > 0:
        attr_2d = attr_2d / attr_2d.max()
    occ_maps.append(attr_2d)
    print(f"  [{i+1}/{NUM_SAMPLES}] Occlusion 完了")

print("\n全手法の計算完了")

# ============================================================
# 可視化 1: 手法比較 (4行 x 5列)
# 行: 元画像, Grad-CAM, Integrated Gradients, Occlusion
# 列: 5つのサンプル
# ============================================================
fig, axes = plt.subplots(4, NUM_SAMPLES, figsize=(15, 12))

row_labels = ['元画像', 'Grad-CAM', 'Integrated\nGradients', 'Occlusion']
cmaps = [None, 'jet', 'hot', 'inferno']

for col in range(NUM_SAMPLES):
    original = raw_images[col].permute(1, 2, 0).numpy()
    class_name = CLASS_NAMES[labels[col]]

    # 行0: 元画像
    axes[0, col].imshow(original)
    axes[0, col].set_title(class_name, fontsize=12, fontweight='bold')
    axes[0, col].axis('off')

    # 行1: Grad-CAM (重畳)
    gc_colored = cm.jet(gradcam_maps[col])[:, :, :3]
    gc_overlay = 0.5 * gc_colored + 0.5 * original
    gc_overlay = np.clip(gc_overlay, 0, 1)
    axes[1, col].imshow(gc_overlay)
    axes[1, col].axis('off')

    # 行2: Integrated Gradients (重畳)
    ig_colored = cm.hot(ig_maps[col])[:, :, :3]
    ig_overlay = 0.5 * ig_colored + 0.5 * original
    ig_overlay = np.clip(ig_overlay, 0, 1)
    axes[2, col].imshow(ig_overlay)
    axes[2, col].axis('off')

    # 行3: Occlusion (重畳)
    occ_colored = cm.inferno(occ_maps[col])[:, :, :3]
    occ_overlay = 0.5 * occ_colored + 0.5 * original
    occ_overlay = np.clip(occ_overlay, 0, 1)
    axes[3, col].imshow(occ_overlay)
    axes[3, col].axis('off')

# 行ラベル
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=12, rotation=0,
                             labelpad=80, va='center')

fig.suptitle('XAI 手法比較: Grad-CAM vs Integrated Gradients vs Occlusion\n'
             '(ResNet18 / CIFAR-10)', fontsize=16, y=0.98)
plt.tight_layout()
comp_path = os.path.join(RESULTS_DIR, 'comparison_methods.png')
plt.savefig(comp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"手法比較画像保存: {comp_path}")

# ============================================================
# 可視化 2: 手法特性サマリー
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 左上: 各手法の概要テーブル ---
ax = axes[0, 0]
ax.axis('off')
table_data = [
    ['手法', '種別', '粒度', '計算コスト'],
    ['Grad-CAM', '勾配ベース\n(層ごと)', '粗い\n(特徴マップ\n解像度)', '低い'],
    ['Integrated\nGradients', '勾配ベース\n(ピクセル)', '細かい\n(入力解像度)', '中程度'],
    ['Occlusion', '摂動ベース', 'ウィンドウ\nサイズ依存', '高い'],
]
colors = [['#444444'] * 4] + [['#333333'] * 4] * 3
table = ax.table(cellText=table_data, cellColours=colors,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.2)
for key, cell in table.get_celld().items():
    cell.set_edgecolor('#666666')
    cell.set_text_props(color='white')
    if key[0] == 0:
        cell.set_facecolor('#555555')
        cell.set_text_props(fontweight='bold', color='white')
ax.set_title('各手法の特性比較', fontsize=13, fontweight='bold', pad=20)

# --- 右上: 解像度の違い ---
ax = axes[0, 1]
if len(raw_images) > 0:
    original = raw_images[0].permute(1, 2, 0).numpy()
    # Grad-CAM は粗い -> 高解像度化
    # IG / Occlusion はピクセルレベル
    ax.imshow(gradcam_maps[0], cmap='jet', interpolation='nearest')
    ax.set_title(f'Grad-CAM の特徴マップ解像度\n({CLASS_NAMES[labels[0]]})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

# --- 左下: IG の符号付き属性 ---
ax = axes[1, 0]
# IG の正負の寄与を示す
if len(raw_images) > 0:
    input_tensor = normalize_image(raw_images[0]).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    baseline = torch.zeros_like(input_tensor).to(device)
    attr = ig_method.attribute(
        input_tensor, baselines=baseline, target=labels[0],
        n_steps=50, internal_batch_size=50
    )
    attr_signed = attr.squeeze().detach().cpu().numpy()
    attr_signed_2d = np.sum(attr_signed, axis=0)  # 符号付き
    vmax = max(abs(attr_signed_2d.min()), abs(attr_signed_2d.max()))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(attr_signed_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'IG 符号付き属性 (赤=正, 青=負)\n({CLASS_NAMES[labels[0]]})',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

# --- 右下: 手法の利点・欠点 ---
ax = axes[1, 1]
ax.axis('off')
summary_text = (
    "Grad-CAM\n"
    "  + 計算が高速、直感的なヒートマップ\n"
    "  + クラス判別的な可視化が可能\n"
    "  - 空間解像度が粗い (最終畳み込み層依存)\n"
    "  - ピクセルレベルの詳細は不明\n\n"
    "Integrated Gradients\n"
    "  + ピクセルレベルの細かい属性付け\n"
    "  + 理論的保証 (完全性公理)\n"
    "  - ベースラインの選択に依存\n"
    "  - ノイズが多い場合がある\n\n"
    "Occlusion Sensitivity\n"
    "  + モデル非依存 (勾配不要)\n"
    "  + 直感的に理解しやすい\n"
    "  - 計算コストが高い\n"
    "  - ウィンドウサイズの選択に依存"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        color='#cccccc',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222',
                  edgecolor='#555555'))
ax.set_title('各手法の利点と欠点', fontsize=13, fontweight='bold', pad=20)

fig.suptitle('XAI 手法の特性サマリー', fontsize=16, y=1.01)
plt.tight_layout()
summary_path = os.path.join(RESULTS_DIR, 'comparison_summary.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"サマリー画像保存: {summary_path}")

print("\n手法比較可視化完了!")
