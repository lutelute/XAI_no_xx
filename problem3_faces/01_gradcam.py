"""
Problem 3: Grad-CAM による顔認識モデルの可視化
===============================================
CNN の最終畳み込み層 (128フィルタ) に対して Grad-CAM を適用し、
顔のどの領域にモデルが注目しているかを可視化する。
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
import torch.nn.functional as F

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
# Grad-CAM 実装
# ---------------------------------------------------------------------------
class GradCAM:
    """指定した畳み込み層に対する Grad-CAM を計算する。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # フック登録
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Grad-CAM ヒートマップを生成する。

        Parameters
        ----------
        input_tensor : torch.Tensor  (1, 1, H, W)
        target_class : int or None (Noneなら予測クラスを使用)

        Returns
        -------
        heatmap : np.ndarray (H, W), 0-1 正規化済み
        predicted_class : int
        """
        self.model.eval()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # GAP → 重み
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # 入力画像サイズにリサイズ
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = np.array(
            __import__('PIL').Image.fromarray(cam).resize((w, h),
            __import__('PIL').Image.BILINEAR)
        )

        # 0-1 正規化
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class


def overlay_heatmap(image, heatmap, alpha=0.5):
    """グレースケール画像に Grad-CAM ヒートマップをオーバーレイする。"""
    # image: (H, W) 0-1
    # heatmap: (H, W) 0-1
    colored_heatmap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3) RGB
    # グレースケール画像を3チャネルに拡張
    img_rgb = np.stack([image] * 3, axis=-1)
    overlay = alpha * colored_heatmap + (1 - alpha) * img_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay


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

# Grad-CAM: 最終畳み込み層 (conv3 の Conv2d) をターゲットにする
# conv3 は Sequential(Conv2d, ReLU, AdaptiveAvgPool2d) なので [0] が Conv2d
target_layer = model.conv3[0]
grad_cam = GradCAM(model, target_layer)

# ---------------------------------------------------------------------------
# サンプル準備: 最初5人 × 2サンプル = 最大10枚
# ---------------------------------------------------------------------------
n_people = min(5, n_classes)
samples_per_person = 2
all_samples = []  # (image, class_idx)

for cls_idx in range(n_people):
    imgs = sample_images[cls_idx]
    n_take = min(samples_per_person, len(imgs))
    for s in range(n_take):
        all_samples.append((imgs[s], cls_idx))

print(f"Grad-CAM 生成対象: {len(all_samples)} 枚")

# ---------------------------------------------------------------------------
# 全サンプルの Grad-CAM を生成
# ---------------------------------------------------------------------------
results = []
for img, cls_idx in all_samples:
    input_t = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    heatmap, pred = grad_cam.generate(input_t, target_class=cls_idx)
    overlay = overlay_heatmap(img, heatmap, alpha=0.5)
    results.append({
        'image': img,
        'heatmap': heatmap,
        'overlay': overlay,
        'true_class': cls_idx,
        'pred_class': pred,
        'name': target_names[cls_idx],
    })

# ---------------------------------------------------------------------------
# グリッド画像の保存
# ---------------------------------------------------------------------------
n_samples = len(results)
fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples), dpi=150)
if n_samples == 1:
    axes = axes[np.newaxis, :]

for i, r in enumerate(results):
    # 元画像
    axes[i, 0].imshow(r['image'], cmap='gray')
    axes[i, 0].set_title(f"元画像: {r['name'].split()[-1]}", fontsize=10)
    axes[i, 0].axis('off')

    # ヒートマップ
    axes[i, 1].imshow(r['heatmap'], cmap='jet')
    axes[i, 1].set_title('Grad-CAM ヒートマップ', fontsize=10)
    axes[i, 1].axis('off')

    # オーバーレイ
    axes[i, 2].imshow(r['overlay'])
    pred_name = target_names[r['pred_class']].split()[-1]
    axes[i, 2].set_title(f'オーバーレイ (予測: {pred_name})', fontsize=10)
    axes[i, 2].axis('off')

fig.suptitle('顔認識AIの注目領域 - Grad-CAM', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gradcam_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("グリッド画像を保存: results/gradcam_grid.png")

# ---------------------------------------------------------------------------
# 個別の詳細画像保存 (最大5枚)
# ---------------------------------------------------------------------------
for i in range(min(5, n_samples)):
    r = results[i]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    axes[0].imshow(r['image'], cmap='gray')
    axes[0].set_title(f"元画像\n{r['name']}", fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(r['heatmap'], cmap='jet')
    axes[1].set_title('Grad-CAM ヒートマップ', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(r['overlay'])
    pred_name = target_names[r['pred_class']].split()[-1]
    axes[2].set_title(f'オーバーレイ\n予測: {pred_name}', fontsize=12)
    axes[2].axis('off')

    fig.suptitle(f'Grad-CAM 詳細 - サンプル {i}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'gradcam_sample{i}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"個別画像を保存: results/gradcam_sample0.png ~ results/gradcam_sample{min(4, n_samples-1)}.png")

# ---------------------------------------------------------------------------
# Grad-CAM 結果を他のスクリプト用に保存
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, 'gradcam_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("Grad-CAM 結果を保存: results/gradcam_results.pkl")

print("\nGrad-CAM 解析完了!")
print("考察: Grad-CAM は顔のどの領域をモデルが重視しているかを示します。")
print("       目、鼻、口の周辺や髪型の特徴的な部分が強調される傾向があります。")
