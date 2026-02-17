# XAI 解析プラットフォーム

講演・報告用のインタラクティブ XAI (Explainable AI) 解析プラットフォーム。
4 つの異なる問題設定に対して網羅的に XAI 手法を適用し、ブラウザで操作できるビューアで結果を閲覧できる。

## デモ

`viewer.html` をブラウザで開くだけで全結果をインタラクティブに閲覧可能（サーバー不要）。

## プロジェクト構成

```
XAI_no_xx/
├── requirements.txt          # Python 依存パッケージ
├── run_all.sh                # 全分析を一括実行
├── viewer.html               # インタラクティブ講演ビューア
│
├── problem1_housing/         # 表形式: 住宅価格予測 (回帰)
│   ├── 00_train_model.py     #   RF + GBT 学習
│   ├── 01_shap_analysis.py   #   SHAP 分析
│   ├── 02_lime_analysis.py   #   LIME 分析
│   ├── 03_permutation_analysis.py  # Permutation Importance
│   ├── 04_pdp_ice_analysis.py      # PDP / ICE
│   ├── 05_comparison.py      #   手法比較
│   └── results/              #   生成画像・中間ファイル
│
├── problem2_animals/         # 画像: CIFAR-10 動物分類
│   ├── 00_train_model.py     #   ResNet18 fine-tune
│   ├── 01_gradcam.py         #   Grad-CAM
│   ├── 02_integrated_gradients.py  # Integrated Gradients
│   ├── 03_occlusion.py       #   Occlusion Sensitivity
│   ├── 04_comparison.py      #   手法比較
│   └── results/
│
├── problem3_faces/           # 画像: LFW 顔識別
│   ├── 00_train_model.py     #   CNN 学習
│   ├── 01_gradcam.py         #   Grad-CAM
│   ├── 02_integrated_gradients.py  # Integrated Gradients
│   ├── 03_occlusion.py       #   Occlusion Sensitivity
│   ├── 04_comparison.py      #   手法比較
│   └── results/
│
├── problem4_power/           # 電力系統: 潮流異常検知 + 電圧安定性
│   ├── case1_power_flow/
│   │   └── generate_and_train.py   # 潮流パターン異常検知
│   ├── case2_voltage/
│   │   └── generate_and_train.py   # 電圧安定性限界予測
│   ├── xai_analysis.py       #   SHAP / LIME / PI / PDP
│   ├── report.md
│   └── results/
│
└── taxonomy/
    └── xai_taxonomy.md       # XAI 手法の体系的分類
```

## 各 Problem の概要

| Problem | テーマ | データ | モデル | XAI 手法 |
|---------|--------|--------|--------|----------|
| P1 | 住宅価格予測 (回帰) | California Housing (sklearn) | RF, GBT | SHAP, LIME, PI, PDP/ICE |
| P2 | 動物分類 (画像) | CIFAR-10 (torchvision) | ResNet18 | Grad-CAM, IG, Occlusion |
| P3 | 顔識別 (画像) | LFW (sklearn) | CNN (PyTorch) | Grad-CAM, IG, Occlusion |
| P4 | 電力系統 | 合成データ | RF, GBT | SHAP, LIME, PI, PDP |

### Problem 1: 住宅価格予測
California Housing データセットを用いた回帰問題。特徴量（収入中央値、築年数、人口など）の重要度を複数の XAI 手法で比較分析。

### Problem 2: CIFAR-10 動物分類
ImageNet 事前学習済み ResNet18 を CIFAR-10 で fine-tune。Grad-CAM / Integrated Gradients / Occlusion Sensitivity で「モデルが画像のどこを見ているか」を可視化。

### Problem 3: LFW 顔識別
LFW データセットで人物識別 CNN を学習。「AI は顔のどこを見て人物を判別しているか」を XAI で解析し、プライバシー・バイアスの議論につなげる。

### Problem 4: 電力系統
- **Case 1**: 潮流パターン異常検知 — 「どの母線・パラメータが異常判定の根拠か」
- **Case 2**: 電圧安定性限界予測 — 「電圧崩壊の主要因は何か」

## セットアップ

```bash
pip install -r requirements.txt
```

### 主な依存パッケージ
- numpy, pandas, scikit-learn, matplotlib, seaborn
- shap, lime
- torch, torchvision, captum
- Pillow, tqdm

## 実行方法

### 一括実行
```bash
bash run_all.sh
```

### 個別実行
```bash
# 例: Problem 1
python problem1_housing/00_train_model.py
python problem1_housing/01_shap_analysis.py
python problem1_housing/02_lime_analysis.py
python problem1_housing/03_permutation_analysis.py
python problem1_housing/04_pdp_ice_analysis.py
python problem1_housing/05_comparison.py
```

### 結果閲覧
```bash
open viewer.html
```

## viewer.html の機能

- **タブ切替**: Problem 1-4 + Taxonomy + 手法比較表
- **XAI 手法セレクタ**: 各 Problem で手法をクリックして結果を切替
- **サンプル切替**: 個別サンプルの説明を切り替えて表示
- **ライトボックス**: 画像クリックで拡大表示
- **ダークテーマ**: 講演・プロジェクタ向きの暗色テーマ
- **KaTeX 数式**: XAI 手法の数学的定義をインライン表示
- **レスポンシブ**: プロジェクタ・大画面対応

## 技術的な注意事項

- **MPS (Apple Metal)**: macOS Apple Silicon 環境では ResNet18 のトレーニングに MPS を自動使用
- **Captum + MPS**: Integrated Gradients / Occlusion は MPS の float64 非対応のため CPU にフォールバック
- **日本語フォント**: `Hiragino Sans` を使用（macOS 標準搭載）
- **特徴量名**: 全て日本語で表示
