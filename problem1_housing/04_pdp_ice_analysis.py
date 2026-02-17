"""
04_pdp_ice_analysis.py
PDP (Partial Dependence Plot) と ICE (Individual Conditional Expectation) 分析

- sklearn.inspection.PartialDependenceDisplay を使用
- 上位 4 特徴量の PDP / ICE を 2x2 グリッドで表示
- 上位 2 特徴量ペアの 2D PDP を生成
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150
TOP_N = 4  # PDP/ICE で表示する上位特徴量数


def load_artifacts():
    """保存済みモデルとテストデータを読み込む"""
    print("[INFO] モデルとテストデータを読み込み中...")
    with open(os.path.join(RESULTS_DIR, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'gb_model.pkl'), 'rb') as f:
        gb = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    print("  読み込み完了")
    return rf, gb, test_data


def get_top_features(model, feature_names, n=TOP_N):
    """モデルの feature_importances_ から上位 n 特徴量のインデックスを取得"""
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    top_names = [feature_names[i] for i in top_idx]
    print(f"  上位 {n} 特徴量: {top_names}")
    return top_idx.tolist(), top_names


def save_pdp(model, X_test, feature_names, top_indices, top_names, model_name, filename):
    """PDP (Partial Dependence Plot) を 2x2 グリッドで保存"""
    print(f"[INFO] PDP 保存: {filename}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    # サブセットで高速化
    n_samples = min(500, X_test.shape[0])
    X_subset = X_test[:n_samples]

    display = PartialDependenceDisplay.from_estimator(
        model,
        X_subset,
        features=top_indices,
        feature_names=feature_names,
        kind='average',
        ax=axes_flat,
        line_kw={'color': '#00bfff', 'linewidth': 2},
        pd_line_kw={'color': '#00bfff', 'linewidth': 2},
    )

    for i, ax in enumerate(axes_flat):
        if i < len(top_names):
            ax.set_title(f'PDP: {top_names[i]}', fontsize=12)
            ax.set_xlabel(top_names[i], fontsize=10)
            ax.set_ylabel('部分依存 (Partial Dependence)', fontsize=10)

    fig.suptitle(f'Partial Dependence Plots - {model_name}', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def save_ice(model, X_test, feature_names, top_indices, top_names, model_name, filename):
    """ICE (Individual Conditional Expectation) plots を 2x2 グリッドで保存"""
    print(f"[INFO] ICE 保存: {filename}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()

    # ICE は計算量が多いのでサブセット
    n_samples = min(200, X_test.shape[0])
    X_subset = X_test[:n_samples]

    display = PartialDependenceDisplay.from_estimator(
        model,
        X_subset,
        features=top_indices,
        feature_names=feature_names,
        kind='both',  # PDP + ICE の両方
        ax=axes_flat,
        ice_lines_kw={'color': '#00bfff', 'alpha': 0.08, 'linewidth': 0.5},
        pd_line_kw={'color': '#ff6347', 'linewidth': 2.5},
    )

    for i, ax in enumerate(axes_flat):
        if i < len(top_names):
            ax.set_title(f'ICE + PDP: {top_names[i]}', fontsize=12)
            ax.set_xlabel(top_names[i], fontsize=10)
            ax.set_ylabel('予測値', fontsize=10)

    fig.suptitle(f'ICE + PDP Plots - {model_name}', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def save_pdp_2d(model, X_test, feature_names, top_indices, top_names, model_name, filename):
    """2D PDP (特徴量ペアの交互作用) を保存"""
    print(f"[INFO] 2D PDP 保存: {filename}")

    # 上位 2 特徴量のペア
    pairs = [(top_indices[0], top_indices[1])]
    if len(top_indices) >= 4:
        pairs.append((top_indices[2], top_indices[3]))

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 7))
    if n_pairs == 1:
        axes = [axes]

    # サブセット
    n_samples = min(300, X_test.shape[0])
    X_subset = X_test[:n_samples]

    for idx, (feat_pair, ax) in enumerate(zip(pairs, axes)):
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_subset,
            features=[feat_pair],
            feature_names=feature_names,
            kind='average',
            ax=ax,
        )
        f1_name = feature_names[feat_pair[0]]
        f2_name = feature_names[feat_pair[1]]
        ax.set_title(f'2D PDP: {f1_name} x {f2_name}', fontsize=12)

    fig.suptitle(f'2D Partial Dependence - {model_name}', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    rf, gb, test_data = load_artifacts()
    X_test = test_data['X_test']
    feature_names = test_data['feature_names']

    models = {'rf': ('Random Forest', rf), 'gb': ('Gradient Boosting', gb)}

    for key, (label, model) in models.items():
        print(f"\n{'='*60}")
        print(f"  {label} の PDP / ICE 分析")
        print(f"{'='*60}")

        top_indices, top_names = get_top_features(model, feature_names, n=TOP_N)

        # PDP
        save_pdp(model, X_test, feature_names, top_indices, top_names, label, f'pdp_{key}.png')

        # ICE
        save_ice(model, X_test, feature_names, top_indices, top_names, label, f'ice_{key}.png')

        # 2D PDP (RF のみ)
        if key == 'rf':
            save_pdp_2d(model, X_test, feature_names, top_indices, top_names, label, f'pdp_2d_{key}.png')

    print("\n[DONE] 04_pdp_ice_analysis.py 完了")


if __name__ == '__main__':
    main()
