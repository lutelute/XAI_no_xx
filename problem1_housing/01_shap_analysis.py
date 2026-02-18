"""
01_shap_analysis.py
SHAP (SHapley Additive exPlanations) を用いた特徴量重要度分析

- TreeExplainer で RF / GB の SHAP 値を計算
- beeswarm plot, bar plot, dependence plot, waterfall plot を生成・保存
- shap_artifacts.pkl に SHAP 値を保存 (後続スクリプトで利用)
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150
SAMPLE_INDICES = [0, 1, 2]


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


def compute_shap_values(model, X, model_name):
    """TreeExplainer で SHAP 値を計算"""
    print(f"[INFO] {model_name} の SHAP 値を計算中...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    print(f"  SHAP 値の shape: {shap_values.values.shape}")
    return explainer, shap_values


def save_beeswarm(shap_values, feature_names, model_name, filename):
    """beeswarm (summary) plot を保存"""
    print(f"[INFO] Beeswarm plot 保存: {filename}")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(feature_names))
    plt.title(f'SHAP Beeswarm Plot - {model_name}', fontsize=14, pad=15)
    plt.xlabel('SHAP値 (モデル出力への影響)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close('all')


def save_bar(shap_values, feature_names, model_name, filename):
    """bar (feature importance) plot を保存"""
    print(f"[INFO] Bar plot 保存: {filename}")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, show=False, max_display=len(feature_names))
    plt.title(f'SHAP 特徴量重要度 (Bar) - {model_name}', fontsize=14, pad=15)
    plt.xlabel('平均 |SHAP値|', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close('all')


def save_dependence(shap_values, X, feature_names, model_name, filename):
    """最重要特徴量の dependence plot を保存"""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feature = feature_names[top_idx]
    # 2番目に重要な特徴量を色として使用
    sorted_idx = np.argsort(mean_abs)[::-1]
    color_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx
    color_feature = feature_names[color_idx]
    print(f"[INFO] Dependence plot 保存 (top feature: {top_feature}, color: {color_feature}): {filename}")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.dependence_plot(top_idx, shap_values.values, X,
                         feature_names=feature_names,
                         interaction_index=color_idx,
                         ax=ax, show=False)
    ax.set_title(f'SHAP Dependence Plot: {top_feature} - {model_name}', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close('all')


def save_waterfall(shap_values, sample_idx, model_name, filename):
    """個別サンプルの waterfall plot を保存"""
    print(f"[INFO] Waterfall plot (sample {sample_idx}) 保存: {filename}")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=10)
    plt.title(f'SHAP Waterfall - {model_name} (sample {sample_idx})', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close('all')


def main():
    rf, gb, test_data = load_artifacts()
    X_test = test_data['X_test']
    feature_names = test_data['feature_names']

    # テストデータのサブセットを使用 (高速化)
    n_samples = min(500, X_test.shape[0])
    X_subset = pd.DataFrame(X_test[:n_samples], columns=feature_names)
    print(f"[INFO] SHAP 計算に {n_samples} サンプルを使用")

    models = {'rf': ('Random Forest', rf), 'gb': ('Gradient Boosting', gb)}
    all_shap = {}

    for key, (label, model) in models.items():
        print(f"\n{'='*60}")
        print(f"  {label} の SHAP 分析")
        print(f"{'='*60}")

        explainer, shap_values = compute_shap_values(model, X_subset, label)
        all_shap[key] = {
            'shap_values': shap_values,
            'explainer': explainer,
        }

        # --- Beeswarm ---
        save_beeswarm(shap_values, feature_names, label, f'shap_summary_{key}.png')

        # --- Bar ---
        save_bar(shap_values, feature_names, label, f'shap_bar_{key}.png')

        # --- Dependence (top feature) ---
        if key == 'rf':
            save_dependence(shap_values, X_subset, feature_names, label, f'shap_dependence_{key}.png')

        # --- Waterfall (RF のみ 3 サンプル) ---
        if key == 'rf':
            for idx in SAMPLE_INDICES:
                save_waterfall(shap_values, idx, label, f'shap_waterfall_{key}_sample{idx}.png')

    # ---------- アーティファクト保存 ----------
    artifacts_path = os.path.join(RESULTS_DIR, 'shap_artifacts.pkl')
    save_data = {}
    for key in all_shap:
        sv = all_shap[key]['shap_values']
        save_data[key] = {
            'shap_values_array': sv.values,
            'base_values': sv.base_values,
            'data': sv.data,
            'feature_names': feature_names,
        }
    with open(artifacts_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n[INFO] SHAP アーティファクト保存: {artifacts_path}")

    print("\n[DONE] 01_shap_analysis.py 完了")


if __name__ == '__main__':
    main()
