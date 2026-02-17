"""
03_permutation_analysis.py
Permutation Importance (置換重要度) 分析

- sklearn.inspection.permutation_importance を用いて
  RF / GB の特徴量重要度を計算
- 個別の棒グラフ、モデル比較図を保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150
N_REPEATS = 10
RANDOM_STATE = 42


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


def compute_pi(model, X_test, y_test, model_name):
    """Permutation Importance を計算"""
    print(f"[INFO] {model_name} の Permutation Importance を計算中 (n_repeats={N_REPEATS})...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
    )
    print(f"  計算完了")
    return result


def save_pi_bar(pi_result, feature_names, model_name, filename):
    """Permutation Importance の棒グラフを保存"""
    print(f"[INFO] PI 棒グラフ保存: {filename}")

    mean_imp = pi_result.importances_mean
    std_imp = pi_result.importances_std

    # 重要度の降順にソート
    sorted_idx = np.argsort(mean_imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(feature_names))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_mean = mean_imp[sorted_idx]
    sorted_std = std_imp[sorted_idx]

    bars = ax.barh(
        y_pos, sorted_mean,
        xerr=sorted_std,
        color='#00bfff', edgecolor='white', linewidth=0.5,
        height=0.7, capsize=3,
        error_kw={'ecolor': '#ff6347', 'linewidth': 1.5}
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Permutation Importance (MSE低下量)', fontsize=12)
    ax.set_title(f'Permutation Importance - {model_name}', fontsize=14, pad=15)
    ax.invert_yaxis()
    ax.axvline(x=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def save_comparison(pi_rf, pi_gb, feature_names, filename):
    """RF vs GB の Permutation Importance を比較する図を保存"""
    print(f"[INFO] 比較図保存: {filename}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    data_list = [
        (pi_rf, 'Random Forest', '#00bfff'),
        (pi_gb, 'Gradient Boosting', '#ff9f43'),
    ]

    # RF の重要度降順でソート (両モデル共通のソート順)
    sorted_idx = np.argsort(pi_rf.importances_mean)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    y_pos = np.arange(len(feature_names))

    for ax, (pi_result, label, color) in zip(axes, data_list):
        sorted_mean = pi_result.importances_mean[sorted_idx]
        sorted_std = pi_result.importances_std[sorted_idx]

        ax.barh(
            y_pos, sorted_mean,
            xerr=sorted_std,
            color=color, edgecolor='white', linewidth=0.5,
            height=0.7, capsize=3,
            error_kw={'ecolor': '#ff6347', 'linewidth': 1.2}
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=11)
        ax.set_xlabel('Permutation Importance', fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.invert_yaxis()
        ax.axvline(x=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    fig.suptitle('Permutation Importance: RF vs GB 比較', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    rf, gb, test_data = load_artifacts()
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    feature_names = test_data['feature_names']

    # ---------- 計算 ----------
    pi_rf = compute_pi(rf, X_test, y_test, 'Random Forest')
    pi_gb = compute_pi(gb, X_test, y_test, 'Gradient Boosting')

    # ---------- 個別棒グラフ ----------
    save_pi_bar(pi_rf, feature_names, 'Random Forest', 'pi_rf.png')
    save_pi_bar(pi_gb, feature_names, 'Gradient Boosting', 'pi_gb.png')

    # ---------- 比較図 ----------
    save_comparison(pi_rf, pi_gb, feature_names, 'pi_comparison.png')

    # ---------- アーティファクト保存 ----------
    artifacts_path = os.path.join(RESULTS_DIR, 'pi_artifacts.pkl')
    pi_data = {
        'rf': {
            'importances_mean': pi_rf.importances_mean,
            'importances_std': pi_rf.importances_std,
            'importances': pi_rf.importances,
        },
        'gb': {
            'importances_mean': pi_gb.importances_mean,
            'importances_std': pi_gb.importances_std,
            'importances': pi_gb.importances,
        },
        'feature_names': feature_names,
    }
    with open(artifacts_path, 'wb') as f:
        pickle.dump(pi_data, f)
    print(f"\n[INFO] PI アーティファクト保存: {artifacts_path}")

    print("\n[DONE] 03_permutation_analysis.py 完了")


if __name__ == '__main__':
    main()
