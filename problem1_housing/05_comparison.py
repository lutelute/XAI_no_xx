"""
05_comparison.py
全 XAI 手法の比較サマリー

- SHAP, Permutation Importance の特徴量ランキングを横断比較
- 各手法の特徴量重要度スコアを正規化してランキング表を作成
- comparison_feature_ranking.png と comparison_table.png を保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150


def load_all_artifacts():
    """全アーティファクトを読み込む"""
    artifacts = {}

    # SHAP
    shap_path = os.path.join(RESULTS_DIR, 'shap_artifacts.pkl')
    if os.path.exists(shap_path):
        with open(shap_path, 'rb') as f:
            artifacts['shap'] = pickle.load(f)
        print("[INFO] SHAP アーティファクト読み込み完了")
    else:
        print("[WARN] SHAP アーティファクトが見つかりません")

    # Permutation Importance
    pi_path = os.path.join(RESULTS_DIR, 'pi_artifacts.pkl')
    if os.path.exists(pi_path):
        with open(pi_path, 'rb') as f:
            artifacts['pi'] = pickle.load(f)
        print("[INFO] PI アーティファクト読み込み完了")
    else:
        print("[WARN] PI アーティファクトが見つかりません")

    # Models (for built-in feature importance)
    rf_path = os.path.join(RESULTS_DIR, 'rf_model.pkl')
    gb_path = os.path.join(RESULTS_DIR, 'gb_model.pkl')
    if os.path.exists(rf_path) and os.path.exists(gb_path):
        with open(rf_path, 'rb') as f:
            artifacts['rf_model'] = pickle.load(f)
        with open(gb_path, 'rb') as f:
            artifacts['gb_model'] = pickle.load(f)
        print("[INFO] モデル読み込み完了")

    # Test data (for feature names)
    td_path = os.path.join(RESULTS_DIR, 'test_data.pkl')
    if os.path.exists(td_path):
        with open(td_path, 'rb') as f:
            artifacts['test_data'] = pickle.load(f)
        print("[INFO] テストデータ読み込み完了")

    return artifacts


def normalize(arr):
    """0-1 に正規化"""
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def build_importance_table(artifacts):
    """各手法の特徴量重要度を正規化してテーブルにまとめる"""
    feature_names = artifacts['test_data']['feature_names']
    n_features = len(feature_names)

    methods = {}  # method_name -> normalized importance array

    # --- Built-in Feature Importance (Gini / gain) ---
    if 'rf_model' in artifacts:
        methods['RF (built-in)'] = normalize(artifacts['rf_model'].feature_importances_)
    if 'gb_model' in artifacts:
        methods['GB (built-in)'] = normalize(artifacts['gb_model'].feature_importances_)

    # --- SHAP mean |SHAP value| ---
    if 'shap' in artifacts:
        for key, label in [('rf', 'RF (SHAP)'), ('gb', 'GB (SHAP)')]:
            if key in artifacts['shap']:
                shap_vals = artifacts['shap'][key]['shap_values_array']
                mean_abs = np.abs(shap_vals).mean(axis=0)
                methods[label] = normalize(mean_abs)

    # --- Permutation Importance ---
    if 'pi' in artifacts:
        for key, label in [('rf', 'RF (Perm)'), ('gb', 'GB (Perm)')]:
            if key in artifacts['pi']:
                mean_imp = artifacts['pi'][key]['importances_mean']
                methods[label] = normalize(mean_imp)

    return feature_names, methods


def save_feature_ranking(feature_names, methods, filename):
    """各手法の特徴量ランキングを並べた棒グラフ"""
    print(f"[INFO] 特徴量ランキング比較図保存: {filename}")

    n_features = len(feature_names)
    n_methods = len(methods)

    # RF (SHAP) のランキング順で並べる (あれば)
    if 'RF (SHAP)' in methods:
        sort_key = methods['RF (SHAP)']
    elif 'RF (built-in)' in methods:
        sort_key = methods['RF (built-in)']
    else:
        sort_key = list(methods.values())[0]
    sorted_idx = np.argsort(sort_key)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # カラーパレット
    colors = ['#00bfff', '#ff9f43', '#2ecc71', '#ff6347', '#a29bfe', '#fd79a8']

    fig, ax = plt.subplots(figsize=(14, 8))

    bar_width = 0.8 / n_methods
    y_pos = np.arange(n_features)

    for i, (method_name, scores) in enumerate(methods.items()):
        sorted_scores = scores[sorted_idx]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.barh(
            y_pos + offset, sorted_scores,
            height=bar_width * 0.9,
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.3,
            label=method_name, alpha=0.85,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=12)
    ax.set_xlabel('正規化された重要度 (0-1)', fontsize=12)
    ax.set_title('特徴量重要度ランキング比較 (全手法)', fontsize=15, pad=15)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def save_comparison_table(feature_names, methods, filename):
    """特徴量重要度スコアのテーブル画像を保存"""
    print(f"[INFO] 比較テーブル保存: {filename}")

    n_features = len(feature_names)
    method_names = list(methods.keys())
    n_methods = len(method_names)

    # ランキング (1=最重要) を計算
    rankings = {}
    for method_name, scores in methods.items():
        ranked = np.zeros(n_features, dtype=int)
        sorted_idx = np.argsort(scores)[::-1]
        for rank, idx in enumerate(sorted_idx):
            ranked[idx] = rank + 1
        rankings[method_name] = ranked

    # テーブルデータ作成
    cell_text = []
    cell_colors = []

    # カスタムカラーマップ (dark theme に合う)
    cmap = LinearSegmentedColormap.from_list('rank', ['#2ecc71', '#f39c12', '#e74c3c'], N=n_features)

    for fi in range(n_features):
        row = []
        row_colors = []
        for method_name in method_names:
            rank = rankings[method_name][fi]
            score = methods[method_name][fi]
            row.append(f'{rank}位\n({score:.3f})')
            # ランクに応じた色
            intensity = (rank - 1) / max(n_features - 1, 1)
            c = cmap(intensity)
            row_colors.append((*c[:3], 0.4))  # 半透明
        cell_text.append(row)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(max(12, 2.2 * n_methods), max(8, 0.8 * n_features)))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        rowLabels=feature_names,
        colLabels=method_names,
        cellColours=cell_colors,
        rowColours=[('#333333')] * n_features,
        colColours=[('#444444')] * n_methods,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # セルのテキスト色を白に
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#555555')
        cell.get_text().set_color('white')
        if key[0] == 0:  # ヘッダ行
            cell.get_text().set_fontweight('bold')
            cell.get_text().set_fontsize(9)
        if key[1] == -1:  # 行ラベル
            cell.get_text().set_fontweight('bold')

    ax.set_title(
        '特徴量重要度 比較テーブル (ランキング / 正規化スコア)',
        fontsize=14, pad=20, color='white'
    )

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    print("[INFO] 全手法の比較分析を開始...")
    artifacts = load_all_artifacts()

    if 'test_data' not in artifacts:
        print("[ERROR] テストデータが見つかりません。先に 00_train_model.py を実行してください。")
        return

    feature_names, methods = build_importance_table(artifacts)

    if not methods:
        print("[ERROR] 比較可能な手法がありません。先に各分析スクリプトを実行してください。")
        return

    print(f"\n[INFO] 比較対象の手法: {list(methods.keys())}")

    # ---------- ランキング比較図 ----------
    save_feature_ranking(feature_names, methods, 'comparison_feature_ranking.png')

    # ---------- テーブル ----------
    save_comparison_table(feature_names, methods, 'comparison_table.png')

    # ---------- サマリー出力 ----------
    print("\n" + "=" * 60)
    print("特徴量重要度サマリー")
    print("=" * 60)
    for method_name, scores in methods.items():
        sorted_idx = np.argsort(scores)[::-1]
        top3 = [feature_names[i] for i in sorted_idx[:3]]
        print(f"  {method_name}: Top3 = {top3}")
    print("=" * 60)

    print("\n[DONE] 05_comparison.py 完了")


if __name__ == '__main__':
    main()
