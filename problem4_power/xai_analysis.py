"""
xai_analysis.py
電力系統 XAI 分析: SHAP, LIME, Permutation Importance, PDP

Case 1 (潮流異常検知) と Case 2 (電圧安定性余裕予測) の両方に対して
説明可能AI手法を適用し、特徴量の重要度・影響を可視化する。
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ========== 設定 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DPI = 150

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']


def load_case_data(case_num):
    """ケースデータを読み込む"""
    path = os.path.join(RESULTS_DIR, f'case{case_num}_models.pkl')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# ============================================================
# Case 1: 潮流異常検知 (分類)
# ============================================================
def analyze_case1(data):
    """Case 1 の XAI 分析"""
    print("\n" + "=" * 60)
    print("Case 1: 潮流異常検知の XAI 分析")
    print("=" * 60)

    models = data['models']
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # 主要分析には GradientBoosting を使用
    model = models['gb']

    # 異常サンプルのインデックス
    anomaly_idx = np.where(y_test == 1)[0]

    # --- SHAP TreeExplainer ---
    print("\n[INFO] Case 1: SHAP 分析中...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot (beeswarm)
    print("  SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title('潮流異常検知: SHAP Summary Plot', fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case1_shap_summary.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"  保存: {path}")

    # SHAP bar plot
    print("  SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=20,
    )
    plt.title('潮流異常検知: SHAP 特徴量重要度', fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case1_shap_bar.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"  保存: {path}")

    # SHAP waterfall for 3 anomaly samples
    print("  SHAP waterfall plots...")
    # expected_value が配列の場合はスカラーに変換
    base_val = explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(base_val[0]) if len(base_val) == 1 else base_val

    for i in range(3):
        idx = anomaly_idx[i]
        single_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test[idx],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(single_explanation, max_display=15, show=False)
        plt.title(f'潮流異常検知: SHAP Waterfall (異常サンプル {i+1})',
                  fontsize=13, pad=15)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'case1_shap_waterfall{i}.png')
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  保存: {path}")

    # --- LIME ---
    print("\n[INFO] Case 1: LIME 分析中...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['正常', '異常'],
        mode='classification',
        random_state=42,
    )

    for i in range(3):
        idx = anomaly_idx[i]
        exp = lime_explainer.explain_instance(
            X_test[idx],
            model.predict_proba,
            num_features=15,
            top_labels=1,
        )
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(12, 7)
        fig.suptitle(f'潮流異常検知: LIME 説明 (異常サンプル {i+1})',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'case1_lime_sample{i}.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  保存: {path}")

    # --- Permutation Importance ---
    print("\n[INFO] Case 1: Permutation Importance 分析中...")
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    sorted_idx = perm_result.importances_mean.argsort()[::-1][:20]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(
        perm_result.importances[sorted_idx].T,
        vert=False,
        tick_labels=[feature_names[i] for i in sorted_idx],
    )
    ax.set_title('潮流異常検知: Permutation Importance', fontsize=14)
    ax.set_xlabel('重要度の低下量', fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case1_pi.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {path}")

    # --- PDP for top 4 features ---
    print("\n[INFO] Case 1: Partial Dependence Plot 分析中...")
    top4_idx = sorted_idx[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feat_idx in enumerate(top4_idx):
        ax = axes[i // 2, i % 2]
        PartialDependenceDisplay.from_estimator(
            model, X_test,
            features=[feat_idx],
            feature_names=feature_names,
            ax=ax,
            kind='average',
        )
        ax.set_title(f'PDP: {feature_names[feat_idx]}', fontsize=11)

    fig.suptitle('潮流異常検知: Partial Dependence Plots (上位4特徴量)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case1_pdp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {path}")

    return perm_result, feature_names


# ============================================================
# Case 2: 電圧安定性余裕予測 (回帰)
# ============================================================
def analyze_case2(data):
    """Case 2 の XAI 分析"""
    print("\n" + "=" * 60)
    print("Case 2: 電圧安定性余裕予測の XAI 分析")
    print("=" * 60)

    models = data['models']
    X_train = data['X_train']
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

    model = models['gb']

    # --- SHAP TreeExplainer ---
    print("\n[INFO] Case 2: SHAP 分析中...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot
    print("  SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False,
        max_display=10,
    )
    plt.title('電圧安定性: SHAP Summary Plot', fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case2_shap_summary.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"  保存: {path}")

    # SHAP bar plot
    print("  SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=10,
    )
    plt.title('電圧安定性: SHAP 特徴量重要度', fontsize=14, pad=15)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case2_shap_bar.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f"  保存: {path}")

    # SHAP waterfall for 3 samples (pick varied margin values)
    print("  SHAP waterfall plots...")
    # expected_value が配列の場合はスカラーに変換
    base_val = explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(base_val[0]) if len(base_val) == 1 else base_val

    # 低・中・高マージンのサンプルを1つずつ選択
    y_pred = model.predict(X_test)
    sorted_by_pred = np.argsort(y_pred)
    sample_indices = [
        sorted_by_pred[len(sorted_by_pred) // 10],       # 低マージン
        sorted_by_pred[len(sorted_by_pred) // 2],        # 中マージン
        sorted_by_pred[9 * len(sorted_by_pred) // 10],   # 高マージン
    ]

    for i, idx in enumerate(sample_indices):
        margin_labels = ['低マージン', '中マージン', '高マージン']
        single_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test[idx],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(single_explanation, max_display=10, show=False)
        plt.title(f'電圧安定性: SHAP Waterfall ({margin_labels[i]}, '
                  f'予測={y_pred[idx]:.1f}%)',
                  fontsize=13, pad=15)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'case2_shap_waterfall{i}.png')
        plt.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  保存: {path}")

    # --- LIME ---
    print("\n[INFO] Case 2: LIME 分析中...")
    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode='regression',
        random_state=42,
    )

    for i, idx in enumerate(sample_indices):
        margin_labels = ['低マージン', '中マージン', '高マージン']
        exp = lime_explainer.explain_instance(
            X_test[idx],
            model.predict,
            num_features=10,
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(12, 7)
        fig.suptitle(f'電圧安定性: LIME 説明 ({margin_labels[i]}, '
                     f'予測={y_pred[idx]:.1f}%)',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f'case2_lime_sample{i}.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  保存: {path}")

    # --- Permutation Importance ---
    print("\n[INFO] Case 2: Permutation Importance 分析中...")
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    sorted_idx = perm_result.importances_mean.argsort()[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(
        perm_result.importances[sorted_idx].T,
        vert=False,
        tick_labels=[feature_names[i] for i in sorted_idx],
    )
    ax.set_title('電圧安定性: Permutation Importance', fontsize=14)
    ax.set_xlabel('重要度の低下量', fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case2_pi.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {path}")

    # --- PDP for top 4 features ---
    print("\n[INFO] Case 2: Partial Dependence Plot 分析中...")
    top4_idx = sorted_idx[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feat_idx in enumerate(top4_idx):
        ax = axes[i // 2, i % 2]
        PartialDependenceDisplay.from_estimator(
            model, X_test,
            features=[feat_idx],
            feature_names=feature_names,
            ax=ax,
            kind='average',
        )
        ax.set_title(f'PDP: {feature_names[feat_idx]}', fontsize=11)

    fig.suptitle('電圧安定性: Partial Dependence Plots (上位4特徴量)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'case2_pdp.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {path}")

    return perm_result, feature_names


# ============================================================
# 比較分析
# ============================================================
def comparison_analysis(case1_perm, case1_features, case2_perm, case2_features):
    """2つのケースの特徴量重要度を比較する"""
    print("\n" + "=" * 60)
    print("比較分析: 潮流異常検知 vs 電圧安定性")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Case 1: 潮流異常検知
    ax = axes[0]
    sorted_idx1 = case1_perm.importances_mean.argsort()[::-1][:15]
    names1 = [case1_features[i] for i in sorted_idx1]
    values1 = case1_perm.importances_mean[sorted_idx1]
    colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(names1)))

    bars1 = ax.barh(range(len(names1)), values1, color=colors1, edgecolor='white',
                    linewidth=0.5)
    ax.set_yticks(range(len(names1)))
    ax.set_yticklabels(names1, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_title('潮流異常検知\n(分類タスク)', fontsize=13)

    # 値のラベル
    for bar, val in zip(bars1, values1):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='white')

    # Case 2: 電圧安定性
    ax = axes[1]
    sorted_idx2 = case2_perm.importances_mean.argsort()[::-1]
    names2 = [case2_features[i] for i in sorted_idx2]
    values2 = case2_perm.importances_mean[sorted_idx2]
    colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, len(names2)))

    bars2 = ax.barh(range(len(names2)), values2, color=colors2, edgecolor='white',
                    linewidth=0.5)
    ax.set_yticks(range(len(names2)))
    ax.set_yticklabels(names2, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_title('電圧安定性余裕予測\n(回帰タスク)', fontsize=13)

    for bar, val in zip(bars2, values2):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='white')

    fig.suptitle('電力系統 XAI 比較: 特徴量重要度ランキング', fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'power_comparison.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[INFO] 比較プロット保存: {path}")


# ============================================================
# メイン
# ============================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # データ読み込み
    print("[INFO] Case 1 データを読み込み中...")
    case1_data = load_case_data(1)
    print("[INFO] Case 2 データを読み込み中...")
    case2_data = load_case_data(2)

    # Case 1 分析
    case1_perm, case1_features = analyze_case1(case1_data)

    # Case 2 分析
    case2_perm, case2_features = analyze_case2(case2_data)

    # 比較分析
    comparison_analysis(case1_perm, case1_features, case2_perm, case2_features)

    print("\n" + "=" * 60)
    print("[DONE] xai_analysis.py 完了")
    print(f"全ての結果は {RESULTS_DIR} に保存されました。")
    print("=" * 60)


if __name__ == '__main__':
    main()
