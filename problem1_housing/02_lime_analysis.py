"""
02_lime_analysis.py
LIME (Local Interpretable Model-agnostic Explanations) を用いた局所説明

- LimeTabularExplainer で RF / GB の個別予測を説明
- 3 サンプルについて説明図を保存
- LIME アーティファクトを results/ に保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150
SAMPLE_INDICES = [0, 1, 2]
NUM_FEATURES = 8


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


def create_explainer(X_train, feature_names):
    """LimeTabularExplainer を作成"""
    print("[INFO] LIME Explainer を作成中...")
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='regression',
        random_state=42,
        verbose=False,
    )
    return explainer


def save_lime_explanation(explanation, model_name, sample_idx, filename, predicted_val, actual_val):
    """LIME の説明を画像として保存"""
    print(f"[INFO] LIME 説明保存 (sample {sample_idx}): {filename}")

    # 説明の重みを取得
    exp_list = explanation.as_list()
    features = [e[0] for e in exp_list]
    weights = [e[1] for e in exp_list]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#ff6347' if w < 0 else '#00bfff' for w in weights]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, weights, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('寄与度 (Weight)', fontsize=12)
    ax.set_title(
        f'LIME 説明 - {model_name} (sample {sample_idx})\n'
        f'予測値: {predicted_val:.3f} / 実測値: {actual_val:.3f}',
        fontsize=13, pad=15
    )
    ax.axvline(x=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    rf, gb, test_data = load_artifacts()
    X_train = test_data['X_train']
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    feature_names = test_data['feature_names']

    explainer = create_explainer(X_train, feature_names)

    models = {'rf': ('Random Forest', rf), 'gb': ('Gradient Boosting', gb)}
    lime_artifacts = {}

    for model_key, (model_label, model) in models.items():
        print(f"\n{'='*60}")
        print(f"  {model_label} の LIME 分析")
        print(f"{'='*60}")

        model_artifacts = []

        for sample_idx in SAMPLE_INDICES:
            print(f"\n[INFO] サンプル {sample_idx} の説明を生成中...")
            instance = X_test[sample_idx]
            actual_val = y_test[sample_idx]
            predicted_val = model.predict(instance.reshape(1, -1))[0]

            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=model.predict,
                num_features=NUM_FEATURES,
                num_samples=5000,
            )

            filename = f'lime_{model_key}_sample{sample_idx}.png'
            save_lime_explanation(
                explanation, model_label, sample_idx, filename,
                predicted_val, actual_val
            )

            # アーティファクト保存用
            model_artifacts.append({
                'sample_idx': sample_idx,
                'exp_list': explanation.as_list(),
                'predicted_val': predicted_val,
                'actual_val': actual_val,
                'intercept': explanation.intercept[0] if hasattr(explanation.intercept, '__getitem__') else explanation.intercept,
                'local_pred': explanation.local_pred[0] if explanation.local_pred is not None else None,
            })

        lime_artifacts[model_key] = model_artifacts

    # ---------- アーティファクト保存 ----------
    artifacts_path = os.path.join(RESULTS_DIR, 'lime_artifacts.pkl')
    with open(artifacts_path, 'wb') as f:
        pickle.dump(lime_artifacts, f)
    print(f"\n[INFO] LIME アーティファクト保存: {artifacts_path}")

    print("\n[DONE] 02_lime_analysis.py 完了")


if __name__ == '__main__':
    main()
