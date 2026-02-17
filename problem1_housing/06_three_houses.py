"""
06_three_houses.py
3軒の住宅を同時に比較し、何が価格を上げる/下げるかを可視化する。

- SHAP waterfall を3軒横並びで表示
- 各住宅の特徴量値と予測価格を表形式で比較
- SHAP force plot スタイルの横棒グラフで価格押し上げ/押し下げ要因を比較
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DPI = 150
SAMPLE_INDICES = [0, 1, 2]


def load_artifacts():
    with open(os.path.join(RESULTS_DIR, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, 'shap_artifacts.pkl'), 'rb') as f:
        shap_data = pickle.load(f)
    return rf, test_data, shap_data


def main():
    rf, test_data, shap_data = load_artifacts()
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    feature_names = test_data['feature_names']

    sv_data = shap_data['rf']
    shap_values = sv_data['shap_values_array']  # (n_samples, n_features)
    base_values = sv_data['base_values']         # (n_samples,) or scalar
    data_vals = sv_data['data']                  # (n_samples, n_features)

    # 予測値
    y_pred = rf.predict(X_test[:500])

    # 3軒のサンプル情報
    houses = []
    for idx in SAMPLE_INDICES:
        base_val = base_values[idx] if np.ndim(base_values) > 0 else base_values
        houses.append({
            'idx': idx,
            'features': data_vals[idx],
            'shap': shap_values[idx],
            'base': base_val,
            'pred': y_pred[idx],
            'actual': y_test[idx],
        })

    # ============================================================
    # 図1: 3軒の特徴量比較テーブル + 予測価格
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    ax.axis('off')

    col_labels = [f'住宅 {i+1}\n(サンプル {idx})' for i, idx in enumerate(SAMPLE_INDICES)]
    row_labels = feature_names + ['予測価格 (×$100K)', '真値価格 (×$100K)']

    cell_text = []
    for fn_idx, fn in enumerate(feature_names):
        row = []
        for h in houses:
            val = h['features'][fn_idx]
            if fn in ['緯度', '経度']:
                row.append(f'{val:.2f}')
            elif fn == '人口':
                row.append(f'{val:,.0f}')
            else:
                row.append(f'{val:.2f}')
        cell_text.append(row)

    # 予測価格と真値価格
    cell_text.append([f'${h["pred"]*100:.1f}K' for h in houses])
    cell_text.append([f'${h["actual"]*100:.1f}K' for h in houses])

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # スタイリング
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#444444')
        if key[0] == 0:  # ヘッダー行
            cell.set_facecolor('#2a3f5f')
            cell.set_text_props(fontweight='bold', color='white')
        elif key[1] == -1:  # 行ラベル
            cell.set_facecolor('#1e2d3d')
            cell.set_text_props(color='#cccccc')
        elif key[0] == len(row_labels) - 1 or key[0] == len(row_labels):
            # 価格行はハイライト
            cell.set_facecolor('#1a3a2a')
            cell.set_text_props(fontweight='bold', color='#4ecdc4')
        else:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#e0e0e0')

    ax.set_title('3軒の住宅 — 特徴量比較', fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'three_houses_table.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("保存: three_houses_table.png")

    # ============================================================
    # 図2: 3軒のSHAP寄与を横棒グラフで並列比較
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=DPI)

    for col, (h, ax) in enumerate(zip(houses, axes)):
        sv = h['shap']
        sorted_idx = np.argsort(np.abs(sv))[::-1]

        vals = sv[sorted_idx]
        names = [feature_names[i] for i in sorted_idx]
        feat_vals = [h['features'][i] for i in sorted_idx]

        colors = ['#ff6b6b' if v > 0 else '#4a9eff' for v in vals]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=colors, alpha=0.85, height=0.7)

        # 特徴量の真値の値を棒の先に表示
        for i, (bar, name, fv) in enumerate(zip(bars, names, feat_vals)):
            width = bar.get_width()
            if name in ['緯度', '経度']:
                val_str = f'{fv:.1f}'
            elif name == '人口':
                val_str = f'{fv:,.0f}'
            else:
                val_str = f'{fv:.2f}'

            if width >= 0:
                ax.text(width + 0.01, i, f'  {val_str}', va='center', fontsize=8, color='#aaaaaa')
            else:
                ax.text(width - 0.01, i, f'{val_str}  ', va='center', fontsize=8, color='#aaaaaa', ha='right')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('SHAP値 (価格への影響)', fontsize=10)

        pred_str = f'${h["pred"]*100:.1f}K'
        actual_str = f'${h["actual"]*100:.1f}K'
        ax.set_title(f'住宅 {col+1} (予測: {pred_str} / 真値: {actual_str})',
                     fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.1, axis='x')

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='価格を上げる要因 (+)'),
        Patch(facecolor='#4a9eff', label='価格を下げる要因 (-)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.8, facecolor='#1a1a2e',
               edgecolor='#555', bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('SHAP分析 — 3軒の住宅の価格決定要因を比較\n'
                 '(棒の横の数値 = その住宅の真値の特徴量値)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'three_houses_shap.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    print("保存: three_houses_shap.png")

    # ============================================================
    # 図3: SHAP waterfall 3枚横並び
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=DPI)

    for col, (h, ax) in enumerate(zip(houses, axes)):
        # SHAP Explanation オブジェクトを再構築
        exp = shap.Explanation(
            values=h['shap'],
            base_values=h['base'],
            data=h['features'],
            feature_names=feature_names,
        )
        plt.sca(ax)
        shap.plots.waterfall(exp, show=False, max_display=8)

        pred_str = f'${h["pred"]*100:.1f}K'
        actual_str = f'${h["actual"]*100:.1f}K'
        ax.set_title(f'住宅 {col+1}\n予測: {pred_str} (真値: {actual_str})',
                     fontsize=11, fontweight='bold', pad=10)

    fig.suptitle('SHAP Waterfall — 3軒同時比較\n'
                 '各特徴量がベース価格からどれだけ予測を押し上げ/押し下げたか',
                 fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'three_houses_waterfall.png'), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print("保存: three_houses_waterfall.png")

    # ============================================================
    # サマリー出力
    # ============================================================
    print("\n" + "=" * 60)
    print("3軒の住宅 — 価格決定要因サマリー")
    print("=" * 60)

    for col, h in enumerate(houses):
        sv = h['shap']
        sorted_idx = np.argsort(sv)[::-1]
        print(f"\n住宅 {col+1} (予測: ${h['pred']*100:.1f}K, 真値: ${h['actual']*100:.1f}K)")
        print("  価格を上げた要因:")
        for i in sorted_idx[:3]:
            if sv[i] > 0:
                print(f"    {feature_names[i]} = {h['features'][i]:.2f} → +{sv[i]:.3f}")
        print("  価格を下げた要因:")
        for i in sorted_idx[::-1][:3]:
            if sv[i] < 0:
                print(f"    {feature_names[i]} = {h['features'][i]:.2f} → {sv[i]:.3f}")

    print("\n[DONE] 06_three_houses.py 完了")


if __name__ == '__main__':
    main()
