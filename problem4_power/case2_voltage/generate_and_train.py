"""
generate_and_train.py
電圧安定性余裕の予測: 合成データ生成 + 回帰モデル学習

- 電圧安定性に関連する特徴量から電圧安定性余裕 (0-100%) を予測
- 高負荷 + 低無効電力予備力 → 低マージン (現実的な相関)
- RandomForestRegressor / GradientBoostingRegressor を学習
- MSE, R2 を表示し、実測 vs 予測散布図を保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========== 設定 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'results')
RANDOM_STATE = 42
TEST_SIZE = 0.2
DPI = 150
N_SAMPLES = 2000

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

FEATURE_NAMES = [
    '総負荷 MW',
    '総負荷 Mvar',
    '無効電力予備 Mvar',
    '最大線路負荷率 %',
    '最弱母線電圧',
    'オンライン発電機数',
    'タップ位置 平均',
    '調相設備 Mvar',
    '送電限界 MW',
    '外気温',
]


def generate_voltage_stability_data(rng, n_samples):
    """
    電圧安定性データを生成する。

    物理的に妥当な相関関係:
    - 高負荷 → 低マージン
    - 低無効電力予備力 → 低マージン
    - 低電圧 → 低マージン
    - 高線路負荷率 → 低マージン
    - オンライン発電機数が少ない → 低マージン
    - 高温 → (送電容量低下) → 低マージン
    """
    # 基本特徴量を生成
    total_load_mw = rng.uniform(200, 800, n_samples)
    total_load_mvar = total_load_mw * rng.uniform(0.25, 0.50, n_samples)

    # 無効電力予備力: 負荷が大きいほど予備力が少なくなりやすい
    reactive_reserve_base = rng.uniform(50, 300, n_samples)
    reactive_reserve = reactive_reserve_base - 0.15 * total_load_mw + rng.normal(0, 15, n_samples)
    reactive_reserve = np.clip(reactive_reserve, 0, 350)

    # 最大線路負荷率: 負荷が大きいほど高くなる
    max_line_loading = 20 + 0.08 * total_load_mw + rng.normal(0, 8, n_samples)
    max_line_loading = np.clip(max_line_loading, 10, 120)

    # 最弱母線電圧: 負荷が大きく予備力が少ないほど低い
    weakest_voltage = (1.02
                       - 0.0003 * total_load_mw
                       + 0.0004 * reactive_reserve
                       + rng.normal(0, 0.015, n_samples))
    weakest_voltage = np.clip(weakest_voltage, 0.80, 1.06)

    # オンライン発電機数
    gen_count = rng.integers(3, 9, n_samples).astype(float)

    # タップ位置平均
    tap_position = rng.uniform(0.95, 1.05, n_samples)

    # 調相設備 (Mvar)
    shunt_comp = rng.uniform(0, 100, n_samples)

    # 送電限界
    transfer_limit = 300 + 50 * gen_count + rng.normal(0, 30, n_samples)

    # 外気温度
    ambient_temp = rng.uniform(-5, 42, n_samples)

    # ========== 目的変数: 電圧安定性余裕 (%) ==========
    # 物理的直観に基づく非線形モデル
    margin = (
        80.0
        - 0.08 * total_load_mw                    # 負荷増 → マージン減
        + 0.12 * reactive_reserve                  # 予備力増 → マージン増
        - 0.25 * max_line_loading                  # 線路過負荷 → マージン減
        + 30.0 * (weakest_voltage - 0.95)          # 電圧低下 → マージン減
        + 2.5 * gen_count                          # 発電機多い → マージン増
        + 15.0 * (tap_position - 1.0)              # タップ調整効果
        + 0.05 * shunt_comp                        # 調相設備効果
        + 0.01 * transfer_limit                    # 送電余力効果
        - 0.15 * ambient_temp                      # 高温 → マージン減
        # 交互作用項
        - 0.0001 * total_load_mw * max_line_loading
        + 0.0005 * reactive_reserve * gen_count
        + rng.normal(0, 3, n_samples)              # ノイズ
    )
    margin = np.clip(margin, 0, 100)

    X = np.column_stack([
        total_load_mw,
        total_load_mvar,
        reactive_reserve,
        max_line_loading,
        weakest_voltage,
        gen_count,
        tap_position,
        shunt_comp,
        transfer_limit,
        ambient_temp,
    ])

    return X, margin


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # ---------- データ生成 ----------
    print(f"[INFO] 電圧安定性データ生成中... ({N_SAMPLES} サンプル)")
    X, y = generate_voltage_stability_data(rng, N_SAMPLES)
    print(f"  サンプル数: {X.shape[0]}, 特徴量数: {X.shape[1]}")
    print(f"  特徴量: {FEATURE_NAMES}")
    print(f"  目的変数 (電圧安定性余裕): mean={y.mean():.1f}%, "
          f"std={y.std():.1f}%, range=[{y.min():.1f}%, {y.max():.1f}%]")

    # ---------- 学習/テスト分割 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[INFO] 学習データ: {X_train.shape[0]} 件, テストデータ: {X_test.shape[0]} 件")

    # ---------- モデル学習 ----------
    models = {}

    print("\n[INFO] RandomForestRegressor を学習中...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    print("  学習完了")

    print("[INFO] GradientBoostingRegressor を学習中...")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    models['gb'] = gb
    print("  学習完了")

    # ---------- 評価 ----------
    print("\n" + "=" * 60)
    print("モデル評価結果")
    print("=" * 60)

    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        label = "RandomForest" if name == 'rf' else "GradientBoosting"
        print(f"  {label}:")
        print(f"    MSE  = {mse:.4f}")
        print(f"    R2   = {r2:.4f}")
    print("=" * 60)

    # ---------- モデル・データ保存 ----------
    save_data = {
        'models': models,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': FEATURE_NAMES,
    }
    save_path = os.path.join(RESULTS_DIR, 'case2_models.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n[INFO] モデル・データ保存: {save_path}")

    # ---------- 散布図: 実測 vs 予測 ----------
    print("[INFO] 散布図を作成中...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        label = "Random Forest" if name == 'rf' else "Gradient Boosting"
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.4, s=12, c='#00e5ff', edgecolors='none')
        lims = [
            min(y_test.min(), y_pred.min()) - 2,
            max(y_test.max(), y_pred.max()) + 2,
        ]
        ax.plot(lims, lims, '--', color='#ff6347', linewidth=1.5, label='理想線 (y=x)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('実測値 (%)', fontsize=12)
        ax.set_ylabel('予測値 (%)', fontsize=12)
        ax.set_title(f'{label}\nMSE={mse:.4f}  R\u00b2={r2:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

    fig.suptitle('電圧安定性余裕: 実測値 vs 予測値', fontsize=15, y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(RESULTS_DIR, 'case2_actual_vs_predicted.png')
    fig.savefig(scatter_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] 散布図保存: {scatter_path}")

    print("\n[DONE] case2_voltage/generate_and_train.py 完了")


if __name__ == '__main__':
    main()
