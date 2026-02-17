"""
00_train_model.py
California Housing データセットを用いた回帰モデルの学習・評価・保存

- RandomForestRegressor / GradientBoostingRegressor を学習
- MSE, R2 を表示
- モデル・テストデータ・散布図を results/ に保存
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ========== 設定 ==========
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RANDOM_STATE = 42
TEST_SIZE = 0.2
DPI = 150

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- データ読み込み ----------
    print("[INFO] California Housing データセットを読み込み中...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    # 特徴量名を日本語に変換
    FEATURE_NAME_JP = {
        'MedInc': '収入中央値',
        'HouseAge': '築年数',
        'AveRooms': '平均部屋数',
        'AveBedrms': '平均寝室数',
        'Population': '人口',
        'AveOccup': '平均世帯人数',
        'Latitude': '緯度',
        'Longitude': '経度',
    }
    feature_names = [FEATURE_NAME_JP.get(f, f) for f in housing.feature_names]
    print(f"  サンプル数: {X.shape[0]}, 特徴量数: {X.shape[1]}")
    print(f"  特徴量: {feature_names}")

    # ---------- 学習/テスト分割 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  学習データ: {X_train.shape[0]} 件, テストデータ: {X_test.shape[0]} 件")

    # ---------- モデル学習 ----------
    models = {}

    # Random Forest
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

    # Gradient Boosting
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

    # ---------- モデル保存 ----------
    rf_path = os.path.join(RESULTS_DIR, 'rf_model.pkl')
    gb_path = os.path.join(RESULTS_DIR, 'gb_model.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n[INFO] RF モデル保存: {rf_path}")

    with open(gb_path, 'wb') as f:
        pickle.dump(gb, f)
    print(f"[INFO] GB モデル保存: {gb_path}")

    # ---------- テストデータ保存 ----------
    test_data_path = os.path.join(RESULTS_DIR, 'test_data.pkl')
    test_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names),
    }
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"[INFO] テストデータ保存: {test_data_path}")

    # ---------- 散布図: 実測 vs 予測 ----------
    print("\n[INFO] 散布図を作成中...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        label = "Random Forest" if name == 'rf' else "Gradient Boosting"
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.3, s=8, c='#00bfff', edgecolors='none')
        # 対角線
        lims = [
            min(y_test.min(), y_pred.min()) - 0.2,
            max(y_test.max(), y_pred.max()) + 0.2,
        ]
        ax.plot(lims, lims, '--', color='#ff6347', linewidth=1.5, label='理想線 (y=x)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('実測値 (Actual)', fontsize=12)
        ax.set_ylabel('予測値 (Predicted)', fontsize=12)
        ax.set_title(f'{label}\nMSE={mse:.4f}  R²={r2:.4f}', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

    fig.suptitle('California Housing: 実測値 vs 予測値', fontsize=15, y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(RESULTS_DIR, 'actual_vs_predicted.png')
    fig.savefig(scatter_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] 散布図保存: {scatter_path}")

    print("\n[DONE] 00_train_model.py 完了")


if __name__ == '__main__':
    main()
