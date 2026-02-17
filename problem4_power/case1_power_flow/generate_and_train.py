"""
generate_and_train.py
電力系統の潮流異常検知: 合成データ生成 + 分類モデル学習

- 10母線系統を模擬した合成データを生成 (有効電力, 無効電力, 電圧, 線路潮流)
- N-1 想定事故的な異常パターンを注入
- RandomForestClassifier / GradientBoostingClassifier を学習
- 混同行列・分類レポートを出力
"""

import matplotlib
matplotlib.use('Agg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ========== 設定 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'results')
RANDOM_STATE = 42
TEST_SIZE = 0.2
DPI = 150

N_NORMAL = 2000
N_ANOMALY = 500
N_BUSES = 10
N_LINES = 10

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']


def generate_feature_names():
    """特徴量名を生成する（日本語）"""
    names = []
    for i in range(1, N_BUSES + 1):
        names.append(f'有効電力 母線{i}')
    for i in range(1, N_BUSES + 1):
        names.append(f'無効電力 母線{i}')
    for i in range(1, N_BUSES + 1):
        names.append(f'電圧 母線{i}')
    for i in range(1, N_LINES + 1):
        names.append(f'線路負荷率 {i}')
    return names


def generate_normal_data(rng, n_samples):
    """
    正常運転データを生成する。
    キルヒホッフの法則を近似: 有効電力の合計 ≈ 0 (損失を考慮して小さな正値)
    電圧は 0.95-1.05 pu の範囲
    """
    # 有効電力 (MW): 発電母線は正、負荷母線は負
    # Bus 1, 2: 発電機 (正), Bus 3-10: 負荷 (負)
    P = np.zeros((n_samples, N_BUSES))
    P[:, 0] = rng.normal(150, 15, n_samples)   # Generator 1 (slack)
    P[:, 1] = rng.normal(100, 12, n_samples)   # Generator 2
    for i in range(2, N_BUSES):
        P[:, i] = rng.normal(-30, 5, n_samples)  # Loads

    # キルヒホッフの近似: slack bus が残りを吸収
    total_load = P[:, 1:].sum(axis=1)
    losses = rng.normal(5, 1, n_samples)  # 系統損失
    P[:, 0] = -total_load + losses

    # 無効電力 (Mvar)
    Q = np.zeros((n_samples, N_BUSES))
    Q[:, 0] = rng.normal(40, 8, n_samples)
    Q[:, 1] = rng.normal(30, 6, n_samples)
    for i in range(2, N_BUSES):
        Q[:, i] = rng.normal(-10, 3, n_samples)

    # 電圧 (pu): 正常時は 0.95-1.05
    V = np.zeros((n_samples, N_BUSES))
    V[:, 0] = rng.normal(1.02, 0.01, n_samples)  # Slack bus (PV制御)
    V[:, 1] = rng.normal(1.01, 0.01, n_samples)  # Generator bus
    for i in range(2, N_BUSES):
        # 負荷母線: 発電機から離れるほど電圧低下
        drop = 0.005 * (i - 1)
        V[:, i] = rng.normal(1.0 - drop, 0.012, n_samples)

    # 線路潮流 (% loading): 正常時は 20-70%
    L = np.zeros((n_samples, N_LINES))
    for i in range(N_LINES):
        L[:, i] = rng.normal(45, 10, n_samples)
    L = np.clip(L, 5, 85)

    X = np.hstack([P, Q, V, L])
    return X


def generate_anomaly_data(rng, n_samples):
    """
    異常データを生成する (N-1 想定事故パターン)。
    - 線路開放 → 残線路に潮流集中 (過負荷)
    - 電圧低下・崩壊の前兆
    - 有効電力の不均衡
    """
    X = generate_normal_data(rng, n_samples)

    P = X[:, :N_BUSES]
    Q = X[:, N_BUSES:2*N_BUSES]
    V = X[:, 2*N_BUSES:3*N_BUSES]
    L = X[:, 3*N_BUSES:]

    for i in range(n_samples):
        anomaly_type = rng.integers(0, 4)

        if anomaly_type == 0:
            # 線路開放 → 1本の潮流が0に、隣接線路が過負荷
            tripped_line = rng.integers(0, N_LINES)
            L[i, tripped_line] = rng.uniform(0, 3)
            # 隣接線路に潮流転流
            neighbors = [(tripped_line - 1) % N_LINES, (tripped_line + 1) % N_LINES]
            for n in neighbors:
                L[i, n] += rng.uniform(25, 50)
            # 電圧低下
            affected_buses = rng.choice(range(2, N_BUSES), size=3, replace=False)
            for b in affected_buses:
                V[i, b] -= rng.uniform(0.03, 0.08)

        elif anomaly_type == 1:
            # 発電機脱落 → 大幅な電力不均衡
            gen_trip = rng.choice([0, 1])
            P[i, gen_trip] *= rng.uniform(0.0, 0.2)
            Q[i, gen_trip] *= rng.uniform(0.0, 0.3)
            # 周波数低下に伴う負荷変動
            for b in range(2, N_BUSES):
                P[i, b] *= rng.uniform(0.85, 1.0)
            # 広範囲な電圧低下
            V[i, :] -= rng.uniform(0.02, 0.06, N_BUSES)
            # 潮流パターンの変化
            L[i, :] *= rng.uniform(0.5, 1.8, N_LINES)

        elif anomaly_type == 2:
            # 電圧崩壊の前兆: 無効電力不足
            Q[i, 0] *= rng.uniform(0.2, 0.5)
            Q[i, 1] *= rng.uniform(0.3, 0.6)
            for b in range(2, N_BUSES):
                Q[i, b] *= rng.uniform(1.5, 2.5)  # 無効電力需要増大
            V[i, 2:] -= rng.uniform(0.05, 0.12, N_BUSES - 2)
            # 一部線路が重負荷
            heavy = rng.choice(range(N_LINES), size=4, replace=False)
            for h in heavy:
                L[i, h] = rng.uniform(85, 120)

        elif anomaly_type == 3:
            # カスケード障害: 複数線路の連鎖開放
            n_trip = rng.integers(2, 4)
            tripped = rng.choice(range(N_LINES), size=n_trip, replace=False)
            for t in tripped:
                L[i, t] = rng.uniform(0, 5)
            remaining = [j for j in range(N_LINES) if j not in tripped]
            for r in remaining:
                L[i, r] += rng.uniform(15, 40)
            V[i, :] -= rng.uniform(0.03, 0.10, N_BUSES)
            P[i, 0] += rng.uniform(-30, 30)

    X = np.hstack([P, Q, V, L])
    return X


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    feature_names = generate_feature_names()
    print(f"[INFO] 特徴量数: {len(feature_names)}")

    # ---------- データ生成 ----------
    print(f"\n[INFO] 正常データ生成中... ({N_NORMAL} サンプル)")
    X_normal = generate_normal_data(rng, N_NORMAL)
    y_normal = np.zeros(N_NORMAL, dtype=int)

    print(f"[INFO] 異常データ生成中... ({N_ANOMALY} サンプル)")
    X_anomaly = generate_anomaly_data(rng, N_ANOMALY)
    y_anomaly = np.ones(N_ANOMALY, dtype=int)

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])
    print(f"[INFO] 合計サンプル数: {X.shape[0]}, 特徴量数: {X.shape[1]}")
    print(f"  正常: {N_NORMAL}, 異常: {N_ANOMALY}")

    # ---------- 学習/テスト分割 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] 学習データ: {X_train.shape[0]} 件, テストデータ: {X_test.shape[0]} 件")

    # ---------- モデル学習 ----------
    models = {}

    print("\n[INFO] RandomForestClassifier を学習中...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    print("  学習完了")

    print("[INFO] GradientBoostingClassifier を学習中...")
    gb = GradientBoostingClassifier(
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

    for name, model in models.items():
        label = "RandomForest" if name == 'rf' else "GradientBoosting"
        y_pred = model.predict(X_test)
        print(f"\n--- {label} ---")
        print(classification_report(
            y_test, y_pred,
            target_names=['正常 (Normal)', '異常 (Anomaly)']
        ))
    print("=" * 60)

    # ---------- モデル・データ保存 ----------
    save_data = {
        'models': models,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
    }
    save_path = os.path.join(RESULTS_DIR, 'case1_models.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\n[INFO] モデル・データ保存: {save_path}")

    # ---------- 混同行列 ----------
    print("[INFO] 混同行列を作成中...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, model) in zip(axes, models.items()):
        label = "Random Forest" if name == 'rf' else "Gradient Boosting"
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['正常', '異常']
        )
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{label}', fontsize=13)
        ax.set_xlabel('予測ラベル', fontsize=11)
        ax.set_ylabel('真のラベル', fontsize=11)

    fig.suptitle('潮流異常検知: 混同行列', fontsize=15, y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'case1_confusion_matrix.png')
    fig.savefig(cm_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] 混同行列保存: {cm_path}")

    print("\n[DONE] case1_power_flow/generate_and_train.py 完了")


if __name__ == '__main__':
    main()
