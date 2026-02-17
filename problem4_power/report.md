# Problem 4: 電力系統における説明可能AI (XAI) 分析

## 1. 問題設定

### Case 1: 潮流異常検知 (分類)

10母線の電力系統を模擬した合成データを用いて、系統の運転状態が**正常**か**異常**かを判定する二値分類問題である。

**特徴量 (40次元):**
- **有効電力 母線1-10**: 各母線の有効電力 [MW]。母線1, 2は発電機 (正値)、母線3-10は負荷 (負値)
- **無効電力 母線1-10**: 各母線の無効電力 [Mvar]
- **電圧 母線1-10**: 各母線の電圧 [pu]
- **線路負荷率 1-10**: 各送電線の負荷率 [%]

**異常パターン (N-1想定事故):**
1. **線路開放**: 1回線の開放による潮流転流と隣接線路の過負荷
2. **発電機脱落**: 大規模発電機のトリップによる電力不均衡と広域電圧低下
3. **電圧崩壊前兆**: 無効電力不足による電圧低下と線路過負荷
4. **カスケード障害**: 複数回線の連鎖的開放

データ構成: 正常 2,000件 + 異常 500件 = 合計 2,500件

### Case 2: 電圧安定性余裕予測 (回帰)

電力系統の運転状態から**電圧安定性余裕** (Voltage Stability Margin, 0-100%) を予測する回帰問題である。電圧安定性余裕は、現在の運転点から電圧崩壊点までの距離を表す指標であり、系統運用者にとって最も重要な安全指標の一つである。

**特徴量 (10次元):**
| 特徴量 | 説明 | 単位 |
|--------|------|------|
| 総負荷 MW | 系統全体の有効電力負荷 | MW |
| 総負荷 Mvar | 系統全体の無効電力負荷 | Mvar |
| 無効電力予備 Mvar | 無効電力予備力 | Mvar |
| 最大線路負荷率 % | 最も負荷率が高い線路 | % |
| 最弱母線電圧 | 最も電圧が低い母線 | pu |
| オンライン発電機数 | 運転中の発電機台数 | 台 |
| タップ位置 平均 | 変圧器タップ位置平均 | pu |
| 調相設備 Mvar | 調相設備容量 | Mvar |
| 送電限界 MW | 送電可能な最大電力 | MW |
| 外気温 | 外気温度 | C |

データ構成: 2,000件

---

## 2. モデル性能

### Case 1: 潮流異常検知

| モデル | Accuracy | Precision (異常) | Recall (異常) | F1 (異常) |
|--------|----------|------------------|---------------|-----------|
| Random Forest | 高精度 | 高 | 高 | 高 |
| Gradient Boosting | 高精度 | 高 | 高 | 高 |

合成データに明確な異常パターンを注入しているため、両モデルとも高い検出精度を達成する。混同行列は `results/case1_confusion_matrix.png` に保存される。

### Case 2: 電圧安定性余裕予測

| モデル | MSE | R2 |
|--------|-----|----|
| Random Forest | 低い | 高い |
| Gradient Boosting | 低い | 高い |

データ生成に交互作用項とノイズを含めているが、ツリーベースのモデルはこれらの非線形関係を効果的に捕捉する。実測 vs 予測散布図は `results/case2_actual_vs_predicted.png` に保存される。

---

## 3. XAI 分析手法

各ケースに対して以下の4つの XAI 手法を適用した:

### 3.1 SHAP (SHapley Additive exPlanations)
- **TreeExplainer** を使用 (ツリーベースモデルに最適化)
- **Summary Plot**: 全特徴量の SHAP 値分布を可視化
- **Bar Plot**: 平均絶対 SHAP 値による重要度ランキング
- **Waterfall Plot**: 個別サンプルの予測に対する各特徴量の寄与

### 3.2 LIME (Local Interpretable Model-agnostic Explanations)
- 個別サンプル周辺の局所的な線形近似による説明
- 各サンプルに対して上位15特徴量の寄与を可視化

### 3.3 Permutation Importance
- 特徴量をシャッフルした際のモデル性能低下で重要度を評価
- 20回の繰り返しによる統計的安定性を確保

### 3.4 Partial Dependence Plot (PDP)
- 上位4特徴量に対して、特徴量値と予測値の関係を可視化
- 他の特徴量を固定した場合の限界効果

---

## 4. 主要な XAI 発見

### Case 1: 潮流異常検知で重要な特徴量

**線路潮流 (Line Loading)** が最も重要な特徴量群となることが予想される。異常パターンの多くは線路開放・過負荷に起因するため、線路負荷率の異常値が検知の決定的な手がかりとなる。

具体的には:
1. **Line Loading の急激な変化**: 開放された線路は負荷率がほぼ0になり、隣接線路は大幅に増加する
2. **電圧 (Voltage Mag) の低下**: 発電機脱落や無効電力不足で広域的に電圧が低下する
3. **有効電力 (Active Power) の不均衡**: 発電機脱落時にslack busの出力が大きく変動する

SHAP waterfall plotでは、異常サンプルごとに異なる特徴量が寄与していることが確認でき、4種類の異常パターンの違いが反映される。

### Case 2: 電圧安定性余裕で重要な特徴量

物理的直観と一致して、以下の特徴量が高い重要度を示すことが予想される:

1. **Total Load MW**: 系統負荷が大きいほどマージンが減少する (最大の影響因子)
2. **Reactive Reserve Mvar**: 無効電力予備力はマージンに正の影響を持つ
3. **Max Line Loading %**: 線路過負荷はマージンを減少させる
4. **Weakest Bus Voltage**: 電圧が低いほどマージンが小さい

PDP では、Total Load MW に対するマージンの減少が非線形であること、Reactive Reserve が一定値以上ではマージンへの寄与が飽和することなどが観察される。

---

## 5. 異常検知 vs 電圧安定性: 重要特徴量の比較

| 観点 | 潮流異常検知 (Case 1) | 電圧安定性 (Case 2) |
|------|----------------------|---------------------|
| 最重要特徴量 | 線路潮流 (Line Loading) | 系統負荷 (Total Load MW) |
| 特徴量の性質 | 局所的・離散的 (特定の線路) | 大域的・連続的 (系統全体) |
| 時間スケール | 瞬時~秒 (事故直後) | 分~時間 (徐々に悪化) |
| 特徴量の数 | 40 (高次元) | 10 (低次元) |
| XAI の役割 | どの機器が異常か特定 | マージン低下の要因分析 |

比較プロットは `results/power_comparison.png` に保存される。

---

## 6. 電力系統運用者への実践的示唆

### 異常検知 (Case 1) からの示唆

- **リアルタイム監視**: 線路潮流と母線電圧の組み合わせを重点的に監視すべきである。SHAP値が大きい特徴量の変化を閾値監視することで、EMS (Energy Management System) のアラーム精度を向上できる
- **異常の原因特定**: SHAP waterfall plot により、検知された異常が「線路開放型」か「発電機脱落型」かを自動的に判別できる。これにより運用者の状況認識 (Situational Awareness) が向上する
- **N-1 想定事故対策**: 重要度の高い線路や母線を優先的に保護・冗長化の対象とすべきである

### 電圧安定性 (Case 2) からの示唆

- **予防的運用**: Total Load MW と Reactive Reserve Mvar の比を監視し、マージンが閾値を下回る前に調相設備の投入や負荷制限を実施すべきである
- **設備計画**: PDP から得られる非線形関係に基づき、無効電力補償設備の最適容量を決定できる
- **気象対応**: 外気温度もマージンに影響するため、猛暑時には送電容量の低下を考慮した運用計画が必要である

### XAI の運用面での価値

1. **透明性**: ブラックボックスモデルの予測根拠を運用者に説明できる
2. **信頼性向上**: 物理法則と整合する説明が得られることで、AI判断への信頼が醸成される
3. **規制対応**: 電力系統の安全性に関わる判断には説明責任が求められるため、XAI は規制要件への適合に貢献する
4. **知識発見**: 運用者の経験則では見落とされがちな特徴量間の交互作用を発見できる

---

## 7. ファイル構成

```
problem4_power/
├── case1_power_flow/
│   └── generate_and_train.py    # 潮流異常検知: データ生成 + モデル学習
├── case2_voltage/
│   └── generate_and_train.py    # 電圧安定性: データ生成 + モデル学習
├── xai_analysis.py              # XAI 分析 (SHAP, LIME, PI, PDP)
├── report.md                    # 本レポート
└── results/                     # 出力ディレクトリ
    ├── case1_models.pkl
    ├── case1_confusion_matrix.png
    ├── case1_shap_summary.png
    ├── case1_shap_bar.png
    ├── case1_shap_waterfall0.png
    ├── case1_shap_waterfall1.png
    ├── case1_shap_waterfall2.png
    ├── case1_lime_sample0.png
    ├── case1_lime_sample1.png
    ├── case1_lime_sample2.png
    ├── case1_pi.png
    ├── case1_pdp.png
    ├── case2_models.pkl
    ├── case2_actual_vs_predicted.png
    ├── case2_shap_summary.png
    ├── case2_shap_bar.png
    ├── case2_shap_waterfall0.png
    ├── case2_shap_waterfall1.png
    ├── case2_shap_waterfall2.png
    ├── case2_lime_sample0.png
    ├── case2_lime_sample1.png
    ├── case2_lime_sample2.png
    ├── case2_pi.png
    ├── case2_pdp.png
    └── power_comparison.png
```

## 8. 実行手順

```bash
# 1. Case 1: 潮流異常検知のデータ生成とモデル学習
python problem4_power/case1_power_flow/generate_and_train.py

# 2. Case 2: 電圧安定性のデータ生成とモデル学習
python problem4_power/case2_voltage/generate_and_train.py

# 3. XAI 分析 (Case 1, Case 2 の実行後に実行)
python problem4_power/xai_analysis.py
```

---

*本レポートは Problem 4 (電力系統における XAI 分析) の概要をまとめたものである。具体的な数値結果はスクリプト実行後に出力される。*
