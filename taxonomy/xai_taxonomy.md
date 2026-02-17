# XAI（説明可能なAI）手法の体系的分類

---

## 目次

1. [概要](#1-概要)
2. [大分類: モデル固有 vs 事後的](#2-大分類-モデル固有-intrinsic-vs-事後的-post-hoc)
3. [中分類: グローバル vs ローカル](#3-中分類-グローバル-vs-ローカル)
4. [小分類: 手法カテゴリ](#4-小分類-手法カテゴリ)
5. [各手法の詳細](#5-各手法の詳細)
6. [手法選択フローチャート](#6-手法選択フローチャート)
7. [ポジショニングマップ](#7-ポジショニングマップ)
8. [参考文献](#8-参考文献)

---

## 1. 概要

### 1.1 XAIの定義

**XAI (eXplainable Artificial Intelligence; 説明可能なAI)** とは、AIシステムの意思決定プロセスや予測根拠を人間が理解・検証できる形で提示するための技術・研究領域である。

> "Explainable AI refers to methods and techniques in the application of artificial intelligence such that the results of the solution can be understood by humans."
> — DARPA XAI Program (2017)

### 1.2 XAIの必要性

| 観点 | 説明 |
|------|------|
| **信頼性 (Trust)** | モデルの判断根拠を理解することで、人間がAIを適切に信頼できる |
| **公平性 (Fairness)** | バイアスの検出・是正が可能になる |
| **説明責任 (Accountability)** | EU AI Act・GDPR等の規制への対応（「説明を受ける権利」） |
| **デバッグ (Debugging)** | モデルが誤った特徴量に依存していないかを検証できる |
| **科学的発見 (Discovery)** | モデルが学習したパターンから新たな知見を得る |
| **安全性 (Safety)** | 医療・金融・自動運転等、高リスク領域での安全な運用 |

### 1.3 説明可能性 (Explainability) vs 解釈可能性 (Interpretability)

この二つの用語は文献によって定義が揺れるが、以下の区別が広く用いられる。

| | 解釈可能性 (Interpretability) | 説明可能性 (Explainability) |
|---|---|---|
| **定義** | モデルの内部構造・動作を人間が直接理解できる度合い | モデルの判断を人間に伝達可能な形で説明する能力 |
| **対象** | モデルそのもの | モデルの出力・振る舞い |
| **方向性** | 内側から外側へ（モデル構造 → 理解） | 外側から内側へ（出力 → 説明生成） |
| **例** | 決定木のルール、線形回帰の係数 | SHAP値による特徴量貢献度、Grad-CAMのヒートマップ |
| **別名** | Transparency (透明性) | Post-hoc Explanation (事後的説明) |

**Lipton (2018)** は解釈可能性を以下の3段階に分類した:

1. **Simulatability（模倣可能性）**: 人間がモデル全体を頭の中でシミュレートできる
2. **Decomposability（分解可能性）**: 各パラメータに直感的な意味がある
3. **Algorithmic Transparency（アルゴリズムの透明性）**: 学習アルゴリズムの挙動が数学的に理解できる

---

## 2. 大分類: モデル固有 (Intrinsic) vs 事後的 (Post-hoc)

XAI手法の最も基本的な分類軸は、説明がモデルの **設計段階** で組み込まれているか、**学習後** に別途生成されるかである。

### 2.1 モデル固有の説明可能性 (Intrinsic Interpretability)

モデルの構造自体が解釈可能であり、追加の説明手法を必要としない。

```
┌─────────────────────────────────────────────────────┐
│           Intrinsic (モデル固有)                      │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  線形回帰     │  │  決定木       │  │ ルールベース│ │
│  │  (Linear     │  │  (Decision   │  │ (Rule-     │ │
│  │   Regression)│  │   Tree)      │  │  based)    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  GAM         │  │  ベイジアン   │  │  k-NN      │ │
│  │  (一般化加法  │  │  ルール      │  │            │ │
│  │   モデル)     │  │  リスト      │  │            │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

| モデル | 解釈の仕組み | 制約 |
|--------|------------|------|
| **線形回帰 (Linear Regression)** | 係数 $w_i$ が各特徴量の寄与を直接表す | 非線形関係の表現力が低い |
| **ロジスティック回帰 (Logistic Regression)** | オッズ比 $e^{w_i}$ が解釈可能 | 同上 |
| **決定木 (Decision Tree)** | if-then ルールの連鎖として可視化 | 深い木は解釈困難 |
| **ルールベースモデル (Rule-based)** | 明示的なルール集合 | 表現力の制限 |
| **GAM (Generalized Additive Model)** | $g(E[y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots$ | 交互作用の表現が限定的 |
| **k-NN** | 類似事例の提示が説明となる | 高次元で性能低下 |
| **ベイジアンルールリスト (BRL)** | 確率的なif-thenルール | スケーラビリティ |

### 2.2 事後的説明 (Post-hoc Explanation)

学習済みモデルに対して、別途説明を生成する手法群。モデルの内部構造に依存する手法（**モデル依存**）と、任意のモデルに適用可能な手法（**モデル非依存**）に分かれる。

```
┌─────────────────────────────────────────────────────────────────┐
│                Post-hoc (事後的)                                  │
│                                                                  │
│  ┌─────────────────────────┐  ┌───────────────────────────────┐ │
│  │   モデル依存              │  │   モデル非依存                  │ │
│  │   (Model-specific)      │  │   (Model-agnostic)            │ │
│  │                         │  │                               │ │
│  │  ・Grad-CAM (CNN)       │  │  ・SHAP (KernelSHAP)         │ │
│  │  ・Integrated Gradients │  │  ・LIME                       │ │
│  │  ・DeepLIFT (NN)        │  │  ・Permutation Importance    │ │
│  │  ・TreeSHAP (Tree)      │  │  ・PDP / ICE                 │ │
│  │  ・Attention可視化       │  │  ・Anchors                   │ │
│  │                         │  │  ・反事実的説明               │ │
│  └─────────────────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 中分類: グローバル vs ローカル

### 3.1 グローバル説明 (Global Explanation)

**モデル全体** の挙動・意思決定パターンを説明する。

- モデルが「一般的に」どの特徴量を重視しているかを把握
- モデルの全体的な振る舞いの理解、検証、監査に有用

**代表的手法:**

| 手法 | 説明内容 |
|------|---------|
| Permutation Importance | 各特徴量の全体的な重要度 |
| PDP (Partial Dependence Plot) | 特徴量と予測値の平均的な関係 |
| ALE (Accumulated Local Effects) | 特徴量の局所的効果の累積 |
| Global Surrogate | モデル全体を解釈可能モデルで近似 |
| SHAP Summary Plot | 全サンプルのSHAP値の分布 |

### 3.2 ローカル説明 (Local Explanation)

**個別の予測** に対して、その判断根拠を説明する。

- 「なぜこの患者は高リスクと判定されたのか？」
- 「なぜこの画像は猫と分類されたのか？」

**代表的手法:**

| 手法 | 説明内容 |
|------|---------|
| SHAP (個別値) | 各特徴量の個別予測への貢献度 |
| LIME | 局所的な線形近似による説明 |
| Grad-CAM | 画像中の重要領域のヒートマップ |
| Integrated Gradients | 基準点からの勾配積分による帰属 |
| Counterfactual | 「何が変われば予測が変わるか」 |
| Anchors | 予測を固定する十分条件ルール |

### 3.3 グローバル vs ローカルの比較

| 観点 | グローバル | ローカル |
|------|----------|---------|
| **対象** | モデル全体 | 個別予測 |
| **粒度** | 粗い（平均的傾向） | 細かい（個別の根拠） |
| **用途** | モデル検証・監査・特徴量選択 | 個別判断の説明・デバッグ |
| **忠実性** | モデルの複雑さに依存 | 局所的には高い忠実性 |
| **計算量** | 一度計算すれば全体に適用 | 予測ごとに計算が必要 |
| **ユーザー** | データサイエンティスト・監査者 | エンドユーザー・意思決定者 |

---

## 4. 小分類: 手法カテゴリ

### 4.1 全体マップ

```
XAI手法
├── 勾配ベース (Gradient-based)
│   ├── Saliency Maps
│   ├── Grad-CAM / Grad-CAM++
│   ├── Integrated Gradients
│   ├── SmoothGrad
│   └── DeepLIFT
│
├── 摂動ベース (Perturbation-based)
│   ├── LIME
│   ├── Occlusion Sensitivity
│   ├── Anchors
│   └── CEM (Contrastive Explanation Method)
│
├── ゲーム理論ベース (Game-theoretic)
│   ├── KernelSHAP
│   ├── TreeSHAP
│   └── DeepSHAP
│
├── 近似モデルベース (Surrogate model)
│   ├── Global Surrogate
│   └── LIME (局所代理モデル)
│
├── 特徴量重要度 (Feature Importance)
│   ├── Permutation Importance
│   ├── MDI (Mean Decrease in Impurity)
│   └── MDA (Mean Decrease in Accuracy)
│
├── 可視化ベース (Visualization)
│   ├── PDP (Partial Dependence Plot)
│   ├── ICE (Individual Conditional Expectation)
│   └── ALE (Accumulated Local Effects)
│
├── 注意機構ベース (Attention-based)
│   ├── Attention Visualization
│   └── BertViz
│
├── 反事実的説明 (Counterfactual Explanation)
│   ├── DiCE
│   └── Wachter et al. (2017)
│
└── 概念ベース (Concept-based)
    ├── TCAV (Testing with Concept Activation Vectors)
    └── ACE (Automated Concept-based Explanations)
```

### 4.2 カテゴリ別概要表

| カテゴリ | 基本原理 | 主な対象モデル | スコープ |
|---------|---------|-------------|---------|
| 勾配ベース | 出力の入力に対する勾配を利用 | ニューラルネットワーク | ローカル |
| 摂動ベース | 入力を変化させて出力への影響を観測 | モデル非依存 | ローカル |
| ゲーム理論ベース | Shapley値による公平な貢献度配分 | モデル非依存（最適化版あり） | ローカル/グローバル |
| 近似モデルベース | 解釈可能モデルでブラックボックスを近似 | モデル非依存 | ローカル/グローバル |
| 特徴量重要度 | 特徴量の予測への全体的影響を定量化 | モデル非依存（一部モデル固有） | グローバル |
| 可視化ベース | 特徴量と予測の関係をプロットで表現 | モデル非依存 | グローバル |
| 注意機構ベース | Attention重みを説明として利用 | Transformer系モデル | ローカル |
| 反事実的説明 | 予測を変える最小の入力変化を探索 | モデル非依存 | ローカル |
| 概念ベース | 人間が理解可能な概念で説明 | ニューラルネットワーク | グローバル |

---

## 5. 各手法の詳細

### 5.1 SHAP (SHapley Additive exPlanations)

#### 概要

SHAP は協力ゲーム理論における **Shapley値** をベースに、各特徴量の予測への貢献度を公平に配分する統一的フレームワークである。Lundberg & Lee (2017) により提案された。

#### 数学的定義

特徴量 $i$ のShapley値は以下で定義される:

$$\phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\;(|N|-|S|-1)!}{|N|!} \left[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S) \right]$$

ここで:
- $N$: 全特徴量の集合（$|N| = p$）
- $S$: $N \setminus \{i\}$ の部分集合
- $f_S(x_S)$: 特徴量集合 $S$ のみを使った場合のモデル出力の期待値

SHAP値は以下の **4つの公理** を満たす唯一の帰属手法である:

1. **効率性 (Efficiency)**: $\sum_{i=1}^{p} \phi_i = f(x) - E[f(X)]$
2. **対称性 (Symmetry)**: 同等の貢献をする特徴量は同じ値を持つ
3. **ダミー (Dummy)**: 寄与しない特徴量の値はゼロ
4. **加法性 (Additivity)**: モデルの和に対してSHAP値も加法的

#### バリエーション

| 手法 | 計算方法 | 対象モデル | 計算量 |
|------|---------|----------|-------|
| **KernelSHAP** | 重み付き線形回帰による近似 | モデル非依存 | $O(2^p)$（サンプリングで近似） |
| **TreeSHAP** | 木構造を利用した厳密計算 | 決定木・ランダムフォレスト・XGBoost等 | $O(TLD^2)$（$T$: 木の数, $L$: 葉の数, $D$: 深さ） |
| **DeepSHAP** | DeepLIFT + Shapley値 | ニューラルネットワーク | バックプロパゲーションと同等 |
| **LinearSHAP** | 解析的に計算 | 線形モデル | $O(p)$ |

#### アルゴリズム概要（KernelSHAP）

```
1. 説明対象のインスタンス x を固定
2. 特徴量の部分集合 z' ∈ {0, 1}^p をサンプリング
3. 各 z' に対して:
   a. z' = 1 の特徴量は x の値を使用
   b. z' = 0 の特徴量は背景データからサンプリング
   c. モデル予測値 f(h_x(z')) を計算
4. SHAPカーネル重み π_x(z') で重み付き線形回帰を実行:
   π_x(z') = (p - 1) / (C(p, |z'|) · |z'| · (p - |z'|))
5. 回帰係数が SHAP 値 φ_i となる
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| 理論的に堅牢（Shapley値の公理を満たす） | KernelSHAPは特徴量数が多いと計算コストが高い |
| 局所的・大域的説明の両方に対応 | 特徴量間の相関がある場合、非現実的な組み合わせが生じうる |
| 一貫した特徴量帰属（Efficiency公理） | 背景データの選択に結果が依存する |
| 可視化が豊富（force plot, summary plot等） | カテゴリ特徴量の扱いに注意が必要 |

#### 適用可能なモデル・データ型

- **モデル**: 任意（KernelSHAP）、ツリー系（TreeSHAP）、NN（DeepSHAP）
- **データ型**: 表形式、画像（SuperPixel化）、テキスト（トークン単位）

---

### 5.2 LIME (Local Interpretable Model-agnostic Explanations)

#### 概要

LIME は個別の予測に対して、その近傍で **局所的に解釈可能なモデル（代理モデル）** を学習することで説明を生成する。Ribeiro et al. (2016) により提案された。

#### 数学的定義

LIME の最適化問題は以下で定式化される:

$$\xi(x) = \arg\min_{g \in G} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

ここで:
- $f$: 説明対象のブラックボックスモデル
- $g \in G$: 解釈可能なモデルの集合（通常は線形モデル）
- $\pi_x$: インスタンス $x$ の近傍を定義するカーネル関数
- $\mathcal{L}(f, g, \pi_x)$: $f$ と $g$ の近傍における不一致度（忠実性損失）
- $\Omega(g)$: モデル $g$ の複雑性ペナルティ

具体的には:

$$\mathcal{L}(f, g, \pi_x) = \sum_{z, z' \in \mathcal{Z}} \pi_x(z) \left( f(z) - g(z') \right)^2$$

近傍カーネル:

$$\pi_x(z) = \exp\left( -\frac{D(x, z)^2}{\sigma^2} \right)$$

#### アルゴリズム概要

```
1. 説明対象のインスタンス x を入力
2. x の近傍でデータを摂動生成:
   - 表形式: 各特徴量をランダムに変化
   - 画像: SuperPixel単位でON/OFF
   - テキスト: 単語をランダムに除去
3. 各摂動サンプル z に対してモデル予測 f(z) を取得
4. x からの距離に基づくカーネル重み π_x(z) を計算
5. 重み付き線形回帰で局所代理モデル g を学習:
   g(z') = w_0 + w_1 z'_1 + w_2 z'_2 + ... + w_k z'_k
6. 回帰係数 w_i が各特徴量の局所的重要度
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| モデル非依存（任意のブラックボックスに適用可能） | 摂動の方法・カーネル幅に結果が敏感 |
| 直感的な説明（線形モデルの係数） | 安定性が低い（同じ入力でも結果が変動しうる） |
| 画像・テキスト・表形式データに適用可能 | 局所的な線形近似の忠実性が保証されない |
| 実装が比較的容易 | 特徴量間の相関を無視した摂動が問題になりうる |

#### 適用可能なモデル・データ型

- **モデル**: 任意（予測関数にアクセスできれば良い）
- **データ型**: 表形式、画像（SuperPixel）、テキスト（BoW表現）

#### 計算量

- $O(n \cdot p)$（$n$: 摂動サンプル数、$p$: 特徴量数）
- モデル推論回数: 通常 $n = 1000 \sim 5000$ 回

---

### 5.3 Grad-CAM (Gradient-weighted Class Activation Mapping)

#### 概要

Grad-CAM は CNN の最終畳み込み層の活性化マップに対して、分類スコアの勾配で重み付けすることで、判断に重要な画像領域をヒートマップとして可視化する。Selvaraju et al. (2017) により提案された。

#### 数学的定義

クラス $c$ に対する Grad-CAM のヒートマップ:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c \cdot A^k \right)$$

重み $\alpha_k^c$ は以下で計算される:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

ここで:
- $y^c$: クラス $c$ のスコア（softmax前）
- $A^k$: 最終畳み込み層の $k$ 番目の特徴量マップ
- $A^k_{ij}$: 特徴量マップの位置 $(i, j)$ の値
- $Z$: 特徴量マップの空間サイズ（$= H \times W$）
- ReLU: 正の寄与のみを可視化するため

#### アルゴリズム概要

```
1. 入力画像をCNNに順伝播
2. 対象クラス c のスコア y^c を取得
3. y^c を最終畳み込み層の各特徴量マップ A^k に対して逆伝播
4. 勾配 ∂y^c/∂A^k を空間方向に Global Average Pooling → 重み α_k^c
5. 特徴量マップの重み付き和を計算: Σ_k α_k^c · A^k
6. ReLU を適用（正の寄与のみ）
7. 入力画像サイズにアップサンプリング
8. 入力画像に重畳してヒートマップとして表示
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| 微分可能なCNNアーキテクチャに汎用的に適用可能 | 最終畳み込み層の解像度に制約される |
| 計算が高速（1回の逆伝播で完了） | 複数の重要領域がある場合に粗くなる |
| クラスごとのヒートマップ生成が可能 | 細粒度の空間情報が失われる |
| 学習済みモデルの変更不要 | 中間層の情報を活用できない |

#### 適用可能なモデル・データ型

- **モデル**: CNN（VGG, ResNet, DenseNet等）、一部Transformerにも適用可能
- **データ型**: 画像

#### 計算量

- $O(1)$ 回の順伝播 + $O(1)$ 回の逆伝播
- 非常に高速（リアルタイム処理可能）

#### 発展形

| 手法 | 改善点 |
|------|-------|
| **Grad-CAM++** | 重みの計算にピクセル単位の勾配の二次・三次導関数を使用。複数物体の検出が改善 |
| **Score-CAM** | 勾配を使わず、各チャネルのマスクを通した出力変化を重みとして使用 |
| **Eigen-CAM** | 特徴量マップの主成分分析に基づく |

---

### 5.4 Integrated Gradients (IG; 統合勾配法)

#### 概要

Integrated Gradients はベースライン（基準入力）から対象入力までの経路上で勾配を積分することで、各特徴量の帰属値を計算する。Sundararajan et al. (2017) により提案された。**公理的帰属法** の一つであり、2つの基本公理を満たすことが証明されている。

#### 数学的定義

特徴量 $i$ に対する Integrated Gradients:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

実装上はリーマン和で近似:

$$\text{IG}_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

ここで:
- $x$: 説明対象の入力
- $x'$: ベースライン（基準入力。画像なら黒画像、テキストならパディングトークン等）
- $F$: ニューラルネットワークの出力関数
- $m$: 積分の近似ステップ数（通常 $m = 20 \sim 300$）

#### 満たす公理

1. **完全性 (Completeness)**:

$$\sum_{i=1}^{n} \text{IG}_i(x) = F(x) - F(x')$$

全帰属値の合計がモデル出力の差分と一致する。

2. **感度 (Sensitivity)**:

入力の一つの特徴量だけが異なり、かつ予測が変わる場合、その特徴量のIG値は非ゼロとなる。

#### アルゴリズム概要

```
1. ベースライン x' を設定（黒画像、ゼロベクトル等）
2. x' から x までの直線経路上に m 個の補間点を生成:
   x_k = x' + (k/m) · (x - x'),  k = 0, 1, ..., m
3. 各補間点 x_k でモデルの勾配 ∂F/∂x_i を計算
4. 勾配を経路上で平均（リーマン和による積分近似）
5. スケーリング: (x_i - x'_i) を乗じて最終帰属値を得る
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| 公理的な正当性（完全性・感度を満たす） | ベースラインの選択に結果が依存 |
| 実装が比較的シンプル | ステップ数 $m$ に計算量が比例 |
| 入力空間で直感的な解釈が可能 | 非直線経路の方が良い場合がある |
| 勾配飽和問題を回避 | 計算コストがSaliency Mapの $m$ 倍 |

#### 適用可能なモデル・データ型

- **モデル**: 微分可能なモデル全般（NN、特にDNN）
- **データ型**: 画像、テキスト、表形式

#### 計算量

- $O(m)$ 回の順伝播 + 逆伝播（$m$: ステップ数）
- 通常 $m = 50 \sim 300$

---

### 5.5 Occlusion Sensitivity（オクルージョン感度分析）

#### 概要

入力の一部を体系的に隠蔽（マスク）し、予測の変化を観測することで重要な領域を特定する手法。Zeiler & Fergus (2014) により提案された。概念が単純で直感的な手法である。

#### 数学的定義

位置 $(i, j)$ を中心とするオクルージョンパッチ $P_{ij}$ を適用した際の感度:

$$S(i, j) = f(x) - f(x \odot M_{ij})$$

ここで:
- $f$: モデルの予測関数
- $x$: 入力（画像）
- $M_{ij}$: 位置 $(i, j)$ を中心とする領域を隠蔽するマスク
- $\odot$: マスク適用操作（隠蔽領域をゼロ or グレー等に置換）

#### アルゴリズム概要

```
1. オクルージョンパッチのサイズとストライドを設定
2. 元の入力に対するモデル予測 f(x) を記録
3. パッチを入力上でスライドさせる:
   for 各位置 (i, j):
     a. パッチ領域を隠蔽値（グレー等）で置換
     b. 隠蔽後の予測 f(x_masked) を計算
     c. 感度 S(i,j) = f(x) - f(x_masked) を記録
4. 感度マップ S をヒートマップとして可視化
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| 概念が非常に単純で直感的 | パッチサイズの選択がヒューリスティック |
| モデル非依存（ブラックボックスでOK） | 計算コストが高い（多数の推論が必要） |
| 勾配計算不要 | 特徴量間の相互作用を捉えにくい |
| 実装が容易 | パッチサイズにより解像度が制限される |

#### 適用可能なモデル・データ型

- **モデル**: 任意
- **データ型**: 主に画像（表形式やテキストにも適用可能）

#### 計算量

- $O\left(\frac{H \times W}{s^2}\right)$ 回のモデル推論（$H, W$: 画像サイズ、$s$: ストライド）
- パッチサイズが小さいほど高解像度だが計算量増大

---

### 5.6 Permutation Importance（順列重要度）

#### 概要

各特徴量の値をランダムにシャッフル（順列を変更）し、モデル性能の低下度合いで重要度を測定する。Breiman (2001) がランダムフォレストで導入し、Fisher et al. (2019) がモデル非依存版として一般化した。

#### 数学的定義

特徴量 $j$ の Permutation Importance:

$$\text{PI}_j = s(f, X, y) - \frac{1}{K} \sum_{k=1}^{K} s(f, X^{(k,j)}_{\text{perm}}, y)$$

ここで:
- $s(f, X, y)$: 元データにおけるモデルの評価指標（精度、MSE等）
- $X^{(k,j)}_{\text{perm}}$: 特徴量 $j$ の列を $k$ 回目にシャッフルしたデータ
- $K$: シャッフルの繰り返し回数

$\text{PI}_j > 0$ : 特徴量 $j$ が重要（シャッフルで性能低下）

$\text{PI}_j \approx 0$ : 特徴量 $j$ は不要（シャッフルしても性能変化なし）

#### アルゴリズム概要

```
1. モデル f を検証データ (X, y) で評価 → ベースラインスコア s_0
2. 各特徴量 j = 1, ..., p について:
   for k = 1, ..., K:
     a. X の特徴量 j の列をランダムにシャッフル → X_perm
     b. シャッフル後のスコア s_k = s(f, X_perm, y) を計算
   c. PI_j = s_0 - mean(s_1, ..., s_K)
3. PI値の大きい順にソート → 重要度ランキング
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| モデル非依存 | 相関の強い特徴量間で重要度が分散する |
| 実装が非常にシンプル | 特徴量のシャッフルが非現実的なデータ分布を生む |
| 訓練データ・テストデータの両方で計算可能 | 計算コストが特徴量数に比例 |
| 直感的に理解しやすい | 交互作用効果を個別に分離できない |

#### 適用可能なモデル・データ型

- **モデル**: 任意
- **データ型**: 主に表形式データ

#### 計算量

- $O(K \cdot p \cdot n)$ （$K$: 繰り返し回数、$p$: 特徴量数、$n$: サンプル数）

---

### 5.7 PDP (Partial Dependence Plot) / ICE (Individual Conditional Expectation)

#### 概要

**PDP** は一つ（または二つ）の特徴量とモデル予測の **平均的な関係** をプロットする手法で、Friedman (2001) により提案された。**ICE** は各サンプルごとに個別の依存関係を描画し、PDP を構成する個別の曲線群を可視化する（Goldstein et al., 2015）。

#### 数学的定義

**PDP（部分依存関数）:**

$$\hat{f}_S(x_S) = E_{X_C}\left[ f(x_S, X_C) \right] = \frac{1}{n} \sum_{i=1}^{n} f(x_S, x_C^{(i)})$$

ここで:
- $x_S$: 関心のある特徴量の値（1つまたは2つ）
- $X_C$: その他の特徴量（補完特徴量）
- $x_C^{(i)}$: $i$ 番目のサンプルの補完特徴量値
- $n$: データセットのサンプル数

**ICE（個別条件付き期待値）:**

$$\hat{f}^{(i)}_S(x_S) = f(x_S, x_C^{(i)})$$

各サンプル $i$ ごとの依存曲線。PDPはICE曲線の平均:

$$\text{PDP}(x_S) = \frac{1}{n} \sum_{i=1}^{n} \text{ICE}^{(i)}(x_S)$$

#### アルゴリズム概要

```
[PDP]
1. 関心特徴量 x_S のグリッド値 {v_1, v_2, ..., v_G} を設定
2. 各グリッド値 v_g について:
   a. 全サンプルの特徴量 x_S を v_g に固定
   b. モデル予測 f(v_g, x_C^(i)) を全サンプルで計算
   c. 平均を取る: PDP(v_g) = (1/n) Σ f(v_g, x_C^(i))
3. (v_g, PDP(v_g)) をプロット

[ICE]
1. 各サンプル i ごとに:
   各グリッド値 v_g で f(v_g, x_C^(i)) を計算
2. サンプルごとに曲線をプロット
```

#### 長所・短所

| 長所 | 短所 |
|------|------|
| 直感的で理解しやすい | 特徴量間の相関を無視（非現実的な組み合わせ） |
| モデル非依存 | 高次元（3変数以上）の可視化が困難 |
| ICEで個別の異質性を確認可能 | 計算量がグリッド数×サンプル数に比例 |
| 非線形関係の発見に有効 | 相互作用効果の平均化による情報損失（PDPの場合） |

**ALE (Accumulated Local Effects)** は PDP の相関問題を解決する改良手法:

$$\text{ALE}(x_S) = \int_{z_{\min}}^{x_S} E_{X_C | X_S = z}\left[ \frac{\partial f(z, X_C)}{\partial z} \right] dz$$

#### 適用可能なモデル・データ型

- **モデル**: 任意
- **データ型**: 表形式（連続変数・カテゴリ変数の両方）

#### 計算量

- PDP: $O(G \cdot n)$（$G$: グリッド数、$n$: サンプル数）
- ICE: $O(G \cdot n)$（同上だが全曲線を描画）

---

### 5.8 その他の重要手法

#### 5.8.1 Anchors

**定義**: 予測を高確率で固定する「十分条件」ルールを発見する手法 (Ribeiro et al., 2018)。

$$A(x) \text{ is an anchor if } E_{z \sim \mathcal{D}(\cdot|A)}[1_{f(z) = f(x)}] \geq \tau$$

- ルール $A$ が成立する入力に対して、確率 $\tau$ 以上で同じ予測が維持される
- 例: 「年収 > 50万 AND 勤続年数 > 5年 → ローン承認（精度95%）」

#### 5.8.2 TCAV (Testing with Concept Activation Vectors)

**定義**: 人間が定義した「概念」（例: 「縞模様」）の方向ベクトルを使い、モデルの感度を測定する (Kim et al., 2018)。

$$\text{TCAV}_Q^l(C) = \frac{|\{x \in X_k : S_{C,k,l}(x) > 0\}|}{|X_k|}$$

- $C$: 概念（ユーザーが定義）
- $l$: 対象のニューラルネットワーク層
- $S_{C,k,l}(x)$: 概念方向への感度

#### 5.8.3 反事実的説明 (Counterfactual Explanation)

**定義**: 「最小の入力変更で予測が変わる」反事実的事例を生成する (Wachter et al., 2017)。

$$\arg\min_{x'} \max_\lambda \; \lambda(f(x') - y')^2 + d(x, x')$$

- $x'$: 反事実的事例
- $y'$: 望ましい予測結果
- $d(x, x')$: 元の入力からの距離

**DiCE** (Mothilal et al., 2020) は多様な反事実的事例を効率的に生成する手法。

#### 5.8.4 SmoothGrad

勾配ノイズを低減するため、入力にガウスノイズを加えた複数サンプルの勾配を平均する (Smilkov et al., 2017):

$$\text{SmoothGrad}(x) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f(x + \epsilon_k)}{\partial x}, \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2)$$

#### 5.8.5 DeepLIFT

基準入力との差分に基づいて帰属値を計算する (Shrikumar et al., 2017):

$$C_{\Delta x_i \Delta t} = m_{\Delta x_i \Delta t} \cdot \Delta x_i$$

乗数 $m$ はバックプロパゲーションのルール（Rescale / RevealCancel）で伝播される。

---

## 6. 手法選択フローチャート

### 6.1 テキストベースフローチャート

```
START: XAI手法を選択したい
│
├─ Q1: データの種類は？
│   │
│   ├─ [表形式データ]
│   │   │
│   │   ├─ Q2: 説明のスコープは？
│   │   │   │
│   │   │   ├─ [グローバル（モデル全体の理解）]
│   │   │   │   │
│   │   │   │   ├─ Q3: モデルの種類は？
│   │   │   │   │   ├─ [ツリー系] → Permutation Importance + PDP/ICE + TreeSHAP Summary
│   │   │   │   │   ├─ [線形モデル] → 係数の解釈 + PDP
│   │   │   │   │   └─ [任意/NN] → Permutation Importance + PDP/ICE + KernelSHAP Summary
│   │   │   │   │
│   │   │   │   └─ 特徴量間相関が強い場合 → ALE を PDP の代替に
│   │   │   │
│   │   │   └─ [ローカル（個別予測の説明）]
│   │   │       │
│   │   │       ├─ Q3: モデルの種類は？
│   │   │       │   ├─ [ツリー系] → TreeSHAP (高速・正確)
│   │   │       │   ├─ [NN] → DeepSHAP or KernelSHAP
│   │   │       │   └─ [任意] → KernelSHAP or LIME
│   │   │       │
│   │   │       ├─ ルール形式の説明が欲しい → Anchors
│   │   │       └─ 「何を変えれば？」を知りたい → DiCE (反事実的説明)
│   │   │
│   │   └─ [両方] → SHAP (Summary Plot + Force Plot の組み合わせ)
│   │
│   ├─ [画像データ]
│   │   │
│   │   ├─ Q2: モデルの種類は？
│   │   │   │
│   │   │   ├─ [CNN]
│   │   │   │   ├─ 高速に領域を知りたい → Grad-CAM
│   │   │   │   ├─ ピクセル精度の帰属が必要 → Integrated Gradients
│   │   │   │   ├─ ノイズの少ない帰属が必要 → SmoothGrad
│   │   │   │   └─ 勾配不要/ブラックボックス → Occlusion Sensitivity or LIME (画像)
│   │   │   │
│   │   │   └─ [Vision Transformer]
│   │   │       ├─ Attention Rollout
│   │   │       └─ Integrated Gradients
│   │   │
│   │   └─ 概念レベルの説明が必要 → TCAV
│   │
│   ├─ [テキストデータ]
│   │   │
│   │   ├─ Q2: モデルの種類は？
│   │   │   ├─ [BERT等 Transformer] → Attention可視化 (BertViz) + IG
│   │   │   ├─ [任意] → LIME (テキスト) or SHAP (テキスト)
│   │   │   └─ ルール形式 → Anchors
│   │   │
│   │   └─ 概念レベル → TCAV
│   │
│   └─ [時系列データ]
│       ├─ 特徴量重要度 → Permutation Importance (時間窓単位)
│       ├─ 局所説明 → SHAP (時系列対応) or LIME
│       └─ 時間的パターン → Attention可視化 (Transformer系)
│
END
```

### 6.2 推奨手法の組み合わせ（ユースケース別）

| ユースケース | 推奨手法の組み合わせ |
|------------|-------------------|
| **表形式 + ツリー系モデル + 全方位** | TreeSHAP (ローカル+グローバル) + PDP/ICE + Permutation Importance |
| **表形式 + NN + ローカル** | KernelSHAP or LIME + 反事実的説明 (DiCE) |
| **画像分類 + CNN** | Grad-CAM (概要) + Integrated Gradients (精密) + Occlusion (検証) |
| **テキスト分類 + BERT** | Attention可視化 + IG + LIME (テキスト) |
| **医療AI（規制対応）** | SHAP + Anchors + 反事実的説明 |
| **金融モデル監査** | Permutation Importance + PDP + SHAP + Global Surrogate |

---

## 7. ポジショニングマップ

### 7.1 局所性 × モデル依存性

```
         モデル固有 (Intrinsic)                  モデル非依存 (Agnostic)
              ◄──────────────────────────────────────────────►

  グ    │                                                        │
  ロ    │  ・線形回帰の係数        ・PDP / ICE                    │
  ー    │  ・決定木の可視化        ・ALE                          │
  バ    │  ・GAM                  ・Permutation Importance        │
  ル    │  ・MDI (ツリー)         ・Global Surrogate              │
        │                        ・SHAP Summary Plot             │
  ▲    │                                                        │
  │    │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
  │    │                                                        │
  ▼    │                                                        │
        │  ・TreeSHAP             ・KernelSHAP                   │
  ロ    │  ・Grad-CAM             ・LIME                          │
  ー    │  ・Integrated Gradients ・Anchors                       │
  カ    │  ・DeepLIFT             ・Occlusion Sensitivity         │
  ル    │  ・DeepSHAP             ・反事実的説明 (DiCE)             │
        │  ・Attention可視化      ・CEM                           │
        │  ・SmoothGrad                                          │
        │                                                        │
```

### 7.2 計算コスト × 説明の忠実性

```
  高忠実性 ▲
           │  ・TreeSHAP              ・KernelSHAP
           │  ・Integrated Gradients
           │
           │  ・DeepSHAP    ・LIME
           │  ・DeepLIFT    ・Permutation Importance
           │
           │  ・Grad-CAM    ・PDP/ICE
           │
           │  ・Saliency Map ・Occlusion Sensitivity
           │
  低忠実性 │───────────────────────────────────────────► 高計算コスト
           低計算コスト
```

### 7.3 比較表

| 手法 | スコープ | モデル依存性 | 計算コスト | 忠実性 | 安定性 | データ型 |
|------|---------|------------|----------|-------|-------|---------|
| **SHAP (KernelSHAP)** | ローカル/グローバル | 非依存 | 高 | 高 | 高 | 表形式/画像/テキスト |
| **SHAP (TreeSHAP)** | ローカル/グローバル | ツリー系 | 低 | 高 | 高 | 表形式 |
| **LIME** | ローカル | 非依存 | 中 | 中 | 低 | 表形式/画像/テキスト |
| **Grad-CAM** | ローカル | CNN | 低 | 中 | 高 | 画像 |
| **Integrated Gradients** | ローカル | NN | 中 | 高 | 中 | 画像/テキスト/表形式 |
| **Occlusion** | ローカル | 非依存 | 高 | 中 | 高 | 画像 |
| **Permutation Importance** | グローバル | 非依存 | 中 | 中 | 中 | 表形式 |
| **PDP / ICE** | グローバル | 非依存 | 中 | 低〜中 | 高 | 表形式 |
| **ALE** | グローバル | 非依存 | 中 | 中〜高 | 高 | 表形式 |
| **Anchors** | ローカル | 非依存 | 高 | 高 | 中 | 表形式/テキスト |
| **DiCE** | ローカル | 非依存 | 中 | 高 | 中 | 表形式 |
| **TCAV** | グローバル | NN | 中 | 中 | 中 | 画像 |
| **Attention可視化** | ローカル | Transformer | 低 | 低〜中 | 高 | テキスト/画像 |
| **DeepLIFT** | ローカル | NN | 低 | 中〜高 | 高 | 画像/表形式 |
| **SmoothGrad** | ローカル | NN | 中 | 中 | 中 | 画像 |

> **忠実性 (Faithfulness)**: 説明がモデルの実際の意思決定をどれだけ正確に反映するか
>
> **安定性 (Stability)**: 類似した入力に対して類似した説明が生成されるか

---

## 8. 参考文献

### 8.1 サーベイ・総合文献

| 文献 | 概要 |
|------|------|
| Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* 2nd ed. | XAIの包括的教科書（無料公開） |
| Barredo Arrieta, A. et al. (2020). "Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI." *Information Fusion*, 58, 82-115. | XAI分類体系のサーベイ |
| Guidotti, R. et al. (2018). "A survey of methods for explaining black box models." *ACM Computing Surveys*, 51(5), 1-42. | ブラックボックス説明手法の体系的調査 |

### 8.2 個別手法の主要論文

#### SHAP

- Lundberg, S. M. & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS 2017*. [[paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)]
- Lundberg, S. M. et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56-67.

#### LIME

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "'Why should I trust you?': Explaining the predictions of any classifier." *KDD 2016*. [[paper](https://arxiv.org/abs/1602.04938)]

#### Grad-CAM

- Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." *ICCV 2017*. [[paper](https://arxiv.org/abs/1610.02391)]
- Chattopadhay, A. et al. (2018). "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks." *WACV 2018*.

#### Integrated Gradients

- Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic attribution for deep networks." *ICML 2017*. [[paper](https://arxiv.org/abs/1703.01365)]

#### Occlusion

- Zeiler, M. D. & Fergus, R. (2014). "Visualizing and understanding convolutional networks." *ECCV 2014*. [[paper](https://arxiv.org/abs/1311.1901)]

#### Permutation Importance

- Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5-32.
- Fisher, A., Rudin, C., & Dominici, F. (2019). "All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously." *JMLR*, 20(177), 1-81.

#### PDP / ICE / ALE

- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine." *Annals of Statistics*, 29(5), 1189-1232.
- Goldstein, A. et al. (2015). "Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation." *JCGS*, 24(1), 44-65.
- Apley, D. W. & Zhu, J. (2020). "Visualizing the effects of predictor variables in black box supervised learning models." *JRSS-B*, 82(4), 1059-1086.

#### Anchors

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). "Anchors: High-precision model-agnostic explanations." *AAAI 2018*. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11491)]

#### 反事実的説明

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." *Harvard Journal of Law & Technology*, 31(2).
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). "Explaining machine learning classifiers through diverse counterfactual explanations." *FAT* 2020*.

#### TCAV

- Kim, B. et al. (2018). "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)." *ICML 2018*. [[paper](https://arxiv.org/abs/1711.11279)]

#### DeepLIFT

- Shrikumar, A., Greenside, P., & Kundaje, A. (2017). "Learning important features through propagating activation differences." *ICML 2017*. [[paper](https://arxiv.org/abs/1704.02685)]

#### SmoothGrad

- Smilkov, D. et al. (2017). "SmoothGrad: Removing noise by adding noise." *ICML Workshop*. [[paper](https://arxiv.org/abs/1706.03825)]

#### Saliency Maps

- Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). "Deep inside convolutional networks: Visualising image classification models and saliency maps." *ICLR Workshop 2014*. [[paper](https://arxiv.org/abs/1312.6034)]

#### 解釈可能性の理論的基盤

- Lipton, Z. C. (2018). "The mythos of model interpretability." *Queue*, 16(3), 31-57.
- Doshi-Velez, F. & Kim, B. (2017). "Towards a rigorous science of interpretable machine learning." *arXiv:1702.08608*.

#### Attention と説明可能性

- Jain, S. & Wallace, B. C. (2019). "Attention is not explanation." *NAACL 2019*.
- Wiegreffe, S. & Pinter, Y. (2019). "Attention is not not explanation." *EMNLP 2019*.

---

> **本資料について**: この分類体系は講義・プレゼンテーション用にまとめたものである。各手法の詳細な実装については、対応する問題演習（problem1〜problem4）を参照されたい。
