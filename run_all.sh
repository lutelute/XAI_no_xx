#!/bin/bash
# XAI Analysis Platform - Run All
# 全分析スクリプトを順次実行
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  XAI 解析プラットフォーム — 一括実行"
echo "============================================"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

run_script() {
    local script="$1"
    local name="$2"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}▶ $name${NC}"
    echo -e "${BLUE}  $script${NC}"
    echo ""
    if python "$script"; then
        echo -e "${GREEN}✓ 完了: $name${NC}"
    else
        echo -e "${RED}✗ エラー: $name${NC}"
        echo -e "${RED}  スクリプト: $script${NC}"
        return 1
    fi
    echo ""
}

# =============================================
# Problem 1: 住宅価格予測 (California Housing)
# =============================================
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Problem 1: 住宅価格予測                  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

run_script "problem1_housing/00_train_model.py"        "P1: モデル学習 (RF, GBT)"
run_script "problem1_housing/01_shap_analysis.py"       "P1: SHAP 分析"
run_script "problem1_housing/02_lime_analysis.py"       "P1: LIME 分析"
run_script "problem1_housing/03_permutation_analysis.py" "P1: Permutation Importance"
run_script "problem1_housing/04_pdp_ice_analysis.py"    "P1: PDP / ICE 分析"
run_script "problem1_housing/05_comparison.py"          "P1: 手法比較"

# =============================================
# Problem 2: 動物分類 (CIFAR-10 + ResNet18)
# =============================================
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Problem 2: 動物分類 (CIFAR-10)           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

run_script "problem2_animals/00_train_model.py"         "P2: ResNet18 学習"
run_script "problem2_animals/01_gradcam.py"             "P2: Grad-CAM 分析"
run_script "problem2_animals/02_integrated_gradients.py" "P2: Integrated Gradients"
run_script "problem2_animals/03_occlusion.py"           "P2: Occlusion Sensitivity"
run_script "problem2_animals/04_comparison.py"          "P2: 手法比較"

# =============================================
# Problem 3: 顔識別 (LFW)
# =============================================
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Problem 3: 顔識別 (LFW)                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

run_script "problem3_faces/00_train_model.py"           "P3: CNN 学習 (LFW)"
run_script "problem3_faces/01_gradcam.py"               "P3: Grad-CAM 分析"
run_script "problem3_faces/02_integrated_gradients.py"   "P3: Integrated Gradients"
run_script "problem3_faces/03_occlusion.py"             "P3: Occlusion Sensitivity"
run_script "problem3_faces/04_comparison.py"            "P3: 手法比較"

# =============================================
# Problem 4: 電力系統
# =============================================
echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Problem 4: 電力系統                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

run_script "problem4_power/case1_power_flow/generate_and_train.py" "P4-Case1: 潮流異常検知 データ生成+学習"
run_script "problem4_power/case2_voltage/generate_and_train.py"    "P4-Case2: 電圧安定性 データ生成+学習"
run_script "problem4_power/xai_analysis.py"                        "P4: XAI 分析 (両ケース)"

# =============================================
# 完了
# =============================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  全分析完了！${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "結果ファイル:"
echo "  problem1_housing/results/"
echo "  problem2_animals/results/"
echo "  problem3_faces/results/"
echo "  problem4_power/results/"
echo ""
echo "ビューアを開く:"
echo "  open viewer.html"
echo ""
