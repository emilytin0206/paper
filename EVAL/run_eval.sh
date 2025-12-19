#!/bin/bash
set -e  # 若有任何錯誤立即停止

# 1. 設定變數
MODEL_NAME=${1:-"qwen2.5:7b"}  # 預設使用 llama3，也可從參數傳入
VENV_DIR="venv"
PROMPT_DIR="./prompts"
RESULT_DIR="./results"
# 在這裡調整你要測的 MMLU 子集 (用逗號分隔)
SUBSETS="high_school_mathematics","high_school_world_history","high_school_physics","professional_law","business_ethics"

# 測試樣本數 (設為 0 代表跑全量，測試時建議設 5 或 10)
LIMIT_SAMPLES=10

echo "🚀 Starting Evaluation Pipeline using Model: $MODEL_NAME"
echo "ℹ️  Running in current Python environment."

# 2. 安裝/檢查依賴 (直接安裝到當前環境)
echo "⬇️  Checking dependencies..."
pip install -q -r requirements.txt

# 3. 建立必要的資料夾
mkdir -p $PROMPT_DIR
mkdir -p $RESULT_DIR

# 5. 執行 Python 主程式
echo "🔥 Running Evaluation..."
python src/main.py \
    --model "$MODEL_NAME" \
    --subsets "$SUBSETS" \
    --prompt_dir "$PROMPT_DIR" \
    --output_dir "$RESULT_DIR" \
    --limit $LIMIT_SAMPLES

echo "✅ All done! Check results in $RESULT_DIR"