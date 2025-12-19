import os
import sys
import json
import glob
import logging
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.client import OllamaClient
from src.loader import MMLUDataLoader
from src.scorer import Scorer

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def main():
    parser = argparse.ArgumentParser(description="Automated Prompt Evaluation System")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")
    parser.add_argument("--subsets", type=str, default="global_facts", help="Comma separated MMLU subsets")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--prompt_dir", type=str, default="./prompts", help="Directory containing prompt JSONs")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--limit", type=int, default=5, help="Num samples per prompt (use 0 for all)")
    
    args = parser.parse_args()
    
    # 1. 準備元件
    client = OllamaClient(model_name=args.model)
    subsets_list = [s.strip() for s in args.subsets.split(',')]
    loader = MMLUDataLoader(subsets=subsets_list, split=args.split)
    scorer = Scorer(client, config_mode='Q_begin')

    # 2. 載入資料
    logger.info("Loading Dataset...")
    dataset = loader.load_data()
    
    # 3. 確保目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 掃描檔案並執行
    json_files = glob.glob(os.path.join(args.prompt_dir, "*.json"))
    
    # --- 修改這裡：更明確的 Limit 判斷與 Log 提示 ---
    if args.limit > 0:
        num_samples = args.limit
        logger.info(f"🔧 Config: Sampling first {num_samples} items per prompt.")
    else:
        num_samples = None
        logger.info("🔧 Config: Limit set to 0. Running on FULL dataset (All samples).")

    for json_file in json_files:
        full_file_name = os.path.basename(json_file)
        base_name = os.path.splitext(full_file_name)[0]
        
        logger.info(f"Processing: {full_file_name}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = []
        prompts = data.get("prompts", [])
        
        for idx, item in enumerate(prompts):
            # 兼容處理：無論輸入是字串還是物件，都取出 Prompt 文字
            p_text = item if isinstance(item, str) else item.get("text", "")
            
            # 雖然輸出不存 ID，但 Log 還是印一下方便你看進度
            p_id_log = f"p_{idx}" if isinstance(item, str) else item.get("id", f"p_{idx}")
            
            if not p_text: continue
            
            logger.info(f"Testing: {p_id_log}")
            res = scorer.score_instruction(p_text, dataset, num_samples=num_samples)
            
            # ==========================================
            # 修改重點：只儲存 score 和 prompt
            # ==========================================
            results.append({
                "score": res['score'],
                "prompt": p_text,
                "count": res['num_evals']  # <--- 加入這行，方便您確認是否真的跑了 300 題
            })
            
            logger.info(f"Score: {res['score']:.2%}")

        # 輸出結果檔案
        out_filename = f"{base_name}_result.json"
        out_path = os.path.join(args.output_dir, out_filename)
        
        # 這裡我保留了外層的 metadata (source_file 等)，讓檔案結構是合法的 JSON
        # 如果你連外層都不要，只想存 results list，可以改為 json.dump(results, f, ...)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "source_file": full_file_name,
                "model": args.model,
                "subsets": subsets_list,
                "results": results  # 這裡面現在只有 score 和 prompt
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved results to: {out_filename}")

if __name__ == "__main__":
    main()