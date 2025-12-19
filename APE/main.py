# main.py
import os
import json
import datetime
import random
from ape.llm import Ollama_Forward
from ape.generate import generate_prompts
from ape.evaluator import exec_accuracy_evaluator
from data.mmlu import load_merged_mmlu_data

# === [修改] 增加 usage_info 參數 ===
def save_experiment_results(config, task_name, results, usage_info=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_model_safe = config['target']['model'].replace(':', '-')
    optimizer_model_safe = config['optimizer']['model'].replace(':', '-')
    
    dir_name = f"{target_model_safe}_{optimizer_model_safe}_{task_name}_{timestamp}"
    base_dir = os.path.join("experiments", "results")
    save_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Saving] 儲存實驗結果至: {save_dir}")

    # 1. 儲存 Config
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # 2. 儲存所有結果
    output_data = []
    for rank, (prompt, score) in enumerate(results):
        output_data.append({"rank": rank + 1, "score": score, "prompt": prompt})

    with open(os.path.join(save_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    # 3. 儲存最佳 Prompt 與 Token 統計
    if output_data:
        best_result = output_data[0]
        with open(os.path.join(save_dir, "best_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(f"Experiment: {dir_name}\nTask: {task_name}\nBest Score: {best_result['score']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best Prompt:\n{best_result['prompt']}\n")
            f.write("-" * 50 + "\n")
            
            # === [新增] 寫入 Token 統計到 txt ===
            if usage_info:
                f.write("\nToken Usage Statistics:\n")
                f.write(json.dumps(usage_info['grand_total'], indent=4))
                f.write("\n" + "-" * 50 + "\n")
            # ===================================

    # === [新增] 另外存一個 JSON 檔方便分析 ===
    if usage_info:
        with open(os.path.join(save_dir, "usage_stats.json"), "w", encoding="utf-8") as f:
            json.dump(usage_info, f, indent=4)

    print("[Saving] 儲存完成！")

def main():
    # --- Configuration ---
    OLLAMA_URL = "http://140.113.86.14:11434"
    
    ALL_SUBSETS = [
        "high_school_mathematics",
        "high_school_world_history",
        "high_school_physics",
        "professional_law",
        "business_ethics"
    ]
    
    TASK_LABEL = f"merged_{len(ALL_SUBSETS)}_subsets"

    conf = {
        'optimizer': {
            'model': 'qwen2.5:32b',
            'api_url': OLLAMA_URL,
            'temperature': 0.7 
        },
        'target': {
            'model': 'qwen2.5:7b',
            'api_url': OLLAMA_URL,
            'temperature': 0.0
        },
        'generation': {
            'num_subsamples': 5,            
            'num_prompts_per_subsample': 50, 
            'num_demos': 5,
            'prompt_gen_template': "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [INSTRUCTION]; output ONLY the instruction itself, with no explanation, reasoning, or additional text.",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        },
        'evaluation': {
            'task_name': 'mmlu',
            'num_samples': 50,
            'num_few_shot': 0,
            'eval_template': "Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]",
            'demos_template': "Input: [INPUT]\nOutput: [OUTPUT]"
        }
    }
    
    # --- Step 1: Data Loading & Pooling ---
    print(f"\n[Data] Loading and merging {len(ALL_SUBSETS)} subsets...")
    
    # === [修正] 將讀取量調大，確保有足夠樣本 ===
    # limit_per_subset=20 (5個子集共100筆)
    raw_inputs, raw_outputs = load_merged_mmlu_data(ALL_SUBSETS, split='test', limit_per_subset=20)
    
    if not raw_inputs:
        print("[Error] No data loaded.")
        return

    # --- Step 2: Shuffling & Strict Splitting ---
    print("[Data] Shuffling and splitting data...")
    
    paired_data = list(zip(raw_inputs, raw_outputs))
    random.seed(42) 
    random.shuffle(paired_data)
    
    shuffled_inputs, shuffled_outputs = zip(*paired_data)
    shuffled_inputs = list(shuffled_inputs)
    shuffled_outputs = list(shuffled_outputs)

    # === [修正] 增加 TRAIN_SIZE，確保 num_demos (5) 有足夠樣本可抽 ===
    TRAIN_SIZE = 20
    if len(shuffled_inputs) < TRAIN_SIZE + 10:
        TRAIN_SIZE = int(len(shuffled_inputs) * 0.4) # 若資料不足，取 40% 當訓練集

    train_data = (shuffled_inputs[:TRAIN_SIZE], shuffled_outputs[:TRAIN_SIZE])
    eval_data = (shuffled_inputs[TRAIN_SIZE:], shuffled_outputs[TRAIN_SIZE:])

    print(f"  - Total Data Pool: {len(shuffled_inputs)}")
    print(f"  - Train Set (Generation): {len(train_data[0])} samples (Indices 0-{TRAIN_SIZE})")
    print(f"  - Test Set (Evaluation): {len(eval_data[0])} samples (Indices {TRAIN_SIZE}-end)")
    
    # --- Step 3: Generate Prompts ---
    print("\n[APE] Step 1: Generating Prompts...")
    # 初始化模型 (內部計數器歸零)
    optimizer_model = Ollama_Forward(conf['optimizer'])
    
    candidates = generate_prompts(optimizer_model, train_data, conf['generation'])
    
    print(f"Generated {len(candidates)} candidates.")
    if not candidates:
        return

    # --- Step 4: Evaluate Prompts ---
    print("\n[APE] Step 2: Evaluating Prompts...")
    # 初始化目標模型 (內部計數器歸零)
    target_model = Ollama_Forward(conf['target'])
    
    scored_results = exec_accuracy_evaluator(
        model=target_model, 
        prompts=candidates, 
        eval_data=eval_data, 
        few_shot_data=train_data,
        config=conf['evaluation']
    )

    # --- Step 5: Save & Stats ---
    print("\n=== Final Leaderboard ===")
    for rank, (prompt, score) in enumerate(scored_results[:5]):
        print(f"Rank {rank+1} | Score: {score:.2f} | Prompt: {prompt}")

    # === [新增] 收集並整合 Token 統計資訊 ===
    opt_usage = optimizer_model.get_usage()
    tgt_usage = target_model.get_usage()
    
    total_usage = {
        "optimizer_model": opt_usage,
        "target_model": tgt_usage,
        "grand_total": {
            "prompt_tokens": opt_usage['prompt_tokens'] + tgt_usage['prompt_tokens'],
            "completion_tokens": opt_usage['completion_tokens'] + tgt_usage['completion_tokens'],
            "total_tokens": opt_usage['total_tokens'] + tgt_usage['total_tokens'],
            "total_calls": opt_usage['total_calls'] + tgt_usage['total_calls']
        }
    }
    
    print("\n[Usage Stats]")
    print(json.dumps(total_usage['grand_total'], indent=2))
    
    # 傳入 usage_info 進行存檔
    save_experiment_results(conf, TASK_LABEL, scored_results, usage_info=total_usage)

if __name__ == "__main__":
    main()