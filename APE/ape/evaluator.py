# ape/evaluator.py
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from . import utility 

def exec_accuracy_evaluator(
    model, 
    prompts: List[str],
    eval_data: Tuple[List[str], List[List[str]]],
    few_shot_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    使用並發 (Threading) 加速的評估器
    """
    
    # --- 1. 設定參數 ---
    num_samples = config.get('num_samples', 20)
    num_few_shot = config.get('num_few_shot', 0)
    
    # 根據 dataset 名稱決定 task_type (用於 utility 解析)
    # 若 config 沒寫，預設 'general'
    task_name = config.get('task_name', 'general').lower()
    if 'gsm8k' in task_name:
        task_type = 'gsm8k'
    elif any(x in task_name for x in ['boolean', 'causal']):
        task_type = 'boolean'
    else:
        task_type = 'general'

    eval_template = config.get('eval_template', "Instruction: [PROMPT]\n\n[INPUT]\n[OUTPUT]")
    demos_template = config.get('demos_template', "Input: [INPUT]\nOutput: [OUTPUT]")
    
    inputs, outputs = eval_data
    indices = list(range(len(inputs)))
    sample_indices = indices[:min(num_samples, len(indices))]
    
    print(f"Evaluating {len(prompts)} prompts on {len(sample_indices)} samples (Task Type: {task_type})...")

    # --- 2. 內部函數：單一請求處理 ---
    def process_single_request(prompt_str, input_str, ground_truth_list):
        # 準備 Few-shot (如果需要)
        full_demo = ""
        if num_few_shot > 0 and few_shot_data:
            # 隨機採樣 few-shot (注意：這裡為了效率可能要考慮是否每次隨機，或是固定)
            # 為了簡單起見，這裡每次請求都隨機採樣
            fs_indices = random.sample(range(len(few_shot_data[0])), min(num_few_shot, len(few_shot_data[0])))
            demo_strs = []
            for i in fs_indices:
                d = demos_template.replace('[INPUT]', few_shot_data[0][i])\
                                  .replace('[OUTPUT]', few_shot_data[1][i][0])
                demo_strs.append(d)
            full_demo = "\n\n".join(demo_strs)

        # 組合 Query
        query = eval_template.replace('[PROMPT]', prompt_str)\
                             .replace('[INPUT]', input_str)\
                             .replace('[OUTPUT]', "")
        
        if full_demo:
            query = f"{full_demo}\n\n{query}"
            
        # 呼叫模型 (注意：這裡假設 model.generate 支援單字串輸入)
        # 我們將 n=1, 並且只取第一個結果
        try:
            preds = model.generate(query, n=1)
            pred = preds[0] if preds else ""
            
            # 計算分數
            score = utility.get_multi_answer_em(pred, ground_truth_list, task_type=task_type)
            return score
        except Exception as e:
            print(f"Error in eval: {e}")
            return 0.0

    # --- 3. 主迴圈：並發執行 ---
    prompt_scores = []
    
    # 我們需要評估 每個 Prompt x 每個 Sample
    # 為了進度條好看，我們對每個 Prompt 進行一次並發批次
    
    # 設定並發數量 (根據您的顯存/API限制調整，Ollama 本地通常 4-8)
    MAX_WORKERS = 8 

    for prompt in prompts:
        scores = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {}
            for idx in sample_indices:
                inp = inputs[idx]
                out = outputs[idx]
                future = executor.submit(process_single_request, prompt, inp, out)
                future_to_idx[future] = idx
            
            # 等待當前 Prompt 的所有測試完成
            # 使用 tqdm 顯示進度 (如果 sample 數多才有感)
            iterator = as_completed(future_to_idx)
            if len(sample_indices) > 10:
                iterator = tqdm(iterator, total=len(sample_indices), desc="Scoring", leave=False)
                
            for future in iterator:
                scores.append(future.result())

        avg_score = np.mean(scores) if scores else 0.0
        prompt_scores.append((prompt, avg_score))
        
        # 簡單印出當前 Prompt 的表現
        print(f"  > Prompt: {prompt[:50]}... | Score: {avg_score:.2f}")

    # 排序
    prompt_scores.sort(key=lambda x: x[1], reverse=True)
    
    return prompt_scores