# 檔案位置: experiments/data/mmlu/load_merged_data.py
import pandas as pd
import os
import random

# 請將此路徑改為您實際存放 MMLU csv 的資料夾
MMLU_DATA_PATH = 'data/mmlu/' 

# 您指定的子集列表
TARGET_TASKS = [
    "high_school_mathematics",
    "high_school_world_history",
    "high_school_physics",
    "professional_law",
    "business_ethics"
]

def load_merged_data(samples_per_task=300, train_ratio=0.8):
    """
    1. 讀取 TARGET_TASKS 中的每個科目
    2. 每個科目隨機取 samples_per_task (預設300) 筆
    3. 合併所有資料
    4. 依據 train_ratio (預設0.8, 即80%) 切分訓練與測試集
    """
    all_inputs = []
    all_outputs = []

    print(f"開始載入與合併 MMLU 資料 (每科取 {samples_per_task} 筆)...")

    for task in TARGET_TASKS:
        task_dfs = []
        # MMLU 通常有 _dev, _val, _test，我們全部讀進來當作母體池
        for split in ['dev', 'val', 'test']:
            csv_path = os.path.join(MMLU_DATA_PATH, f'{task}_{split}.csv')
            if os.path.exists(csv_path):
                # 讀取 CSV (MMLU 無 header)
                df = pd.read_csv(csv_path, header=None, names=['question', 'A', 'B', 'C', 'D', 'answer'])
                task_dfs.append(df)
        
        if not task_dfs:
            print(f"Warning: 找不到科目 {task} 的檔案，跳過。")
            continue
            
        # 合併該科目的所有檔案
        full_task_df = pd.concat(task_dfs, ignore_index=True)
        
        # 如果資料不足 300 筆，就全取；足夠就隨機取 300 筆
        n = min(len(full_task_df), samples_per_task)
        sampled_df = full_task_df.sample(n=n, random_state=42)
        
        print(f"  - {task}: 取得 {n} 筆")

        # 格式化為 APE 需要的 Input/Output
        for _, row in sampled_df.iterrows():
            input_text = (f"{row['question']}\n"
                          f"(A) {row['A']}\n"
                          f"(B) {row['B']}\n"
                          f"(C) {row['C']}\n"
                          f"(D) {row['D']}")
            all_inputs.append(input_text)
            all_outputs.append(row['answer'])
    
    # --- 合併與打亂 ---
    data_pairs = list(zip(all_inputs, all_outputs))
    random.shuffle(data_pairs) # 打亂順序
    all_inputs, all_outputs = zip(*data_pairs)
    
    total_count = len(all_inputs)
    split_idx = int(total_count * train_ratio)
    
    # --- 切分 80% Train, 20% Test ---
    # Induce Data (給 APE 找 Prompt 用)
    train_inputs = list(all_inputs[:split_idx])
    train_outputs = list(all_outputs[:split_idx])
    
    # Test Data (最後驗證用)
    test_inputs = list(all_inputs[split_idx:])
    test_outputs = list(all_outputs[split_idx:])
    
    print(f"合併完成。總筆數: {total_count}")
    print(f"  - 訓練集 (Induce): {len(train_inputs)} 筆 ({train_ratio*100}%)")
    print(f"  - 測試集 (Test): {len(test_inputs)} 筆 ({(1-train_ratio)*100}%)")
    
    return (train_inputs, train_outputs), (test_inputs, test_outputs)