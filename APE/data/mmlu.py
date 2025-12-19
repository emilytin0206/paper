# data/mmlu.py
import os
import pandas as pd

def load_mmlu_data(subset: str, split: str = 'test', limit: int = None):
    """
    保留此函數以兼容舊介面，轉發給 load_merged_mmlu_data。
    """
    return load_merged_mmlu_data([subset], split, limit)

def load_merged_mmlu_data(subsets: list, split: str = 'test', limit_per_subset: int = None):
    """
    讀取本地 data/mmlu 資料夾中的 CSV 檔案並合併。
    格式假設: header=None, [Question, A, B, C, D, Answer]
    """
    all_inputs = []
    all_outputs = []
    
    # 設定本地資料路徑
    base_path = os.path.join("data", "mmlu")
    
    print(f"Loading local MMLU files from: {base_path}")

    for task in subsets:
        # 組合檔名，例如: high_school_mathematics_test.csv
        file_name = f"{task}_{split}.csv"
        file_path = os.path.join(base_path, file_name)

        if not os.path.exists(file_path):
            print(f"[Warning] 找不到檔案: {file_path}，跳過此任務。")
            continue

        print(f"  - Reading: {file_name}")
        
        try:
            # 讀取 CSV (MMLU 通常沒有 header)
            df = pd.read_csv(file_path, header=None)
            
            # 如果有限制數量
            if limit_per_subset:
                df = df.head(limit_per_subset)

            for _, row in df.iterrows():
                # MMLU CSV 結構通常為:
                # Column 0: Question
                # Column 1-4: Options A, B, C, D
                # Column 5: Answer Key (e.g., 'A', 'B'...)
                
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer_key = str(row[5]).strip()

                # 格式化 Prompt
                q_text = f"Question: {question}\nOptions:\n"
                labels = ['A', 'B', 'C', 'D']
                for label, opt in zip(labels, options):
                    q_text += f"{label}. {opt}\n"
                q_text += "Answer:"

                all_inputs.append(q_text)
                all_outputs.append([answer_key]) # APE 需要 list 格式

        except Exception as e:
            print(f"[Error] 讀取 {file_name} 失敗: {e}")

    print(f"Total samples loaded: {len(all_inputs)}")
    return all_inputs, all_outputs