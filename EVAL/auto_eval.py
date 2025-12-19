import json
import os
import glob
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset  # 需要 pip install datasets
from src.model.base_client import BaseModelClient # 假設這是你的 Client 路徑

# 引入你提供的 Scorer (為了版面簡潔，這裡假設你將上面的 Scorer 存成 scorer.py 並引入，或直接貼在下面)
# 這裡直接包含你提供的 Scorer Class 以確保完整性
# ==========================================
# [在此處貼上你原本提供的 Scorer Class 程式碼]
# 為了示範，我會在下方直接整合
# ==========================================

# 設定 Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoEval")

# --- 1. 擴充 Scorer (貼上你的原始代碼) ---
# (此處省略部分 import，假設已在上方引入)
# ... [User provided Scorer Code goes here, 保持不變] ...

# 為了讓程式碼可執行，我這裡簡化重寫 Scorer 依賴的部分，實際使用請保留你原本的 Scorer
# 請確保 _WORD_TO_NUM, Scorer 等類別定義在此處

# --- 2. MMLU 資料載入器 ---
class MMLUDataLoader:
    """
    負責載入 MMLU 資料集並將其轉換為 Scorer 可接受的格式:
    input: "Question text...\n(A) choice1\n(B) choice2..."
    target: "A" (or B, C, D)
    """
    def __init__(self, subsets=None, split='test'):
        self.subsets = subsets if subsets else ['abstract_algebra'] # 預設測試集
        self.split = split
        self.choices_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def format_mmlu_example(self, example):
        """將 HuggingFace MMLU 格式轉為文字 Prompt 格式"""
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        
        # 組合選項文字
        formatted_choices = []
        letters = ['A', 'B', 'C', 'D']
        for letter, choice_text in zip(letters, choices):
            formatted_choices.append(f"({letter}) {choice_text}")
        
        full_input = f"{question}\n" + "\n".join(formatted_choices)
        target = self.choices_map[answer_idx]
        
        return {
            'input': full_input,
            'target': target
        }

    def load_data(self):
        """載入並合併指定的 Subsets"""
        combined_data = []
        for subset in self.subsets:
            logger.info(f"Loading MMLU subset: {subset}...")
            try:
                # 使用 HuggingFace datasets 庫載入 (需聯網)
                # 如果是本地檔案，請修改此處為 pd.read_csv 或 json.load
                ds = load_dataset("cais/mmlu", subset, split=self.split)
                
                subset_data = [self.format_mmlu_example(ex) for ex in ds]
                logger.info(f"Subset {subset} loaded with {len(subset_data)} samples.")
                combined_data.extend(subset_data)
            except Exception as e:
                logger.error(f"Failed to load subset {subset}: {e}")
        
        return combined_data

# --- 3. 自動化評測系統核心 ---
class PromptEvaluationSystem:
    def __init__(self, model_client, mmlu_subsets, output_dir="./eval_results"):
        self.client = model_client
        self.data_loader = MMLUDataLoader(subsets=mmlu_subsets)
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 預先載入資料 (避免每個 Prompt 重複載入)
        self.eval_data = self.data_loader.load_data()
        logger.info(f"Total evaluation samples: {len(self.eval_data)}")

    class Config:
        """模擬 Scorer 需要的 config"""
        def __init__(self, task_name='mmlu', instruction_pos='Q_begin'):
            self.task_name = task_name
            self.dataset_name = 'mmlu'
            self.instruction_pos = instruction_pos

    def evaluate_prompt_file(self, json_filepath):
        """讀取單個 JSON 檔案，評測其中的所有 Prompts"""
        file_name = os.path.basename(json_filepath)
        logger.info(f"Processing file: {file_name}")

        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_filepath}: {e}")
            return

        # 假設 JSON 結構是: 
        # { "experiment_name": "test1", "prompts": [ {"id": "p1", "text": "..."}, ... ] }
        # 或是簡單的: { "prompts": ["prompt1...", "prompt2..."] }
        
        prompts_list = data.get("prompts", [])
        results = []

        scorer = Scorer(self.client, config=self.Config())

        for idx, p_item in enumerate(prompts_list):
            # 處理不同格式的 Prompt 輸入
            if isinstance(p_item, str):
                prompt_text = p_item
                prompt_id = f"prompt_{idx}"
            else:
                prompt_text = p_item.get("text", "")
                prompt_id = p_item.get("id", f"prompt_{idx}")

            if not prompt_text: 
                continue

            logger.info(f"Evaluating Prompt ID: {prompt_id}")
            
            # *** 核心: 呼叫 Scorer ***
            # num_samples=None 表示跑全量，測試時可設為 5 或 10
            score_res = scorer.score_instruction(
                instruction=prompt_text, 
                dataset=self.eval_data, 
                num_samples=10 # <--- 測試用，正式跑請拿掉或設為 None
            )
            
            # 整理結果
            result_entry = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "score": score_res['score'],
                "num_evals": score_res['num_evals'],
                # "details": score_res['detailed_dataframe'].to_dict() # 若需要詳細 log 可打開
            }
            results.append(result_entry)
            logger.info(f"Prompt {prompt_id} Score: {score_res['score']:.2%}")

        # 輸出結果檔案
        output_path = os.path.join(self.output_dir, f"result_{file_name}")
        output_data = {
            "source_file": file_name,
            "mmlu_subsets_used": self.data_loader.subsets,
            "results": results
        }
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved results to {output_path}")

    def run_all(self, prompt_dir):
        """掃描目錄下所有 JSON 並執行"""
        json_files = glob.glob(os.path.join(prompt_dir, "*.json"))
        logger.info(f"Found {len(json_files)} prompt files to evaluate.")
        
        for json_file in json_files:
            self.evaluate_prompt_file(json_file)

# --- 4. 執行進入點 ---
if __name__ == "__main__":
    # A. 初始化你的 Model Client (這裡需要你原本的實例)
    # my_client = BaseModelClient(...) 
    # 這裡用一個 Mock Client 演示，請替換成你的真實 Client
    class MockClient:
        def generate_text(self, prompt):
            # 隨機回傳一個答案，模擬模型行為
            import random
            return f"Answer is ({random.choice(['A', 'B', 'C', 'D'])})"
    
    my_client = MockClient()

    # B. 設定要測試的 MMLU 子集
    # 建議先用簡單的 'elementary_mathematics' 或 'global_facts' 測試
    target_subsets = ['elementary_mathematics', 'global_facts']

    # C. 設定 Prompt 檔案位置與輸出位置
    PROMPT_DIR = "./prompts_to_test"  # 放你的 Prompt JSON 的資料夾
    OUTPUT_DIR = "./eval_results"     # 輸出結果的資料夾

    # 確保輸入資料夾存在並建立測試檔案 (第一次執行用)
    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)
        sample_prompt_file = {
            "prompts": [
                {"id": "p1_chain_of_thought", "text": "Think step by step properly."},
                {"id": "p2_direct", "text": "Answer directly."}
            ]
        }
        with open(os.path.join(PROMPT_DIR, "test_prompts.json"), 'w') as f:
            json.dump(sample_prompt_file, f)

    # D. 啟動系統
    system = PromptEvaluationSystem(my_client, target_subsets, OUTPUT_DIR)
    system.run_all(PROMPT_DIR)