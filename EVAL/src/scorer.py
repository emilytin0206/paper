import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger("Scorer")

class Scorer:
    def __init__(self, client, config_mode='Q_begin'):
        self.client = client
        self.instruction_pos = config_mode  # 強制設定 Prompt 結構

    def _format_prompt(self, instruction: str, question: str) -> str:
        # 依據你的要求，這裡實作 Q_begin 邏輯
        if self.instruction_pos == 'Q_begin':
            return f"{instruction}\n\nQ: {question}\nA:"
        return f"{instruction}\n{question}"

    def _extract_choice_letter(self, text: str) -> str | None:
        t = text.lower()
        if m := re.search(r'\\boxed\{\s*([a-e])\s*\}', t): return m.group(1)
        if m := re.search(r'\(([a-e])\)', t): return m.group(1)
        if m := re.search(r'\b([a-e])\b', t): return m.group(1)
        return None

    def _get_normalized_prediction(self, prediction: str) -> str:
        """針對選擇題的標準化清理"""
        pred = prediction.lower().strip()
        # 移除常見前綴
        for pat in ['answer is', 'answer:', 'the answer is']:
            if pat in pred: pred = pred.split(pat)[-1].strip()
        
        # 嘗試提取選項 (A/B/C/D)
        choice = self._extract_choice_letter(pred)
        if choice: return choice
        
        # 若無選項，移除句號回傳
        return pred[:-1] if pred.endswith('.') else pred

    def _check_answer(self, prediction: str, target: str) -> float:
        pred_norm = self._get_normalized_prediction(str(prediction))
        target_norm = str(target).lower().strip()
        return 1.0 if pred_norm == target_norm else 0.0

    def score_instruction(self, instruction: str, dataset: list, num_samples: int = None) -> dict:
        import random
        eval_data = dataset
        if num_samples and num_samples < len(dataset):
            # random.seed(42) # 可固定隨機
            eval_data = random.sample(dataset, num_samples)

        scores = []
        # 設定並發數，Ollama 本地端建議設為 1，避免記憶體爆掉
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for ex in eval_data:
                prompt = self._format_prompt(instruction, ex['input'])
                futures.append(executor.submit(self._run_single, prompt, ex['target']))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Eval", leave=False):
                res = future.result()
                if res is not None:  # <--- 只要不是 None (代表程式出錯)，就算是 0.0 也要加入！
                    scores.append(res)

        return {
            'score': float(np.mean(scores)) if scores else 0.0,
            'num_evals': len(scores)
        }

    def _run_single(self, prompt, target):
        try:
            pred = self.client.generate_text(prompt)
            return self._check_answer(pred, target)
        except Exception as e:
            logger.error(f"Error: {e}")
            return None