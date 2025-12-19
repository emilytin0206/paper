import logging
from datasets import load_dataset

logger = logging.getLogger("MMLULoader")

class MMLUDataLoader:
    def __init__(self, subsets=None, split='validation'):
        # 若未指定子集，預設使用 global_facts 作為快速測試
        self.subsets = subsets if subsets else ['global_facts']
        self.split = split
        self.choices_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def format_mmlu_example(self, example):
        """將 MMLU 格式轉換為 Q_begin 需要的純文字輸入"""
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        
        formatted_choices = [f"({l}) {c}" for l, c in zip(['A','B','C','D'], choices)]
        full_input = f"{question}\n" + "\n".join(formatted_choices)
        
        return {
            'input': full_input,
            'target': self.choices_map[answer_idx]
        }

    def load_data(self):
        combined_data = []
        for subset in self.subsets:
            logger.info(f"Loading subset: {subset}...")
            try:
                ds = load_dataset("cais/mmlu", subset, split=self.split)
                subset_data = [self.format_mmlu_example(ex) for ex in ds]
                combined_data.extend(subset_data)
            except Exception as e:
                logger.error(f"Failed to load {subset}: {e}")
        return combined_data