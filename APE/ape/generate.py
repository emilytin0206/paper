
import random
from typing import List, Tuple
from .llm import LLM

def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    組合生成 Prompt 的 Query
    """
    inputs, outputs = subsampled_data
    # 填充 Few-shot 範例
    demos = demos_template.fill(subsampled_data)
    # 填充生成指令 (輸入/輸出/範例)
    # 這裡的 input/output 是指用來讓模型 "逆向工程" 的那一對資料
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)

def generate_prompts(
    model: LLM,
    data: Tuple[List[str], List[List[str]]],
    config: dict
) -> List[str]:
    """
    Generates candidate prompts using the Official APE strategy:
    num_subsamples (Outer Loop) x num_prompts_per_subsample (Inner Batch)
    """
    prompt_gen_template = config.get('prompt_gen_template')
    demos_template = config.get('demos_template')
    
    # Official APE parameters
    num_demos = config.get('num_demos', 3)
    num_subsamples = config.get('num_subsamples', 3)          # 官方預設 3 次迭代
    num_prompts_per_subsample = config.get('num_prompts_per_subsample', 10) # 每次生成 10 個
    
    inputs, outputs = data
    queries = []

    print(f"Generating Prompts: {num_subsamples} subsamples x {num_prompts_per_subsample} prompts...")

    # === Outer Loop: 3 Iterations (Subsamples) ===
    # 每一輪迴圈代表使用一組特定的 Few-shot 範例
    for i in range(num_subsamples):
        # 1. Sample Demonstrations (Context)
        indices = random.sample(range(len(inputs)), min(num_demos, len(inputs)))
        
        demo_strs = []
        for idx in indices:
            d = demos_template.replace('[INPUT]', inputs[idx])\
                              .replace('[OUTPUT]', outputs[idx][0])
            demo_strs.append(d)
        
        full_demo = "\n\n".join(demo_strs)
        
        # 2. Construct Query
        base_query = prompt_gen_template.replace('[full_DEMO]', full_demo)
        
        # 3. Expand for Inner Batch
        # 因為 Ollama 是一個 Prompt 產生一個回應，我們將同一個 Query 複製多次
        # 讓 Temperature > 0 發揮作用，產生多樣化的指令
        queries.extend([base_query] * num_prompts_per_subsample)

    # === Batch Inference ===
    # 一次性發送所有請求 (3 * 10 = 30 queries)
    raw_candidates = model.generate(queries)
    
    # Deduplicate and clean
    unique_candidates = list(set([c.strip() for c in raw_candidates if c.strip()]))
    print(f"Generated {len(unique_candidates)} unique candidates from {len(queries)} raw outputs.")
    
    return unique_candidates