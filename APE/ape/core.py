# ape/core.py
import random
from typing import List, Tuple, Dict, Any
from .llm import create_model
from . import generate, evaluate

def find_prompts(
    train_data: Tuple[List[str], List[List[str]]],
    eval_data: Tuple[List[str], List[List[str]]],
    config: Dict[str, Any]
) -> List[Tuple[str, float]]:
    """
    Main APE pipeline:
    1. Generate candidate prompts using the Optimizer Model.
    2. Evaluate candidates using the Target Model.
    3. Return sorted prompts.
    """
    
    # 1. 初始化模型
    optimizer_model = create_model(config['optimizer'])
    target_model = create_model(config['target'])

    print("\n=== Step 1: Generating Candidate Prompts ===")
    # 準備 Few-shot 範例
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    prompt_gen_template = (
        "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n"
        "[full_DEMO]\n\n"
        "The instruction was to [APE]"
    )
    
    # 執行生成
    candidates = generate.generate_prompts(
        model=optimizer_model,
        data=train_data,
        prompt_gen_template=prompt_gen_template,
        demos_template=demos_template,
        num_demos=config['generation']['num_demos'],
        num_candidates=config['generation']['num_candidates']
    )
    
    print(f"Generated {len(candidates)} unique candidates.")
    if not candidates:
        return []

    print("\n=== Step 2: Evaluating Candidates ===")
    eval_template = "Instruction: [PROMPT]\n\n[INPUT]\nAnswer: [OUTPUT]"
    
    # 執行評估
    scored_prompts = evaluate.evaluate_prompts(
        model=target_model,
        prompts=candidates,
        eval_data=eval_data,
        eval_template=eval_template,
        num_samples=config['evaluation']['num_samples']
    )

    # 排序 (分數高到低)
    scored_prompts.sort(key=lambda x: x[1], reverse=True)
    
    return scored_prompts