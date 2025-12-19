# ape/utility.py
import re
import string
import numpy as np

# --- 常數定義 ---
_WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90,
}

FINAL_ANSWER_BEHIND_PATTERNS = ['answer is', 'answer:', 'answer is:', 'is:', 'are:']
FINAL_ANSWER_AHEAD_PATTERNS = ['is the correct answer', 'is the right answer', 'is the final answer', 'is the answer']
GSM8K_ANSWER_DELIMITER = '####'

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_bracketed_choice(prediction: str):
    """
    Robust extraction for MCQ option letter.
    Supports:
      - (a), [B], {c}
      - Answer: a / answer is (d) / option: C / choice - b
      - "the correct answer is C"
      - trailing " ... c)" / " ... c." / " ... c"
    Returns a single lowercase letter in [a-d] when found; otherwise returns original cleaned text.
    """
    text = str(prediction).lower().strip()

    # 1) Explicit "answer/option/choice" patterns (most reliable)
    explicit_patterns = [
        r'(?:final\s+)?answer\s*(?:is|:)?\s*[\(\[\{]?\s*([a-d])\s*[\)\]\}]?',
        r'(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)?\s*[\(\[\{]?\s*([a-d])\s*[\)\]\}]?',
        r'option\s*(?:is|:)?\s*[\(\[\{]?\s*([a-d])\s*[\)\]\}]?',
        r'choice\s*(?:is|:)?\s*[\(\[\{]?\s*([a-d])\s*[\)\]\}]?',
        r'i\s*(?:choose|pick|select)\s*(?:option\s*)?[\(\[\{]?\s*([a-d])\s*[\)\]\}]?',
    ]
    for pat in explicit_patterns:
        ms = re.findall(pat, text, flags=re.IGNORECASE)
        if ms:
            return ms[-1].lower()

    # 2) Bracketed forms anywhere: (c), [d], {a}
    ms = re.findall(r'[\(\[\{]\s*([a-d])\s*[\)\]\}]', text, flags=re.IGNORECASE)
    if ms:
        return ms[-1].lower()

    # 3) Trailing standalone letter with common punctuation: " ... c)" / " ... c." / " ... c"
    ms = re.findall(r'(?:^|[\s])([a-d])(?:\s*[\)\]\}\.\:]|\s*$)', text, flags=re.IGNORECASE)
    if ms:
        return ms[-1].lower()

    return text

def parse_number(prediction: str):
    """Parse numeric answer (GSM8K). Minimal but safer than plain word replacement."""
    s = str(prediction)

    # Remove common noise
    s = s.replace('$', '').replace(',', '').replace('%', '')

    low = s.lower()

    # Minimal fix: merge "tens ones" => e.g., "twenty one" -> 21
    # We do this before general word->num substitution to avoid "20 1" then taking "1".
    tens_words = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    ones_words = ["one","two","three","four","five","six","seven","eight","nine"]
    tens_re = r'(' + '|'.join(tens_words) + r')'
    ones_re = r'(' + '|'.join(ones_words) + r')'

    def _merge_tens_ones(m):
        tens = _WORD_TO_NUM[m.group(1).lower()]
        ones = _WORD_TO_NUM[m.group(2).lower()]
        return str(tens + ones)

    low = re.sub(rf'\b{tens_re}\s+{ones_re}\b', _merge_tens_ones, low)

    # Now do simple word->num substitution (keeps your original behavior)
    for word, num in _WORD_TO_NUM.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        low = re.sub(pattern, str(num), low, flags=re.IGNORECASE)

    # Extract the last number (keeps your original heuristic)
    matches = re.findall(r'-?\d+\.?\d*', low)
    if matches:
        return matches[-1]
    return low.strip()


def normalize_prediction(prediction, task_type='general'):
    """
    Stronger normalization
    task_type: 'gsm8k', 'boolean', 'mmlu' (or 'general')
    """
    prediction = str(prediction).lower().strip()

    # 1) Handle 'The answer is...' prefixes
    for pattern in FINAL_ANSWER_BEHIND_PATTERNS:
        if pattern in prediction:
            prediction = prediction.split(pattern)[-1]
            break

    # 2) Handle GSM8K delimiter
    if GSM8K_ANSWER_DELIMITER in prediction:
        prediction = prediction.split(GSM8K_ANSWER_DELIMITER)[-1]

    # 3) Handle suffix patterns
    for pattern in FINAL_ANSWER_AHEAD_PATTERNS:
        if pattern in prediction:
            prediction = prediction.split(pattern)[0]

    prediction = prediction.strip().strip('.')

    if task_type == 'gsm8k':
        return parse_number(prediction)

    if task_type == 'boolean':
        bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false',
                    'true': 'true', 'false': 'false'}
        clean_pred = prediction.translate(str.maketrans('', '', string.punctuation)).strip()
        clean_pred = re.sub(r'\s+', ' ', clean_pred).strip()
        return bool_map.get(clean_pred, clean_pred)

    # MMLU / General default
    return extract_bracketed_choice(prediction)

def normalize_target(target, task_type='general'):
    target = str(target).lower().strip()
    if GSM8K_ANSWER_DELIMITER in target:
        target = target.split(GSM8K_ANSWER_DELIMITER)[-1]
    target = target.strip().strip('.')

    if task_type == 'gsm8k':
        target = target.replace(',', '')
        return target

    if task_type in ['mmlu', 'general']:
        return extract_bracketed_choice(target)

    if task_type == 'boolean':
        clean_t = target.translate(str.maketrans('', '', string.punctuation)).strip()
        clean_t = re.sub(r'\s+', ' ', clean_t).strip()
        bool_map = {'yes': 'true', 'no': 'false', 'valid': 'true', 'invalid': 'false',
                    'true': 'true', 'false': 'false'}
        return bool_map.get(clean_t, clean_t)

    return target

def get_em_score(prediction, ground_truth, task_type='general'):
    """計算精確匹配分數"""
    pred_norm = normalize_prediction(prediction, task_type)
    target_norm = normalize_target(ground_truth, task_type)
    
    # 數值比對
    if task_type == 'gsm8k':
        try:
            if abs(float(pred_norm) - float(target_norm)) < 1e-6:
                return 1.0
        except:
            pass
            
    # 字串比對
    return 1.0 if pred_norm == target_norm else 0.0

def get_multi_answer_em(prediction, answers, task_type='general'):
    """支援多個標準答案的檢查"""
    # 這裡假設 answers 是一個 list，如果不適，包成 list
    if not isinstance(answers, list):
        answers = [answers]
    
    for answer in answers:
        # 這裡會呼叫上面更新過的 get_em_score，並傳入 task_type
        if get_em_score(prediction, answer, task_type) == 1.0:
            return 1.0
    return 0.0