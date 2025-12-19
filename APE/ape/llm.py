import ollama
import datetime
import os
import threading
from abc import ABC, abstractmethod

# 嘗試引用進度條，如果沒有安裝則使用 dummy 函數
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class LLM(ABC):
    @abstractmethod
    def generate(self, prompt, n):
        pass

class Ollama_Forward(LLM):
    """
    Wrapper for Ollama (Local LLM) with Logging, Progress Bar, and Token Counting.
    """
    def __init__(self, config):
        self.config = config
        # 初始化 Client
        host = self.config.get('api_url')
        self.client = ollama.Client(host=host) if host else ollama
        
        # 設定 Log 檔案名稱
        self.log_file = "ollama_history.log"
        
        # === [新增] Token 計數器與鎖 ===
        self._usage_lock = threading.Lock()
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0
        }
        # ===============================

        # 啟動時先在 Log 寫一行分隔線，標記新的一次執行
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} New Session Started at {datetime.datetime.now()} {'='*20}\n")
        
    def generate(self, prompt, n=1):
        """
        prompt: 可以是單一字串或字串 list
        n: 每個 prompt 生成幾個回答
        """
        if not isinstance(prompt, list):
            prompt = [prompt]
            
        model_name = self.config.get('model', 'llama3')
        
        options = {
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.9),
        }

        results = []
        total_tasks = len(prompt) * n
        
        # 為了避免 log 太亂，只有當 prompt 數量大於 1 時才印進度條 (Evaluator 內部並發呼叫時通常是一次一個)
        iterator = prompt
        if len(prompt) > 1:
            print(f"[Ollama] Generating {total_tasks} completions using {model_name}...")
            print(f"[Log] 詳細輸出將記錄於: {os.path.abspath(self.log_file)}")
            iterator = tqdm(prompt, desc="Gen Progress")
        
        for p in iterator:
            for _ in range(n):
                try:
                    # 發送請求
                    response = self.client.generate(
                        model=model_name, 
                        prompt=p, 
                        options=options
                    )
                    res_content = response['response']
                    results.append(res_content)
                    
                    # === [新增] 抓取 Token 使用量並更新 ===
                    # Ollama API 回傳欄位: prompt_eval_count (Input), eval_count (Output)
                    p_tokens = response.get('prompt_eval_count', 0)
                    c_tokens = response.get('eval_count', 0)
                    
                    with self._usage_lock:
                        self.usage_stats['prompt_tokens'] += p_tokens
                        self.usage_stats['completion_tokens'] += c_tokens
                        self.usage_stats['total_tokens'] += (p_tokens + c_tokens)
                        self.usage_stats['total_calls'] += 1
                    # ======================================

                    # === 寫入 Log ===
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        f.write(f"[{timestamp}] PROMPT (len={len(p)}) [Tokens: {p_tokens}+{c_tokens}]:\n{p[:200]}...\n") 
                        f.write(f"[{timestamp}] RESPONSE:\n{res_content}\n")
                        f.write("-" * 40 + "\n")

                except Exception as e:
                    error_msg = f"Ollama Error: {e}"
                    print(error_msg)
                    results.append("")
                    
                    # 記錄錯誤到 Log
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"[ERROR] {error_msg}\n")
                    
        return results

    def get_usage(self):
        """回傳目前的 Token 使用統計 (Thread-safe)"""
        with self._usage_lock:
            return self.usage_stats.copy()

def model_from_config(config):
    model_type = config.get("name")
    if model_type == "Ollama_Forward":
        return Ollama_Forward(config)
    return None