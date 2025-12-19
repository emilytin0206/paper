import logging
import ollama

logger = logging.getLogger("OllamaClient")

class OllamaClient:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        try:
            # 使用 Chat API，並設定 temperature=0 確保測試穩定性
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            return ""