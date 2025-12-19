class GenerationTemplate:
    """
    [Official Implementation]
    負責處理 Prompt 生成階段的模板。
    
    格式標籤:
    [APE]       : 模型需要生成的內容位置 (如果是 Forward 模式，通常位於句尾)
    [full_DEMO] : 完整 Few-shot 範例插入的位置
    [INPUT]     : "逆向工程" 目標的輸入
    [OUTPUT]    : "逆向工程" 目標的輸出
    """

    def __init__(self, template):
        self.template = template
        # 官方邏輯：必須檢查模板中是否包含 [APE] 標籤
        if '[APE]' not in self.template:
            raise ValueError("GenerationTemplate must contain '[APE]' placeholder.")
        assert self.template.count('[APE]') == 1, "Template must contain exactly one [APE] tag."

    def fill(self, full_demo='', input='', output=''):
        """
        填入數值並回傳最終的 Prompt。
        """
        # 1. 填入變數
        filled_prompt = self.template.replace('[full_DEMO]', full_demo)\
                                     .replace('[INPUT]', input)\
                                     .replace('[OUTPUT]', output)
        
        # 2. 處理 [APE] 標籤
        # 注意：官方原始碼在這裡其實是保留 [APE] 的，然後在 LLM Wrapper 裡移除。
        # 但為了讓你現在的 Ollama 流程順暢運作 (Forward Generation)，
        # 我們在這裡直接將 [APE] 替換為空字串，讓模型接著續寫。
        return filled_prompt.replace('[APE]', '').strip()


class EvalTemplate:
    """
    [Official Implementation]
    負責處理評估階段的模板。
    
    格式標籤:
    [PROMPT]    : 填入要測試的 Prompt (候選指令)
    [full_DEMO] : Few-shot 範例
    [INPUT]     : 測試資料的輸入
    [OUTPUT]    : 測試資料的輸出 (通常留空讓模型預測)
    """

    def __init__(self, template):
        self.template = template

    def fill(self, prompt='', full_demo='', input='', output=''):
        """
        Fills in the template with the given values.
        """
        return self.template.replace('[PROMPT]', prompt)\
                            .replace('[full_DEMO]', full_demo)\
                            .replace('[INPUT]', input)\
                            .replace('[OUTPUT]', output)

    def convert_to_generation_template(self):
        """
        官方功能：自動將評估模板轉換為生成模板。
        它會將 [PROMPT] 替換為 [APE]，讓你直接使用。
        """
        return GenerationTemplate(self.template.replace('[PROMPT]', '[APE]'))


class DemosTemplate:
    """
    [Official Implementation]
    負責處理 Few-shot 範例的拼接。
    格式標籤: [INPUT], [OUTPUT]
    """

    def __init__(self, template, delimiter='\n\n'):
        self.template = template
        self.delimiter = delimiter

    def fill(self, data):
        """
        Fills in the template with the given values. 
        Data is a tuple of lists: (inputs, outputs)
        """
        if not data or not data[0]:
            return ""
            
        inputs, outputs = data
        demos = []
        for i, (input_, output_) in enumerate(zip(inputs, outputs)):
            # 兼容處理：如果 output 是 list (如 instruction induction)，取第一個
            out_str = output_[0] if isinstance(output_, list) else output_
            
            demo = self.template.replace('[INPUT]', input_)\
                                .replace('[OUTPUT]', out_str)
            demos.append(demo)

        return self.delimiter.join(demos)