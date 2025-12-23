import os
import json
import re
import subprocess
import time
import sys
import warnings
from typing import List, Dict, Tuple
import dashscope

# --- 引入你的工具库 ---
# 确保 tools.py 在同一目录下，且包含 DataProcessor, RiskEvaluator 等所有类
try:
    import tools
except ImportError:
    print("【严重错误】未找到 tools.py 文件！请确保将之前编写的工具类保存为 tools.py。")
    sys.exit(1)

# =================配置区域=================
# 【重要】请在这里填入你的阿里云 API Key
dashscope.api_key = "API_KEY" 

# 文件保存路径
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置
MODEL_SMART = "qwen-plus-latest"           # 均衡，用于Agent B
MODEL_REASONING = "qwen3-max-2025-09-23" 
MODEL_CODER = "qwen3-coder-plus"


# ----------------- 兼容性处理：DuckDuckGo -----------------
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None
        print("Warning: duckduckgo_search not installed. Agent A will use mock data.")

# ================= 注册内置工具箱 =================

# 将字符串指令映射到 tools.py 中的具体函数
TOOL_REGISTRY = {
    # 基础数据类 (DataProcessor)
    "download_data": tools.DataProcessor.download_us_stock,
    "feature_engineering": tools.DataProcessor.add_technical_features,
    
    # 高级分析类
    "monte_carlo": tools.RiskEvaluator.run_monte_carlo,
    "distribution_test": tools.RiskEvaluator.run_distribution_test,
    "rf_prediction": tools.PricePredictor.run_rf_prediction,
    "market_regime": tools.MarketRegime.run_kmeans_regime,
    "seasonal_decomposition": tools.TimeSeriesMiner.run_seasonal_decomposition,
    "linear_regression": tools.PricePredictor.run_regression
}

# 工具描述，用于告诉 Agent B 怎么用
TOOL_DESCRIPTIONS = """
**可用工具箱 (Built-in Tools):**
1. `download_data(symbol, days)`: [必须第一步调用] 下载股票数据。返回 raw csv 路径。
2. `feature_engineering(df_path)`: [必须第二步调用] 计算 MACD, RSI, 布林带等指标。返回 processed csv 路径。
3. `monte_carlo(df_path)`: 执行蒙特卡洛模拟，分析 VaR 风险。
4. `distribution_test(df_path)`: 收益率分布检验（正态性、偏度、峰度）。
5. `rf_prediction(df_path)`: 随机森林预测股价涨跌。
6. `market_regime(df_path)`: K-Means 市场状态聚类。
7. `seasonal_decomposition(df_path)`: 时间序列分解。
"""

# =================基础工具函数=================

def call_qwen(prompt: str, model: str, system_prompt: str = None, history: List = None) -> str:
    """封装 DashScope API 调用 (带错误阻断)"""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if history:
        messages.extend(history)
    messages.append({'role': 'user', 'content': prompt})

    try:
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            result_format='message',
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"\n[API Error] Code: {response.code} - Message: {response.message}")
            return None
    except Exception as e:
        print(f"\n[API Exception] {e}")
        if "SSL" in str(e) or "HTTPSConnectionPool" in str(e):
             print("  -> 提示: 请在终端执行 pip install --upgrade dashscope urllib3 requests certifi")
        return None

def clean_code_block(text: str) -> str:
    """提取 Markdown 中的代码块"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_latex_content(text: str) -> str:
    """精准提取 LaTeX 源码"""
    pattern_md = r"```latex(.*?)```"
    match_md = re.search(pattern_md, text, re.DOTALL)
    if match_md:
        return match_md.group(1).strip()
    
    pattern_tex = r"(\\documentclass.*\\end{document})"
    match_tex = re.search(pattern_tex, text, re.DOTALL)
    if match_tex:
        return match_tex.group(1).strip()
        
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(r"\documentclass"):
            start_idx = i
            break
    return "\n".join(lines[start_idx:])

def extract_json(text: str) -> Dict:
    """从文本中提取 JSON 对象"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass
    return None

# =================各个 Agent 定义=================

class AgentNews:
    """Agent A: 文本情报分析师"""
    def run(self, stock_name: str):
        print(f"\n[Agent A] 正在搜索关于 {stock_name} 的新闻...")
        results = []
        
        mock_news = f"""
        (注：网络搜索失败，使用模拟数据)
        1. {stock_name} 季度财报显示AI数据中心业务强劲增长，毛利率维持高位。
        2. 行业竞争加剧，但 {stock_name} 凭借CUDA生态护城河依然稳固。
        3. 宏观层面，市场预期美联储降息利好科技成长股估值修复。
        """

        if DDGS is None:
            search_context = mock_news
        else:
            try:
                with DDGS() as ddgs:
                    ddgs_gen = ddgs.text(f"{stock_name} stock news analysis", region='wt-wt', timelimit='w', max_results=10)
                    if ddgs_gen:
                        for r in ddgs_gen:
                            results.append(f"Title: {r['title']}\nSnippet: {r['body']}")
                        search_context = "\n---\n".join(results)
                    else:
                        search_context = mock_news
            except Exception as e:
                print(f"  -> 搜索失败: {e}")
                search_context = mock_news
        
        system_prompt = "你是一名资深金融情报师。请总结核心利好、风险及市场情绪。直接输出文本。"
        res = call_qwen(search_context, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "无法获取情报分析结果。"

class AgentCoder:
    """Agent Coder: 负责写代码 (适配 Tools 版)"""
    def run(self, requirement: str, current_csv_path: str, error_msg: str = None):
        if not current_csv_path:
            return "print('Error: 没有数据文件路径，无法编写代码。请先运行 download_data。')"

        system_prompt = f"""
        你是一个Python专家。请编写代码完成需求。
        
        **严厉约束:**
        1. **数据源:** **禁止联网下载数据**。你必须读取本地 CSV 文件：`{current_csv_path}`。
           - 读取方法: `df = pd.read_csv(r'{current_csv_path}', index_col='Date', parse_dates=True)`
           - csv文件包括Date    Open	High	Low	Close	Volume	MA5	MA20	RSI	MACD	MACD_Signal	MACD_Hist	Boll_Upper	Boll_Lower	Boll_Width这些列
        2. **任务:** 基于读取的数据进行分析或绘图（Agent B 指定的任务）。
        3. **路径:** 图片保存到 `{OUTPUT_DIR}`，文件名必须用英文。
        4. **反馈:** 保存图片后，执行 `print(f"IMAGE_SAVED: {{file_path}}")`。
        5. **禁止弹窗:** 不要使用 `plt.show()`。
        6. **只输出代码块**。
        """
        
        prompt = f"需求: {requirement}"
        if error_msg:
            prompt += f"\n\n上次运行输出(含报错): {error_msg}"
            
        code_raw = call_qwen(prompt, model=MODEL_CODER, system_prompt=system_prompt)
        
        if code_raw is None:
            print("  -> [Error] API 调用失败，跳过代码生成。")
            return "print('Error: API_CALL_FAILED')" 
            
        return clean_code_block(code_raw)

class LocalExecutor:
    """本地代码执行环境"""
    def execute(self, code: str):
        indented_code = "\n".join(["    " + line for line in code.splitlines()])
        
        wrapper_script = f"""
import sys
import traceback
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main_logic():
{indented_code}

if __name__ == "__main__":
    try:
        main_logic()
        print("\\n<<EXECUTION_SUCCESS>>") 
    except Exception:
        traceback.print_exc()
        sys.exit(0)
"""
        temp_file = "temp_script.py"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(wrapper_script)
            
        print(f"  -> [Executor] 正在运行代码...")
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,     
                stderr=subprocess.STDOUT,   
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='ignore'
            )
            output = result.stdout
            
            if "<<EXECUTION_SUCCESS>>" in output:
                clean_output = output.replace("<<EXECUTION_SUCCESS>>", "")
                return True, clean_output
            else:
                return False, output

        except subprocess.TimeoutExpired:
            return False, "Code execution timed out (60s)."
        except Exception as e:
            return False, str(e)

class AgentOrchestrator:
    """Agent B: 核心编排器 (Tool-Use 增强版)"""
    def __init__(self):
        self.coder = AgentCoder()
        self.executor = LocalExecutor()
        self.memory = []
        # 状态追踪
        self.current_csv_path = None
        self.is_processed = False
        self.has_called_coder = False
    
    def run(self, stock_code: str, goal: str) -> Tuple[str, List[str]]:
        print(f"\n[Agent B] 开始深度分析流程: {stock_code}")
        
        generated_images = []
        max_turns = 10
        
        # 初始 SOP
        sop_guideline = f"""
        **SOP (标准作业程序):**
        1. **数据准备 (必须严格执行):**
           - 第一步: 调用 `download_data` 获取原始数据(100days以上)。
           - 第二步: 调用 `feature_engineering` 计算技术指标 (MACD, RSI等)。
           - **注意:** 只有执行完这两步，才能进行后续分析。
        2. **深度分析 (灵活选择):**
           - 选择可用工具中你认为有必要的各类函数进行分析获取结论，至少调用三次，鼓励更多次调用，不要反复调用使用过的工具，已经使用过的工具包括。
        3. **定制绘图 (必须执行):**
           - 至少调用一次 `call_coder`，鼓励多次调用，让程序员进行可用工具外的分析并返回结论（如绘制收盘价趋势图、计算并绘制 MACD 或 均线、计算 RSI 或 波动率、绘制收盘价与MA20的对比图，或者特定的成交量分析）。
        
        **可用工具:**
        {TOOL_DESCRIPTIONS}
        """

        tool_used = []
        for turn in range(max_turns):
            # 动态调整 System Prompt，引导 Agent 状态
            status_hint = ""
            if not self.current_csv_path:
                status_hint = "【当前状态: 无数据】你必须先调用 `download_data`。"
            elif not self.is_processed:
                status_hint = f"【当前状态: 有原始数据 {self.current_csv_path}】你必须调用 `feature_engineering` 进行处理。"
            else:
                status_hint = f"【当前状态: 数据就绪 {self.current_csv_path}】请选择高级分析工具，或者调用 Coder。"
                if not self.has_called_coder:
                    status_hint += " (记得：你还没有调用过 Coder，必须调用一次)"

            history_str = json.dumps(self.memory[-5:], indent=2, ensure_ascii=False)
            
            system_prompt = f"""你是一名量化策略分析师。
            {sop_guideline}

            你已经使用过的工具有{tool_used}。

            {status_hint}
            
            **输出 JSON:** {{ 
                "thought": "思考当前步骤...", 
                "action": "call_tool" | "call_coder" | "finish", 
                "content": {{ "tool_name": "...", "params": {{...}} }} 或 "Coder的自然语言需求" 
            }}
            """
            
            user_prompt = f"目标: {goal}\n轮次: {turn+1}/{max_turns}\n已生成图表: {generated_images}\n记忆: {history_str}"
            
            response_raw = call_qwen(user_prompt, model=MODEL_SMART, system_prompt=system_prompt)
            if response_raw is None: continue 
            
            decision = extract_json(response_raw)
            if not decision:
                print(f"  [Warn] JSON解析失败，重试。")
                continue
                
            thought = decision.get('thought')
            action = decision.get('action')
            content = decision.get('content')
            
            print(f"\n[Turn {turn+1}] Agent B 思考:\n└── {thought}")
            print(f"    Action: {action}")
            
            # --- 分支 1: 完成 ---
            if action == "finish":
                # 检查约束
                if not self.is_processed:
                    print("  -> [System] 驳回：未完成数据处理。")
                    self.memory.append({"role": "System", "content": "驳回：必须先完成 feature_engineering。"})
                    continue
                if not self.has_called_coder:
                    print("  -> [System] 驳回：未调用 Coder。")
                    self.memory.append({"role": "System", "content": "驳回：请至少调用一次 call_coder 画一个定制图表。"})
                    continue
                    
                return str(self.memory), generated_images
            
            # --- 分支 2: 调用工具箱 (tools.py) ---
            elif action == "call_tool":
                tool_name = content.get("tool_name")
                tool_used.append(tool_name)
                params = content.get("params", {})
                
                # 自动补全 df_path 参数
                if "df_path" not in params and self.current_csv_path:
                    params["df_path"] = self.current_csv_path
                
                print(f"  -> 调用工具: {tool_name} | 参数: {params}")
                
                func = TOOL_REGISTRY.get(tool_name)
                if not func:
                    print(f"  -> [Error] 工具 {tool_name} 不存在。")
                    continue
                
                # 执行工具函数
                try:
                    # 注意：tools 中的函数通常返回 dict: {'status': 'success', 'summary': '...', 'images': [], 'processed_path': '...'}
                    result = func(**params)
                    
                    if result.get("status") == "success":
                        print(f"  -> 工具执行成功。Summary: {result.get('summary')[:50]}...")
                        
                        # 收集图片
                        new_images = result.get("images", [])
                        for img in new_images:
                            if img not in generated_images:
                                generated_images.append(img)
                                print(f"  -> 捕获图片: {img}")
                        
                        # 特殊处理：如果工具返回了 processed_path (如 feature_engineering)
                        if "processed_path" in result:
                            self.current_csv_path = result["processed_path"]
                            self.is_processed = True
                            print(f"  -> [状态更新] 数据路径更新为: {self.current_csv_path}")
                        # 特殊处理：如果是 download_data，它返回 raw csv 路径（这里假设 download_us_stock 直接返回路径字符串，或者修改 wrapper）
                        # 根据之前的 tools.py 代码，download_us_stock 返回的是 str (path) 或 None
                        # 但为了统一，建议 tools.py 的函数都尽量返回 dict。
                        # 如果你的 download_us_stock 返回的是 path string:
                        if isinstance(result, str) and os.path.exists(result): # 兼容旧版
                             self.current_csv_path = result
                        
                        self.memory.append({"role": "Agent B", "action": "call_tool", "tool": tool_name})
                        self.memory.append({"role": "System", "result": result.get("summary", "Done")})
                        
                    else:
                        print(f"  -> 工具报错: {result.get('error')}")
                        self.memory.append({"role": "System", "result": f"Tool Error: {result.get('error')}"})
                        
                except Exception as e:
                    print(f"  -> 执行异常: {e}")
                    self.memory.append({"role": "System", "result": f"Exception: {e}"})

            # --- 分支 3: 调用 Coder ---
            elif action == "call_coder":
                if not self.current_csv_path:
                    print("  -> [System] 驳回：无数据，无法写代码。")
                    self.memory.append({"role": "System", "content": "驳回：请先下载数据。"})
                    continue
                
                self.has_called_coder = True
                code_success = False
                retry = 0
                error_log = None
                
                while not code_success and retry < 3:
                    code = self.coder.run(content, self.current_csv_path, error_msg=error_log)
                    success, output = self.executor.execute(code)
                    
                    if success:
                        code_success = True
                        print("  -> Coder 代码执行成功。")
                        img_matches = re.findall(r"IMAGE_SAVED:\s*(.*?.png)", output)
                        for img in img_matches:
                            path = img.strip()
                            if path not in generated_images:
                                generated_images.append(path)
                                print(f"  -> 捕获图片: {path}")
                        
                        self.memory.append({"role": "Agent B", "action": "call_coder", "request": content})
                        self.memory.append({"role": "System", "result": f"Output: {output[:200]}..."})
                    else:
                        retry += 1
                        error_log = output
                        print(f"  -> Coder 报错 (Retry {retry})")
                
                if not code_success:
                    self.memory.append({"role": "System", "result": f"Failed: {error_log}"})

        return "分析强制结束。", generated_images

class AgentCIO:
    """Agent E: 首席投资官 (深度研报版)"""
    def run(self, news, quant, images):
        print(f"\n[Agent E] 正在撰写深度研报 (关联数据与图表)...")
        
        # 将图片列表转换为带索引的字符串，方便模型引用
        img_list_desc = "\n".join([f"- {os.path.basename(p)}: {p}" for p in images])
        
        system_prompt = """
        你是一名华尔街顶级对冲基金的首席投资官 (CIO)。你需要针对{target}撰写一份极具专业深度的投资研报。
        
        **核心原则 (图数融合):**
        1. **严禁只放图不说话。** 每一张插入的图表下方，必须紧跟一段深度分析。
        2. **必须引用数据。** 你拥有量化分析师的完整运行日志。当展示图表时，必须从日志中提取对应的具体数值 (如 R-squared, VaR, 准确率, 波动率) 来解释图表。
        3. **逻辑自洽。** 如果量化模型预测下跌，但新闻全是利好，你需要进行风险提示或通过逻辑权衡给出最终判断。
        """
        
        user_prompt = f"""
        【输入数据】
        1. **市场情报 (News):**
        {news}
        
        2. **量化分析日志 (Quant Logic & Data):**
        (注意：这里包含了所有计算的具体数值，请仔细提取)
        {quant}
        
        3. **可用图表库 (Images):**
        {img_list_desc}
        
        【任务目标】
        请撰写一份格式标准的 **《深度量化投资研报》**。
        
        **研报结构要求:**
        
        **第一部分：核心投资建议 (Executive Summary)**
        - 给出明确评级：【强力买入 / 买入 / 持有 / 卖出】。
        - 给出目标仓位建议 (0-100%)。
        - 用一句话总结核心逻辑 (结合基本面和量化信号)。
        
        **第二部分：基本面与情报分析 (Fundamental Insight)**
        - 基于新闻情报，分析公司的护城河、近期催化剂及宏观环境。
        
        **第三部分：量化模型与技术分析 (Quantitative & Technical Analysis)**
        - **这是重点**。请根据提供的图表库，按逻辑顺序插入图表。
        - 引用格式: `[INSERT IMAGE: ./output/xxx.png]`
        - **关键要求**: 对于每一张图，必须结合“量化分析日志”中的数据进行解读。
          - *示例*: 插入 `monte_carlo.png` 后，必须写 "如图所示，通过1000次蒙特卡洛模拟，在95%置信度下的 VaR 为 -3.5%，表明下行风险可控..." (数据需来自日志)。
          - *示例*: 插入 `rf_prediction.png` 后，必须写 "随机森林模型准确率达到 85%，特征重要性显示 '成交量' 是最关键的预测因子..."。
        
        **第四部分：尾部风险提示 (Risk Factors)**
        - 结合分布检验 (Distribution Test) 或回撤数据，提示潜在风险。
        
        **其他你认为必要的部分（鼓励多写）**

        请开始撰写报告。输出 LaTeX 友好的纯文本。
        """
        
        # 调用推理能力最强的模型
        res = call_qwen(user_prompt, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "生成报告失败。"

class AgentLatex:
    """Agent F: 排版工程师 (修复特殊字符版)"""
    def __init__(self):
        self.compiler = LatexCompiler()
    
    def run(self, text, images):
        print(f"\n[Agent F] 开始生成并编译报告...")
        
        # 处理图片路径：只保留文件名，因为编译器会切换到 output 目录运行
        # 例如 ./output/macd.png -> macd.png
        img_filenames = [os.path.basename(p) for p in images]
        img_context = ", ".join(img_filenames)
        
        base_system_prompt = f"""
        你是LaTeX排版专家。请将金融研报转换为 `article` 类代码。
        
        **必须遵守的工程规范:**
        1. **宏包:** 必须包含:
           - `\\usepackage[UTF8]{{ctex}}` (支持中文)
           - `\\usepackage{{graphicx}}` (支持图片)
           - `\\usepackage{{geometry}}` (页面设置)
           - `\\usepackage{{float}}` (图片固定位置)
        
        2. **特殊字符转义 (至关重要):**
           - 文本中所有的下划线 `_` 必须转义为 `\\_` (例如: MACD_Signal -> MACD\\_Signal)。
           - 百分号 `%` 必须转义为 `\\%`。
        
        3. **图片插入:** - 只能使用文件名: {img_context}
           - 语法模板:
             \\begin{{figure}}[H]  % 注意用大写H固定位置
             \\centering
             \\includegraphics[width=0.8\\linewidth]{{FILENAME.png}} 
             \\caption{{图表说明}}
             \\end{{figure}}
             
        4. **文档结构:** 包含 \\title, \\author, \\maketitle, \\section 等。
        5. **输出:** 只输出 LaTeX 源码，不要包含 ```latex 标记。
        """
        
        current_code = ""
        error_history = ""
        
        for attempt in range(3):
            prompt = f"转换内容:\n{text}"
            if error_history:
                print(f"  -> [自愈] 正在修复 LaTeX 错误 (第 {attempt+1} 次)...")
                # 把错误日志喂回给模型，让它知道哪里错了
                prompt += f"\n\n上次编译报错 (请根据报错修正特殊字符或语法):\n{error_history}"
            
            response = call_qwen(prompt, model=MODEL_SMART, system_prompt=base_system_prompt)
            if not response: return None
            
            current_code = extract_latex_content(response)
            
            # --- 强制进行简单的 Python 层面的后处理 ---
            # 以防模型忘记转义，我们在代码层再兜底一次
            # 注意：这可能会误伤命令中的下划线，所以主要依赖模型，这里做简单检查
            if "usepackage" not in current_code:
                print("  -> [Warn] 模型生成的代码似乎不完整，重试...")
                continue
                
            success, message = self.compiler.compile(current_code, OUTPUT_DIR)
            
            if success:
                print(f"  -> {message}")
                return current_code
            else:
                print(f"  -> 编译失败: {message.strip()[:100]}...")
                error_history = message
        
        print("  -> [Error] 最终编译失败。请检查 output/report.tex 手动修复。")
        return current_code

class LatexCompiler:
    """负责本地编译 LaTeX 为 PDF (修复路径版)"""
    def compile(self, tex_code: str, output_dir: str = "./output"):
        # 1. 确保输出目录存在
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # 2. 保存 .tex 文件
        tex_filename = "report.tex"
        tex_file_path = os.path.join(abs_output_dir, tex_filename)
        
        with open(tex_file_path, "w", encoding="utf-8") as f:
            f.write(tex_code)
            
        print(f"  -> [Compiler] 正在编译 PDF (目录: {abs_output_dir})...")
        
        try:
            # 3. 调用 xelatex
            # 关键修改：cwd=abs_output_dir。让子进程直接在这个目录下运行。
            # 这样 LaTeX 里的图片路径只需要文件名 (如 image.png)，不需要 ./output/
            cmd = [
                "xelatex", 
                "-interaction=nonstopmode", # 不交互，报错直接把错误吐出来
                tex_filename 
            ]
            
            result = subprocess.run(
                cmd,
                cwd=abs_output_dir,         # <--- 核心修复：切换工作目录
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                encoding='utf-8',           # 捕获输出为文本
                errors='ignore'             # 防止日志乱码导致 Python 报错
            )
            
            # 4. 检查结果
            if result.returncode == 0:
                pdf_path = os.path.join(abs_output_dir, "report.pdf")
                return True, f"编译成功！PDF路径: {pdf_path}"
            else:
                # 提取错误日志
                log_content = result.stdout
                # 过滤出以 ! 开头的错误行
                errs = [line for line in log_content.splitlines() if line.strip().startswith("!")]
                # 如果没抓到 !，可能是其他错误，取最后几行
                if not errs:
                    errs = log_content.splitlines()[-10:]
                
                return False, "\n".join(errs[:5]) # 只返回前5条错误
                
        except FileNotFoundError:
            return False, "错误: 未找到 'xelatex' 命令。请检查环境变量。"
        except Exception as e:
            return False, str(e)

# =================主程序=================

def main():
    target = "麦当劳"  # 目标股票
    print(f"=== 金融分析系统启动 ({target}) ===")
    
    # 1. 搜集情报
    agent_a = AgentNews()
    news = agent_a.run(target)
    
    # 2. 量化分析 (Agent B + Tools + Coder)
    agent_b = AgentOrchestrator()
    # 目标中明确要求了 SOP 流程
    quant_res, images = agent_b.run(target, f"分析 {target}。SOP: 1.下载数据 2.计算特征 3.风险分析 4.绘制定制图表")
    
    # 3. 决策
    agent_e = AgentCIO()
    report = agent_e.run(news, quant_res, images)
    
    # 4. 排版
    agent_f = AgentLatex()
    latex_code = agent_f.run(report, images)
    
    if latex_code:
        with open(f"{OUTPUT_DIR}/report.tex", "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"\n=== 完成 ===\nPDF路径: {OUTPUT_DIR}/report.pdf")

if __name__ == "__main__":
    main()