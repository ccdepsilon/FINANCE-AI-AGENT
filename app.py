import streamlit as st
import os
import json
import re
import subprocess
import time
import sys
import warnings
from typing import List, Dict, Tuple
from datetime import datetime
import dashscope

# --- 引入工具库 ---
try:
    import tools
except ImportError:
    st.error("【严重错误】未找到 tools.py 文件！请确保 tools.py 在同一目录下。")
    st.stop()

# ================= 页面配置 =================
st.set_page_config(
    page_title="AI 金融首席分析师 (Pro)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 全局配置与状态 =================

# 默认 API Key (也可以在侧边栏修改)
DEFAULT_API_KEY = "API_KEY"

# 模型配置
MODEL_SMART = "qwen-plus-latest"
MODEL_REASONING = "qwen3-max-2025-09-23"
MODEL_CODER = "qwen3-coder-plus"

# 鸭鸭搜索兼容
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# ================= 工具注册 (复刻 main.py) =================
TOOL_REGISTRY = {
    "download_data": tools.DataProcessor.download_us_stock,
    "feature_engineering": tools.DataProcessor.add_technical_features,
    "monte_carlo": tools.RiskEvaluator.run_monte_carlo,
    "distribution_test": tools.RiskEvaluator.run_distribution_test,
    "rf_prediction": tools.PricePredictor.run_rf_prediction,
    "market_regime": tools.MarketRegime.run_kmeans_regime,
    "seasonal_decomposition": tools.TimeSeriesMiner.run_seasonal_decomposition,
    "linear_regression": tools.PricePredictor.run_regression
}

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

# ================= 辅助函数 =================

def setup_run_folder():
    """每次运行创建一个新的输出目录"""
    base_output = "./output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def call_qwen(prompt: str, model: str, system_prompt: str = None, history: List = None) -> str:
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
            st.error(f"API Error: {response.code} - {response.message}")
            return None
    except Exception as e:
        st.error(f"API Exception: {e}")
        return None

def clean_code_block(text: str) -> str:
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_latex_content(text: str) -> str:
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
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass
    return None

# ================= Agent 类定义 (适配 Streamlit 输出) =================

class AgentNews:
    def run(self, stock_name: str, log_container):
        log_container.write(f"🕵️ [Agent A] 正在搜索关于 {stock_name} 的新闻...")
        results = []
        mock_news = f"""(注：网络搜索失败，使用模拟数据) 1. {stock_name} 季度财报显示AI业务强劲。2. 市场预期美联储降息。"""
        
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
                log_container.warning(f"搜索失败: {e}")
                search_context = mock_news
        
        system_prompt = "你是一名资深金融情报师。请总结核心利好、风险及市场情绪。直接输出文本。"
        res = call_qwen(search_context, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "无法获取情报分析结果。"

class AgentCoder:
    def run(self, requirement: str, current_csv_path: str, output_dir: str, error_msg: str = None):
        if not current_csv_path:
            return "print('Error: 没有数据文件路径，无法编写代码。请先运行 download_data。')"

        # 注意：这里我们把 output_dir 动态传入 Prompt，确保 Coder 保存到正确的 Session 文件夹
        system_prompt = f"""
        你是一个Python专家。
        **严厉约束:**
        1. **数据:** 读取本地 CSV: `df = pd.read_csv(r'{current_csv_path}', index_col='Date', parse_dates=True)`
        2. **路径:** 图片保存到 `{output_dir}` (绝对路径或相对路径)，文件名用英文。
        3. **反馈:** 保存图片后，执行 `print(f"IMAGE_SAVED: {{file_path}}")`。
        4. **禁止弹窗:** 不要使用 `plt.show()`。
        5. **只输出代码块**。
        """
        
        prompt = f"需求: {requirement}"
        if error_msg:
            prompt += f"\n\n上次运行输出(含报错): {error_msg}"
            
        code_raw = call_qwen(prompt, model=MODEL_CODER, system_prompt=system_prompt)
        if code_raw is None: return "print('Error: API_CALL_FAILED')"
        return clean_code_block(code_raw)

class LocalExecutor:
    def execute(self, code: str, output_dir: str): # 传入 output_dir 即使不用，保持接口一致性
        indented_code = "\n".join(["    " + line for line in code.splitlines()])
        
        # 动态创建 temp 文件在 output 目录下，避免冲突
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
        temp_file = os.path.join(output_dir, "temp_script.py")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(wrapper_script)
            
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='ignore',
                cwd=output_dir # 在输出目录下运行
            )
            output = result.stdout
            if "<<EXECUTION_SUCCESS>>" in output:
                clean_output = output.replace("<<EXECUTION_SUCCESS>>", "")
                return True, clean_output
            else:
                return False, output
        except Exception as e:
            return False, str(e)

class AgentOrchestrator:
    def __init__(self, output_dir):
        self.output_dir = output_dir # 绑定当前运行目录
        self.coder = AgentCoder()
        self.executor = LocalExecutor()
        self.memory = []
        self.current_csv_path = None
        self.is_processed = False
        self.has_called_coder = False
    
    def run(self, stock_code: str, goal: str, log_container, img_container) -> Tuple[str, List[str]]:
        log_container.write(f"🧠 [Agent B] 开始深度分析流程: {stock_code}")
        
        generated_images = []
        max_turns = 10
        
        sop_guideline = f"""
        **SOP (标准作业程序):**
        1. **数据准备:** 调用 `download_data` (100days+) -> 调用 `feature_engineering`。
        2. **深度分析:** 调用 `monte_carlo`, `distribution_test` 等工具至少3次。
        3. **定制绘图:** 至少调用一次 `call_coder`。
        **可用工具:** {TOOL_DESCRIPTIONS}
        """

        tool_used = []
        
        for turn in range(max_turns):
            status_hint = ""
            if not self.current_csv_path:
                status_hint = "【状态: 无数据】先调用 `download_data`。"
            elif not self.is_processed:
                status_hint = f"【状态: 有数据】先调用 `feature_engineering`。"
            else:
                status_hint = f"【状态: 数据就绪】请分析。未调用Coder" if not self.has_called_coder else "【状态: 数据就绪】"

            history_str = json.dumps(self.memory[-5:], indent=2, ensure_ascii=False)
            system_prompt = f"""你是一名量化策略分析师。
            {sop_guideline}
            已用工具: {tool_used}。
            {status_hint}
            **输出 JSON:** {{ "thought": "...", "action": "call_tool"|"call_coder"|"finish", "content": ... }}
            """
            user_prompt = f"目标: {goal}\n轮次: {turn+1}/{max_turns}\n已生成图表: {generated_images}\n记忆: {history_str}"
            
            response_raw = call_qwen(user_prompt, model=MODEL_SMART, system_prompt=system_prompt)
            if response_raw is None: continue
            
            decision = extract_json(response_raw)
            if not decision: continue
            
            thought = decision.get('thought')
            action = decision.get('action')
            content = decision.get('content')
            
            # UI 日志输出
            with log_container.expander(f"Turn {turn+1}: {action}", expanded=False):
                st.write(f"**Thought:** {thought}")
                st.write(f"**Content:** {content}")

            if action == "finish":
                if not self.is_processed or not self.has_called_coder:
                    self.memory.append({"role": "System", "content": "驳回：未完成数据处理或未调用Coder。"})
                    continue
                return str(self.memory), generated_images
            
            elif action == "call_tool":
                tool_name = content.get("tool_name")
                tool_used.append(tool_name)
                params = content.get("params", {})
                
                # 注入 df_path 和 output_dir (如果需要)
                if "df_path" not in params and self.current_csv_path:
                    params["df_path"] = self.current_csv_path
                
                # 特殊处理 download_data 的路径，确保它知道我们要保存到哪个 output_dir
                # 注意：tools.py 里的 download_us_stock 默认是 OUTPUT_DIR="./output"。
                # 为了让它保存到 run_xxxx 文件夹，我们需要修改全局变量或者 tool 函数接受 output_dir。
                # 鉴于不能修改 tools.py，我们这里只能在调用后把文件挪过来，或者临时修改 tools.OUTPUT_DIR
                # HACK: 临时修改 tools 模块的 OUTPUT_DIR
                if hasattr(tools, 'OUTPUT_DIR'):
                    tools.OUTPUT_DIR = self.output_dir
                if hasattr(tools.DataProcessor, 'OUTPUT_DIR'): # 如果类里面也有
                    pass 

                func = TOOL_REGISTRY.get(tool_name)
                if not func: continue
                
                try:
                    result = func(**params)
                    
                    if result.get("status") == "success":
                        new_images = result.get("images", [])
                        for img in new_images:
                            # 确保路径是相对于 self.output_dir 的 (因为 tools 可能会用 ./output)
                            # 如果 tools 生成在 ./output，我们需要把它 move 到 self.output_dir
                            # 这里假设 tools.OUTPUT_DIR 已经生效
                            if img not in generated_images:
                                generated_images.append(img)
                                # UI 展示图片
                                img_container.image(img, caption=os.path.basename(img))
                        
                        if "processed_path" in result:
                            self.current_csv_path = result["processed_path"]
                            self.is_processed = True
                        
                        self.memory.append({"role": "Agent B", "action": "call_tool", "tool": tool_name})
                        self.memory.append({"role": "System", "result": result.get("summary", "Done")})
                    else:
                        self.memory.append({"role": "System", "result": f"Tool Error: {result.get('error')}"})
                except Exception as e:
                    self.memory.append({"role": "System", "result": f"Exception: {e}"})

            elif action == "call_coder":
                if not self.current_csv_path:
                    self.memory.append({"role": "System", "content": "驳回：请先下载数据。"})
                    continue
                
                self.has_called_coder = True
                code_success = False
                retry = 0
                error_log = None
                
                while not code_success and retry < 3:
                    # 传入 self.output_dir 给 Coder
                    code = self.coder.run(content, self.current_csv_path, self.output_dir, error_msg=error_log)
                    success, output = self.executor.execute(code, self.output_dir)
                    
                    if success:
                        code_success = True
                        img_matches = re.findall(r"IMAGE_SAVED:\s*(.*?.png)", output)
                        for img in img_matches:
                            path = img.strip()
                            if path not in generated_images:
                                generated_images.append(path)
                                img_container.image(path, caption="Coder Generated")
                        
                        self.memory.append({"role": "Agent B", "action": "call_coder", "request": content})
                        self.memory.append({"role": "System", "result": f"Output: {output[:200]}..."})
                    else:
                        retry += 1
                        error_log = output
                
                if not code_success:
                    self.memory.append({"role": "System", "result": f"Failed: {error_log}"})

        return "分析强制结束。", generated_images

class AgentCIO:
    def run(self, news, quant, images, target, log_container):
        log_container.write("👔 [Agent E] 正在撰写深度研报...")
        img_list_desc = "\n".join([f"- {os.path.basename(p)}: {p}" for p in images])
        
        system_prompt = f"""你是一名首席投资官 (CIO)。针对 {target} 撰写深度研报。
        原则: 图数融合。必须引用量化日志中的数据来解释图表。
        """
        user_prompt = f"""
        【输入】
        1. 情报: {news}
        2. 量化日志: {quant}
        3. 图表: {img_list_desc}
        【任务】
        撰写《深度量化投资研报》: 1.核心建议 2.基本面 3.量化技术分析(重点,引用数据解释图表) 4.风险提示。
        """
        res = call_qwen(user_prompt, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "生成报告失败。"

class LatexCompiler:
    def compile(self, tex_code: str, output_dir: str):
        abs_output_dir = os.path.abspath(output_dir)
        tex_file = os.path.join(abs_output_dir, "report.tex")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(tex_code)
            
        try:
            cmd = ["xelatex", "-interaction=nonstopmode", "report.tex"]
            result = subprocess.run(
                cmd, cwd=abs_output_dir, # 关键：在各自的 output_dir 下运行
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=60, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0:
                return True, os.path.join(abs_output_dir, "report.pdf")
            else:
                log = result.stdout
                errs = [l for l in log.splitlines() if l.strip().startswith("!")]
                return False, "\n".join(errs[:5])
        except FileNotFoundError:
            return False, "未找到 xelatex，请检查本地 LaTeX 环境。"
        except Exception as e:
            return False, str(e)

class AgentLatex:
    def __init__(self):
        self.compiler = LatexCompiler()
    
    def run(self, text, images, output_dir, log_container):
        log_container.write("📄 [Agent F] 正在排版 PDF...")
        img_filenames = [os.path.basename(p) for p in images]
        img_context = ", ".join(img_filenames)
        
        base_system_prompt = f"""
        你是LaTeX排版专家。转为 `article` 类代码。
        必须包含: \\usepackage[UTF8]{{ctex}}, \\usepackage{{graphicx}}, \\usepackage{{float}}。
        图片引用仅用文件名: {img_context}。
        下划线 _ 和 % 必须转义。
        只输出代码。
        """
        
        current_code = ""
        error_history = ""
        for attempt in range(3):
            prompt = f"转换内容:\n{text}"
            if error_history: prompt += f"\n上次报错:\n{error_history}"
            
            response = call_qwen(prompt, model=MODEL_SMART, system_prompt=base_system_prompt)
            if not response: return None
            
            current_code = extract_latex_content(response)
            success, message = self.compiler.compile(current_code, output_dir)
            
            if success:
                log_container.success(f"编译成功！")
                return message # 返回 PDF 路径
            else:
                log_container.warning(f"编译尝试 {attempt+1} 失败: {message[:100]}...")
                error_history = message
        
        return None

# ================= Streamlit UI 逻辑 =================

# Sidebar
with st.sidebar:
    st.title("⚙️ 设置")
    api_key_input = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    dashscope.api_key = api_key_input
    
    st.markdown("---")
    target_input = st.text_input("股票代码", value="NVDA")
    run_btn = st.button("🚀 开始全流程分析", type="primary")

# Main Area
st.title("🤖 AI Quant Agent System")
st.caption(f"Target: {target_input} | Model: {MODEL_SMART} + {MODEL_REASONING}")

if run_btn:
    # 1. 创建独立运行文件夹
    current_run_dir = setup_run_folder()
    
    # 临时修改 Tools 的输出目录，确保工具把文件保存到新文件夹
    # (这需要 tools.py 里的函数使用 output_dir 参数，或者我们修改 tools 全局变量)
    if hasattr(tools, 'OUTPUT_DIR'):
        tools.OUTPUT_DIR = current_run_dir
    # 同时也为了 DataProcessor 等类可能有的硬编码，做一次覆盖
    os.environ["OUTPUT_DIR"] = current_run_dir 
    
    st.success(f"📂 工作目录已创建: {current_run_dir}")
    
    # 容器初始化
    status_box = st.status("正在执行多智能体工作流...", expanded=True)
    col_img, col_report = st.columns([1, 1])
    
    with status_box:
        # Step 1: News
        agent_a = AgentNews()
        news = agent_a.run(target_input, st)
        st.write("✅ 情报搜集完成")
        with st.expander("查看情报汇总"):
            st.write(news)
            
        # Step 2: Orchestrator
        agent_b = AgentOrchestrator(current_run_dir)
        goal_text = f"分析 {target_input}。SOP: 1.下载数据 2.计算特征 3.风险分析 4.绘制定制图表"
        
        # 图片展示容器
        st.write("📸 实时图表流:")
        img_gallery = st.container()
        
        quant_res, images = agent_b.run(target_input, goal_text, st, img_gallery)
        st.write("✅ 量化分析完成")
        
        # Step 3: CIO
        agent_e = AgentCIO()
        report = agent_e.run(news, quant_res, images, target_input, st)
        st.write("✅ 研报撰写完成")
        
        # Step 4: Latex
        agent_f = AgentLatex()
        pdf_path = agent_f.run(report, images, current_run_dir, st)
        
        status_box.update(label="✅ 全流程执行完毕!", state="complete", expanded=False)

    # 结果展示
    st.divider()
    st.header("📑 深度投资研报")
    
    # 左侧展示 Markdown 报告
    with st.container():
        st.markdown(report)
    
    # 提供下载
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            "📥 下载 Markdown 报告",
            data=report,
            file_name=f"{target_input}_report.md",
            mime="text/markdown"
        )
    
    if pdf_path and os.path.exists(pdf_path):
        with col_d2:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 下载 PDF 报告",
                    data=f,
                    file_name=f"{target_input}_report.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("PDF 生成失败或未找到本地 LaTeX 环境，仅提供 Markdown 下载。")

    # 底部图表画廊
    st.divider()
    st.subheader("📊 最终图表汇总")
    if images:
        cols = st.columns(3)
        for idx, img_path in enumerate(images):
            # 确保路径指向 current_run_dir
            if os.path.exists(img_path):
                with cols[idx % 3]:
                    st.image(img_path, caption=os.path.basename(img_path))
            else:
                # 尝试在 current_run_dir 找
                local_path = os.path.join(current_run_dir, os.path.basename(img_path))
                if os.path.exists(local_path):
                     with cols[idx % 3]:
                        st.image(local_path, caption=os.path.basename(local_path))