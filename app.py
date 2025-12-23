import streamlit as st
import os
import json
import re
import subprocess
import time
import sys
import warnings
import pandas as pd
from typing import List, Dict, Tuple
import dashscope

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="AI 量化投资研报生成器",
    page_icon="📈",
    layout="wide"
)

# --- 检查 tools.py ---
try:
    import tools
except ImportError:
    st.error("【严重错误】未找到 tools.py 文件！请确保将其上传到 GitHub 仓库根目录。")
    st.stop()

# --- 配置区域 ---
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置 (严格保持原样)
MODEL_SMART = "qwen-plus-latest"
MODEL_REASONING = "qwen3-max-2025-09-23"
MODEL_CODER = "qwen3-coder-plus"

# 工具描述
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

# 注册工具
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

# --- 兼容性处理 ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# ================= 辅助函数 =================

def call_qwen(prompt: str, model: str, system_prompt: str = None, history: List = None) -> str:
    """封装 DashScope API 调用"""
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
            st.error(f"[API Error] Code: {response.code} - Message: {response.message}")
            return None
    except Exception as e:
        st.error(f"[API Exception] {e}")
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

# ================= Agent 类 (UI 适配版) =================

class AgentNews:
    def run(self, stock_name: str, log_container):
        log_container.write(f"🕵️ **Agent A (情报)**: 正在搜索关于 {stock_name} 的新闻...")
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
                log_container.warning(f"搜索 API 异常: {e}，使用模拟数据。")
                search_context = mock_news
        
        system_prompt = "你是一名资深金融情报师。请总结核心利好、风险及市场情绪。直接输出文本。"
        res = call_qwen(search_context, model=MODEL_REASONING, system_prompt=system_prompt)
        final_res = res if res else "无法获取情报分析结果。"
        log_container.success("情报分析完成。")
        with log_container.expander("查看情报摘要"):
            st.markdown(final_res)
        return final_res

class AgentCoder:
    def run(self, requirement: str, current_csv_path: str, error_msg: str = None):
        if not current_csv_path:
            return "print('Error: 没有数据文件路径')"

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
        return clean_code_block(code_raw) if code_raw else "print('Error: API_CALL_FAILED')"

class LocalExecutor:
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
            
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,   
                text=True, timeout=60, encoding='utf-8', errors='ignore'
            )
            output = result.stdout
            if "<<EXECUTION_SUCCESS>>" in output:
                return True, output.replace("<<EXECUTION_SUCCESS>>", "")
            else:
                return False, output
        except Exception as e:
            return False, str(e)

class AgentOrchestrator:
    def __init__(self):
        self.coder = AgentCoder()
        self.executor = LocalExecutor()
        self.memory = []
        self.current_csv_path = None
        self.is_processed = False
        self.has_called_coder = False
    
    def run(self, stock_code: str, goal: str, log_container) -> Tuple[str, List[str]]:
        log_container.write(f"🧠 **Agent B (指挥官)**: 开始执行 SOP 分析流程...")
        generated_images = []
        max_turns = 10
        tool_used = []
        
        sop_guideline = f"""
        **SOP (标准作业程序):**
        1. **数据准备 (必须严格执行):**
           - 第一步: 调用 `download_data` 获取原始数据(一年以上)。
           - 第二步: 调用 `feature_engineering` 计算技术指标 (MACD, RSI等)。
           - **注意:** 只有执行完这两步，才能进行后续分析。
        2. **深度分析 (灵活选择):**
           - 选择可用工具中你认为有必要的各类函数进行分析获取结论，至少调用三次，鼓励更多次调用，不要反复调用使用过的工具。
        3. **定制绘图 (必须执行):**
           - 至少调用一次 `call_coder`，鼓励多次调用，让程序员进行可用工具外的分析并返回结论（如绘制收盘价趋势图、计算并绘制 MACD 或 均线、计算 RSI 或 波动率、绘制收盘价与MA20的对比图，或者特定的成交量分析）。
        
        **可用工具:**
        {TOOL_DESCRIPTIONS}
        """

        progress_bar = log_container.progress(0, text="初始化 Agent B...")

        for turn in range(max_turns):
            progress_bar.progress((turn + 1) / max_turns, text=f"Agent B 思考中 (轮次 {turn+1}/{max_turns})...")
            
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
            if not decision: continue
                
            thought = decision.get('thought')
            action = decision.get('action')
            content = decision.get('content')
            
            log_container.info(f"👉 **Step {turn+1}**: {thought}")

            if action == "finish":
                if not self.is_processed or not self.has_called_coder:
                     self.memory.append({"role": "System", "content": "驳回：SOP未完成(需数据处理+至少一次Coder)。"})
                     continue
                progress_bar.empty()
                return str(self.memory), generated_images
            
            elif action == "call_tool":
                tool_name = content.get("tool_name")
                tool_used.append(tool_name)
                params = content.get("params", {})
                if "df_path" not in params and self.current_csv_path:
                    params["df_path"] = self.current_csv_path
                
                func = TOOL_REGISTRY.get(tool_name)
                if func:
                    try:
                        result = func(**params)
                        # 兼容处理: 某些旧函数可能返回字符串路径
                        if isinstance(result, str):
                            if os.path.exists(result):
                                self.current_csv_path = result
                                result = {"status": "success", "summary": "File saved", "images": [], "processed_path": result}
                        
                        if isinstance(result, dict) and result.get("status") == "success":
                            log_container.caption(f"🔧 工具执行成功: {result.get('summary')[:100]}...")
                            new_images = result.get("images", [])
                            for img in new_images:
                                if img not in generated_images:
                                    generated_images.append(img)
                                    log_container.image(img, caption=os.path.basename(img), width=400)
                            
                            if "processed_path" in result:
                                self.current_csv_path = result["processed_path"]
                                self.is_processed = True
                            
                            self.memory.append({"role": "Agent B", "action": "call_tool", "tool": tool_name})
                            self.memory.append({"role": "System", "result": result.get("summary", "Done")})
                        else:
                             err = result.get("error") if isinstance(result, dict) else "Unknown error"
                             log_container.error(f"工具报错: {err}")
                             self.memory.append({"role": "System", "result": f"Tool Error: {err}"})
                    except Exception as e:
                        log_container.error(f"执行异常: {e}")
            
            elif action == "call_coder":
                self.has_called_coder = True
                code_success = False
                retry = 0
                error_log = None
                log_container.caption(f"💻 调用程序员编写: {content}")
                
                while not code_success and retry < 3:
                    code = self.coder.run(content, self.current_csv_path, error_msg=error_log)
                    success, output = self.executor.execute(code)
                    
                    if success:
                        code_success = True
                        img_matches = re.findall(r"IMAGE_SAVED:\s*(.*?.png)", output)
                        for img in img_matches:
                            path = img.strip()
                            if path not in generated_images:
                                generated_images.append(path)
                                log_container.image(path, caption="Coder Generated", width=400)
                        self.memory.append({"role": "Agent B", "action": "call_coder", "request": content})
                        self.memory.append({"role": "System", "result": f"Output: {output[:200]}..."})
                    else:
                        retry += 1
                        error_log = output
                        log_container.warning(f"代码运行失败，正在重试 ({retry}/3)...")
                
                if not code_success:
                    self.memory.append({"role": "System", "result": f"Failed: {error_log}"})

        return "分析强制结束。", generated_images

class AgentCIO:
    def run(self, news, quant, images, log_container):
        log_container.write("👔 **Agent E (CIO)**: 正在撰写深度研报...")
        img_list_desc = "\n".join([f"- {os.path.basename(p)}: {p}" for p in images])
        
        system_prompt = """
        你是一名华尔街顶级对冲基金的首席投资官 (CIO)。你需要针对{target}撰写一份极具专业深度的投资研报。
        **核心原则 (图数融合):**
        1. **严禁只放图不说话。** 每一张插入的图表下方，必须紧跟一段深度分析。
        2. **必须引用数据。** 当展示图表时，必须从日志中提取对应的具体数值 (如 R-squared, VaR, 准确率, 波动率) 来解释图表。
        3. **逻辑自洽。**
        """
        
        user_prompt = f"""
        【输入数据】
        1. **市场情报:** {news}
        2. **量化分析日志:** {quant}
        3. **可用图表库:** {img_list_desc}
        
        【任务目标】
        请撰写一份格式标准的 **《深度量化投资研报》**。
        **研报结构要求:**
        **第一部分：核心投资建议 (Executive Summary)**
        - 评级、仓位建议、核心逻辑。
        **第二部分：基本面与情报分析**
        **第三部分：量化模型与技术分析**
        - 引用格式: `[INSERT IMAGE: ./output/xxx.png]`
        - 对于每一张图，必须结合“量化分析日志”中的数据进行解读。
        **第四部分：尾部风险提示**
        请开始撰写报告。输出 LaTeX 友好的纯文本。
        """
        res = call_qwen(user_prompt, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "生成报告失败。"

class AgentLatex:
    def __init__(self):
        self.compiler = LatexCompiler()
    
    def run(self, text, images, log_container):
        log_container.write("📝 **Agent F (排版)**: 正在生成 LaTeX 代码并尝试编译...")
        img_filenames = [os.path.basename(p) for p in images]
        img_context = ", ".join(img_filenames)
        
        base_system_prompt = f"""
        你是LaTeX排版专家。请将金融研报转换为 `article` 类代码。
        **必须遵守的工程规范:**
        1. **宏包:** 必须包含: `\\usepackage[UTF8]{{ctex}}`, `\\usepackage{{graphicx}}`, `\\usepackage{{geometry}}`, `\\usepackage{{float}}`。
        2. **特殊字符转义:** 下划线 `_` 转 `\\_`，百分号 `%` 转 `\\%`。
        3. **图片插入:** 只能使用文件名: {img_context}，语法模板:
             \\begin{{figure}}[H]
             \\centering
             \\includegraphics[width=0.8\\linewidth]{{FILENAME.png}} 
             \\caption{{图表说明}}
             \\end{{figure}}
        4. **输出:** 只输出 LaTeX 源码。
        """
        
        prompt = f"转换内容:\n{text}"
        response = call_qwen(prompt, model=MODEL_SMART, system_prompt=base_system_prompt)
        if not response: return None
        
        current_code = extract_latex_content(response)
        success, message = self.compiler.compile(current_code, OUTPUT_DIR)
        
        if success:
            log_container.success("PDF 编译成功！")
            return current_code, True, os.path.join(OUTPUT_DIR, "report.pdf")
        else:
            log_container.warning(f"PDF 编译失败 (可能是云端环境缺少 XeLaTeX): {message[:100]}...")
            return current_code, False, None

class LatexCompiler:
    def compile(self, tex_code: str, output_dir: str = "./output"):
        abs_output_dir = os.path.abspath(output_dir)
        tex_filename = "report.tex"
        tex_file_path = os.path.join(abs_output_dir, tex_filename)
        
        with open(tex_file_path, "w", encoding="utf-8") as f:
            f.write(tex_code)
            
        try:
            cmd = ["xelatex", "-interaction=nonstopmode", tex_filename]
            result = subprocess.run(
                cmd, cwd=abs_output_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=60, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0:
                return True, "Success"
            else:
                return False, result.stdout
        except Exception as e:
            return False, str(e)

# ================= 主流程 =================

def main():
    st.title("🤖 AI Agent Financial Analyst System")
    st.markdown("---")

    # Sidebar: 配置
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("DashScope API Key", value=st.secrets.get("DASHSCOPE_API_KEY", ""), type="password")
        if api_key:
            dashscope.api_key = api_key
        
        target = st.text_input("目标股票 (Target Stock)", value="NVDA")
        run_btn = st.button("🚀 启动分析 (Start Analysis)", type="primary")
        
        st.info("说明：本系统使用多智能体架构 (News -> Quant -> Coder -> CIO) 生成深度研报。")

    if run_btn:
        if not api_key:
            st.error("请先输入 DashScope API Key！")
            st.stop()
            
        # 容器化显示日志
        status_container = st.status("正在运行 AI 分析流程...", expanded=True)
        
        # 1. 搜集情报
        agent_a = AgentNews()
        news = agent_a.run(target, status_container)
        
        # 2. 量化分析
        agent_b = AgentOrchestrator()
        quant_res, images = agent_b.run(target, f"分析 {target}。SOP: 1.下载数据 2.计算特征 3.风险分析 4.绘制定制图表", status_container)
        
        # 3. 决策
        agent_e = AgentCIO()
        report_text = agent_e.run(news, quant_res, images, status_container)
        
        # 4. 排版
        agent_f = AgentLatex()
        latex_code, pdf_success, pdf_path = agent_f.run(report_text, images, status_container)
        
        status_container.update(label="✅ 分析完成！", state="complete", expanded=False)
        
        # --- 结果展示区 ---
        st.divider()
        st.header(f"📊 {target} 深度投资研报")
        
        tab1, tab2, tab3 = st.tabs(["📄 研报全文 (Markdown)", "🖼️ 生成图表", "💾 下载资源"])
        
        with tab1:
            # 简单处理 Markdown 中的图片引用，使其在 Streamlit 显示
            # 将 [INSERT IMAGE: ./output/xxx.png] 替换为空，因为图表在 Tab2 展示，或者可以直接渲染
            display_text = report_text
            st.markdown(display_text)
            
        with tab2:
            cols = st.columns(2)
            for i, img_path in enumerate(images):
                with cols[i % 2]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path))
        
        with tab3:
            st.subheader("下载选项")
            if pdf_success and pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ 下载 PDF 研报", f, file_name=f"{target}_report.pdf", mime="application/pdf")
            else:
                st.warning("由于云端环境限制，PDF 编译失败。您可以下载 LaTeX 源码在本地编译。")
            
            st.download_button("⬇️ 下载 LaTeX 源码", latex_code, file_name=f"{target}_report.tex")
            st.download_button("⬇️ 下载 Markdown 源码", report_text, file_name=f"{target}_report.md")

if __name__ == "__main__":
    main()