import streamlit as st
import os
import json
import re
import requests
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
    from ddgs import DDGS

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

def extract_json(text: str) -> Dict:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except:
        pass
    return None

def render_with_images(text: str):
    """
    解析文本中的 [INSERT IMAGE: path] 标记，实现图文混排显示
    """
    # 1. 使用正则表达式分割文本，保留分隔符（即图片标记）
    # 模式匹配: [INSERT IMAGE: ./output/xxx.png]
    pattern = r"(\[INSERT IMAGE: .*?\])"
    parts = re.split(pattern, text)

    for part in parts:
        # 检查是否是图片标记
        img_match = re.match(r"\[INSERT IMAGE: (.*?)\]", part)
        if img_match:
            img_path = img_match.group(1).strip()
            # 清理路径中的 ./output/ 前缀（因为 st.image 最好用相对路径或绝对路径，这里做个防御性处理）
            # Streamlit Cloud 中，./output/xxx.png 是可以的
            if os.path.exists(img_path):
                # 显示图片
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.warning(f"⚠️ 图片未找到: {img_path}")
        else:
            # 如果是普通文本，直接渲染 Markdown
            if part.strip():
                st.markdown(part)

# ================= Agent 类 (UI 适配版) =================

class AgentNews:
    def run(self, stock_name: str, log_container):
        log_container.write(f"🕵️ **Agent A (情报)**: 正在搜索关于 {stock_name} 的新闻...")
        results = []

        with DDGS() as ddgs:
            ddgs_gen = None
            count = 0
            while not ddgs_gen:
                count += 1
                if count > 3:
                    log_container.error("多次尝试搜索均失败。")
                    sys.exit(1)
                log_container.info("正在使用 DuckDuckGo 搜索新闻...")
                results = []
                ddgs_gen = ddgs.text(f"{stock_name} stock news analysis", region='wt-wt', timelimit='w', max_results=20)
                for r in ddgs_gen:
                    results.append(f"Title: {r['title']}\nSnippet: {r['body']}")
                search_context = "\n---\n".join(results)
                if not ddgs_gen:
                    # 这里的 serper_api_key 需要你提前定义或从环境变量读取
                    serper_api_key = "f6ae770b4865a03061057b8fc3721ebeeefc61de" 
                    
                    search_context = None
                    count = 0

                    log_container.info("正在使用 Serper.dev 搜索新闻...")
                    
                    try:
                        url = "https://google.serper.dev/search"
                        # tbs="qdr:w" 对应原代码的 timelimit='w' (过去一周)
                        payload = json.dumps({
                            "q": f"{stock_name} stock news analysis",
                            "num": 20,
                            "tbs": "qdr:w" 
                        })
                        headers = {
                            'X-API-KEY': serper_api_key,
                            'Content-Type': 'application/json'
                        }

                        response = requests.post(url, headers=headers, data=payload)
                        ddgs_gen = response
                        if response.status_code == 200:
                            data = response.json()
                            # Serper 的普通搜索结果在 'organic' 列表中
                            items = data.get("organic", [])
                            
                            results = []
                            for r in items:
                                # 对应原代码格式: Title + Snippet (原 body)
                                results.append(f"Title: {r.get('title')}\nSnippet: {r.get('snippet')}")
                            
                            search_context = "\n---\n".join(results)
                            success = True # 标记成功，用于跳出循环
                        else:
                            log_container.warning(f"Serper API 返回错误: {response.status_code}")
                    
                    except Exception as e:
                        log_container.warning(f"搜索请求发生异常: {e}")
                
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
        2. **任务:** 基于读取的数据进行分析或绘图（Agent B 指定的任务），绘图时必须使用英文标题或标签。
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
                                    log_container.image(img, caption=os.path.basename(img), width=500)
                            
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
                    log_container.caption("🚀 正在执行代码...")
                    success, output = self.executor.execute(code)
                    
                    if success:
                        code_success = True
                        img_matches = re.findall(r"IMAGE_SAVED:\s*(.*?.png)", output)
                        for img in img_matches:
                            path = img.strip()
                            if path not in generated_images:
                                generated_images.append(path)
                                log_container.image(img, caption=os.path.basename(img), width=500)
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
        
        # 保持原 Prompt 不变
        system_prompt = """
        你是一名华尔街顶级对冲基金的首席投资官 (CIO)。你需要针对{target}撰写一份极具专业深度的投资研报。
        **核心原则 (图数融合):**
        1. **严禁只放图不说话。** 每一张插入的图表下方，必须紧跟一段深度分析。
        2. **必须引用数据。** 你拥有量化分析师的完整运行日志。当展示图表时，必须从日志中提取对应的具体数值 (如 R-squared, VaR, 准确率, 波动率) 来解释图表。
        3. **逻辑自洽。** 如果量化模型预测下跌，但新闻全是利好，你需要进行风险提示或通过逻辑权衡给出最终判断。
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
        res = call_qwen(user_prompt, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "生成报告失败。"

class AgentMarkdown:
    """Agent F: Markdown 排版专家"""
    def run(self, text, images, log_container):
        log_container.write("📝 **Agent F (排版)**: 正在进行 Markdown 排版优化...")
        
        # 简单优化：确保图片路径格式统一，适合下载保存
        # 将 [INSERT IMAGE: ...] 转换为标准 Markdown 图片语法 ![Image](path) 方便用户下载md文件后查看
        # 但为了 Streamlit 的图文混排显示，我们主要依赖原始的 [INSERT IMAGE: ...] 标记进行切分
        
        # 生成一个供下载的纯 Markdown 版本
        downloadable_md = text
        for img_path in images:
            filename = os.path.basename(img_path)
            # 替换标记为标准 MD 语法
            # 注意：下载后图片通常和md在同一目录，所以去掉 ./output/
            placeholder = f"[INSERT IMAGE: {img_path}]"
            md_image = f"\n![{filename}]({filename})\n" 
            downloadable_md = downloadable_md.replace(placeholder, md_image)
            
        return text, downloadable_md

# ================= 主流程 =================

def main():
    st.title("🤖 基于多智能体协作的上市公司多维度自动化研报生成系统（陈长道 弓望涛 刘小淅 温欣）")
    st.markdown("---")

    # Sidebar: 配置
    with st.sidebar:
        st.header("Settings")
        api_key = st.secrets.get("DASHSCOPE_API_KEY", "")
        if api_key:
            dashscope.api_key = api_key
        
        target = st.text_input("目标股票 (Target Stock)", value="NVIDIA")
        run_btn = st.button("🚀 启动分析 (Start Analysis)", type="primary")
        
        st.info("说明：本系统使用多智能体架构 (News -> Quant -> Coder -> CIO) 生成深度研报。")

    if run_btn:
        if not api_key:
            st.error("请先输入 DashScope API Key！")
            st.stop()
            
        status_container = st.status("正在运行 AI 分析流程...", expanded=True)
        
        # 1. 搜集情报
        agent_a = AgentNews()
        news = agent_a.run(target, status_container)
        
        # 2. 量化分析
        agent_b = AgentOrchestrator()
        quant_res, images = agent_b.run(target, f"分析 {target}。SOP: 1.下载数据 2.计算特征 3.风险分析 4.绘制定制图表", status_container)
        
        # 3. 决策
        agent_e = AgentCIO()
        raw_report = agent_e.run(news, quant_res, images, status_container)
        
        # 4. 排版 (Markdown)
        agent_f = AgentMarkdown()
        # raw_report 用于页面渲染 (保留标记), final_md 用于下载 (标准MD语法)
        display_report, download_report = agent_f.run(raw_report, images, status_container)
        
        status_container.update(label="✅ 分析完成！", state="complete", expanded=False)
        
        # --- 结果展示区 ---
        st.divider()
        st.header(f"📊 {target} 深度投资研报")
        
        # 使用自定义渲染函数，实现图文混排
        render_with_images(display_report)
        
        st.divider()
        st.subheader("💾 下载报告")
        st.download_button(
            label="⬇️ 下载 Markdown 源码 (包含图片引用)",
            data=download_report,
            file_name=f"{target}_report.md",
            mime="text/markdown"
        )
        st.info("提示：下载 .md 文件后，请确保图片文件（在 output 文件夹中）与 .md 文件在同一目录下，以正常显示图片。")

if __name__ == "__main__":
    main()