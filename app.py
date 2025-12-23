import streamlit as st
import os
import json
import re
import subprocess
import time
import sys
import warnings
from typing import List, Dict, Tuple
import dashscope

# ================= 页面配置 =================
st.set_page_config(
    page_title="AI 量化投资研报平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 引入工具库 =================
# 确保 tools.py 在同一目录下
try:
    import tools
except ImportError:
    st.error("【严重错误】未找到 tools.py 文件！请确保 tools.py 上传至同一目录。")
    st.stop()

# ================= 配置与初始化 =================
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置 (使用阿里云 Qwen)
MODEL_SMART = "qwen-plus"           # 均衡模型，用于逻辑控制
MODEL_REASONING = "qwen-max"        # 推理模型，用于写研报
MODEL_CODER = "qwen-plus"           # 编程模型 (Coder 使用 Plus 稳定性较好)

# ----------------- 兼容性处理：DuckDuckGo -----------------
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# ================= 注册内置工具箱 =================
# 将字符串指令映射到 tools.py 中的具体函数
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

# ================= 基础 LLM 接口 =================

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
            st.error(f"API Error: {response.code} - {response.message}")
            return None
    except Exception as e:
        st.error(f"API Exception: {e}")
        return None

def clean_code_block(text: str) -> str:
    """提取 Markdown 中的 Python 代码块"""
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

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
    def run(self, stock_name: str, log_container):
        log_container.markdown(f"**[Agent A]** 正在搜索关于 {stock_name} 的新闻...")
        results = []
        
        mock_news = f"""
        (注：网络搜索失败或API受限，使用模拟数据)
        1. {stock_name} 季度财报显示AI数据中心业务强劲增长，毛利率维持高位。
        2. 行业竞争加剧，但 {stock_name} 凭借生态护城河依然稳固。
        3. 宏观层面，市场预期美联储降息利好科技成长股估值修复。
        """

        if DDGS is None:
            search_context = mock_news
        else:
            try:
                # 尝试搜索，如果失败则回退
                with DDGS() as ddgs:
                    ddgs_gen = ddgs.text(f"{stock_name} stock news analysis", region='wt-wt', timelimit='w', max_results=5)
                    if ddgs_gen:
                        for r in ddgs_gen:
                            results.append(f"Title: {r['title']}\nSnippet: {r['body']}")
                        search_context = "\n---\n".join(results)
                    else:
                        search_context = mock_news
            except Exception as e:
                log_container.warning(f"DuckDuckGo 搜索出现问题: {e}，使用模拟数据。")
                search_context = mock_news
        
        system_prompt = "你是一名资深金融情报师。请总结核心利好、风险及市场情绪。直接输出文本。"
        res = call_qwen(search_context, model=MODEL_REASONING, system_prompt=system_prompt)
        return res if res else "无法获取情报分析结果。"

class AgentCoder:
    """Agent Coder: 负责写代码"""
    def run(self, requirement: str, current_csv_path: str, error_msg: str = None):
        if not current_csv_path:
            return "print('Error: 没有数据文件路径')"

        system_prompt = f"""
        你是一个Python专家。请编写代码完成需求。
        
        **严厉约束:**
        1. **数据源:** 必须读取本地 CSV 文件：`{current_csv_path}`。
           - 读取方法: `df = pd.read_csv(r'{current_csv_path}', index_col='Date', parse_dates=True)`
        2. **路径:** 图片保存到 `{OUTPUT_DIR}`，文件名必须用英文。
        3. **反馈:** 保存图片后，执行 `print(f"IMAGE_SAVED: {{file_path}}")`。
        4. **禁止弹窗:** 不要使用 `plt.show()`。
        5. **只输出代码块**。
        """
        
        prompt = f"需求: {requirement}"
        if error_msg:
            prompt += f"\n\n上次运行输出(含报错): {error_msg}"
            
        code_raw = call_qwen(prompt, model=MODEL_CODER, system_prompt=system_prompt)
        return clean_code_block(code_raw) if code_raw else None

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
        except Exception as e:
            return False, str(e)

class AgentOrchestrator:
    """Agent B: 核心编排器"""
    def __init__(self):
        self.coder = AgentCoder()
        self.executor = LocalExecutor()
        self.memory = []
        self.current_csv_path = None
        self.is_processed = False
        self.has_called_coder = False
    
    def run(self, stock_code: str, goal: str, log_container) -> Tuple[str, List[str]]:
        log_container.markdown(f"### [Agent B] 开始深度分析流程: {stock_code}")
        
        generated_images = []
        max_turns = 10
        tool_used = []

        sop_guideline = f"""
        **SOP:**
        1. 必须先调用 `download_data`。
        2. 必须接着调用 `feature_engineering`。
        3. 之后自由使用工具分析，至少3次。
        4. 必须至少调用一次 `call_coder` 进行定制绘图。
        **可用工具:** {TOOL_DESCRIPTIONS}
        """

        for turn in range(max_turns):
            # 状态提示
            status_hint = ""
            if not self.current_csv_path:
                status_hint = "【当前状态: 无数据】请调用 download_data。"
            elif not self.is_processed:
                status_hint = f"【当前状态: 有原始数据】请调用 feature_engineering。"
            else:
                status_hint = f"【当前状态: 数据就绪】请选择分析工具或 call_coder。"

            system_prompt = f"""你是一名量化策略分析师。
            {sop_guideline}
            已用工具: {tool_used}
            {status_hint}
            **输出 JSON:** {{ "thought": "...", "action": "call_tool" | "call_coder" | "finish", "content": {{...}} }}
            """
            
            user_prompt = f"目标: {goal}\n轮次: {turn+1}/{max_turns}\n已生成图表: {generated_images}"
            
            response_raw = call_qwen(user_prompt, model=MODEL_SMART, system_prompt=system_prompt)
            if not response_raw: continue
            
            decision = extract_json(response_raw)
            if not decision: continue
                
            thought = decision.get('thought')
            action = decision.get('action')
            content = decision.get('content')
            
            log_container.info(f"Step {turn+1}: {thought}")
            
            if action == "finish":
                if self.is_processed and self.has_called_coder:
                    return str(self.memory), generated_images
                else:
                    log_container.warning("系统提示：流程未完成，强制继续。")
            
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
                        # 处理结果
                        if isinstance(result, dict) and result.get("status") == "success":
                            log_container.success(f"工具 {tool_name} 执行成功")
                            
                            # 更新图片
                            for img in result.get("images", []):
                                if img not in generated_images:
                                    generated_images.append(img)
                                    st.image(img, caption=os.path.basename(img))
                            
                            # 更新路径
                            if "processed_path" in result:
                                self.current_csv_path = result["processed_path"]
                                self.is_processed = True
                            elif tool_name == "download_data" and "processed_path" in result:
                                self.current_csv_path = result["processed_path"]

                            self.memory.append({"role": "Agent B", "action": tool_name, "result": result.get("summary")})
                        else:
                            log_container.error(f"工具报错: {result}")
                            self.memory.append({"role": "System", "result": f"Error: {result}"})
                    except Exception as e:
                        log_container.error(f"执行异常: {e}")
            
            elif action == "call_coder":
                if not self.current_csv_path:
                    log_container.warning("无数据，无法写代码。")
                    continue
                
                self.has_called_coder = True
                log_container.markdown(f"Wait... Coder 正在绘图: {content}")
                
                # 简单重试机制
                for _ in range(2):
                    code = self.coder.run(content, self.current_csv_path)
                    if code:
                        success, output = self.executor.execute(code)
                        if success:
                            log_container.success("Coder 代码执行成功")
                            img_matches = re.findall(r"IMAGE_SAVED:\s*(.*?.png)", output)
                            for img in img_matches:
                                path = img.strip()
                                if path not in generated_images:
                                    generated_images.append(path)
                                    st.image(path, caption="Coder Generated")
                            self.memory.append({"role": "Coder", "request": content, "result": "Success"})
                            break
                        else:
                            log_container.warning(f"Coder 执行报错，重试中... \n{output[:100]}")
                            
        return str(self.memory), generated_images

class AgentCIO:
    """Agent E: 首席投资官 (Markdown 报告版)"""
    def run(self, news, quant, images, target):
        # 准备图片描述列表
        img_list_desc = "\n".join([f"- {os.path.basename(p)}" for p in images])
        
        system_prompt = """
        你是一名首席投资官 (CIO)。请针对{target}撰写一份极具专业深度的投资研报。
        
        **输出格式要求:**
        1. 使用标准的 **Markdown** 格式。
        2. 使用一级标题 `#` 表示报告题目，二级标题 `##` 表示章节。
        3. **严禁只放图不说话**。报告中提到图表时，必须结合【量化日志】中的具体数据进行分析。
        4. 不需要生成 LaTeX 代码，直接生成易于阅读的 Markdown 文本。
        """
        
        user_prompt = f"""
        【市场情报】
        {news}
        
        【量化日志 (包含具体数值)】
        {quant}
        
        【已生成图表列表】
        {img_list_desc}
        
        【任务】
        请撰写《深度量化投资研报》，结构如下：
        1. **核心投资建议** (评级、仓位、一句话逻辑)
        2. **基本面与情报分析**
        3. **量化模型与技术分析** (这是重点。请在文中适当位置提及相关图表，例如"（参考图表：macd.png）"，并详细解读数据)
        4. **尾部风险提示**
        
        请开始撰写。
        """
        
        res = call_qwen(user_prompt, model=MODEL_REASONING, system_prompt=system_prompt)
        return res

# ================= Streamlit 主界面逻辑 =================

def main():
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        # 优先读取 secrets，如果没有则显示输入框
        default_key = ""
        if "DASHSCOPE_API_KEY" in st.secrets:
            default_key = st.secrets["DASHSCOPE_API_KEY"]
            st.success("✅ API Key 已通过 Secrets 加载")
        
        api_key = st.text_input("DashScope API Key", value=default_key, type="password")
        if api_key:
            dashscope.api_key = api_key
            
        st.divider()
        stock_symbol = st.text_input("美股代码 (Symbol)", value="NVDA", help="例如: NVDA, TSLA, AAPL")
        target_name = st.text_input("公司名称", value="英伟达", help="用于生成报告标题")
        
        st.divider()
        st.caption("支持模型: Qwen-Plus, Qwen-Max")
        run_btn = st.button("🚀 开始 AI 全流程分析", type="primary", use_container_width=True)

    # 主区域
    st.title("🤖 AI Agent 深度研报生成器")
    st.markdown("""
    > 本系统通过多 Agent 协作模拟专业投研流程：
    > 1. **Agent A (情报)**: 搜集全网新闻与情绪。
    > 2. **Agent B (量化)**: 调用 Python 工具箱进行回测、蒙特卡洛模拟与归因分析。
    > 3. **Agent Coder**: 编写自定义代码绘制图表。
    > 4. **Agent E (CIO)**: 汇总数据撰写深度研报。
    """)
    
    st.divider()

    if run_btn:
        if not api_key:
            st.warning("⚠️ 请先在侧边栏输入 DashScope API Key。")
            return

        # 创建主容器
        main_container = st.container()
        
        # --- 阶段 1: 情报搜集 ---
        with st.status("🕵️ [阶段 1/3] Agent A: 正在搜集情报...", expanded=True) as status:
            agent_a = AgentNews()
            news = agent_a.run(target_name, st)
            st.text_area("情报摘要", news, height=100)
            status.update(label="✅ Agent A: 情报搜集完成", state="complete", expanded=False)

        # --- 阶段 2: 量化分析 ---
        with st.status("📊 [阶段 2/3] Agent B: 执行量化分析流程...", expanded=True) as status:
            agent_b = AgentOrchestrator()
            quant_res, images = agent_b.run(stock_symbol, f"分析 {stock_symbol} 的趋势、风险与统计特征", st)
            status.update(label="✅ Agent B: 量化分析结束", state="complete", expanded=False)
            
        if not images:
            st.error("❌ 分析过程中未能生成有效图表，无法继续生成报告。")
            return

        # --- 阶段 3: 撰写报告 ---
        with st.status("✍️ [阶段 3/3] Agent E: 正在撰写深度研报...", expanded=True) as status:
            agent_e = AgentCIO()
            report_md = agent_e.run(news, quant_res, images, target_name)
            status.update(label="✅ Agent E: 研报撰写完成", state="complete", expanded=False)

        # --- 最终展示 ---
        st.divider()
        st.header(f"📑 {target_name} 深度投资研报")
        
        # 使用 Tabs 分开展示报告文本和图表画廊
        tab_report, tab_gallery = st.tabs(["📄 分析报告", "🖼️ 图表画廊"])
        
        with tab_report:
            st.markdown(report_md)
            
        with tab_gallery:
            st.info("以下是本次分析生成的关键图表：")
            cols = st.columns(2)
            for i, img_path in enumerate(images):
                with cols[i % 2]:
                    # 确保路径存在
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                    else:
                        st.warning(f"图片丢失: {img_path}")

if __name__ == "__main__":
    main()