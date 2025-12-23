import pandas as pd
import numpy as np
import akshare as ak
import os
from datetime import datetime, timedelta

# 定义输出目录
OUTPUT_DIR = "./output"

class DataProcessor:
    """
    组员 A：基础数据处理类 (AkShare 实战版)
    """

    @staticmethod
    def normalize_akshare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        [关键适配器] 将 AkShare 的美股数据格式标准化。
        AkShare 美股接口返回的列名通常是小写 (date, open, close)，我们需要转为首字母大写。
        """
        df = df_raw.copy()
        
        # 1. 定义列名映射字典 (AkShare US -> Standard)
        # 即使 AkShare 未来变了，我们也只需要改这里
        rename_map = {
            'date': 'Date', 
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume',
            'adjusted_close': 'Adj Close' 
        }
        
        # 2. 重命名
        df.rename(columns=rename_map, inplace=True)
        
        # 3. 确保必须的列存在
        required = ['Date', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"数据异常：AkShare 返回的数据缺少 '{col}' 列。当前列名: {df.columns.tolist()}")

        # 4. 格式转换
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # 确保按时间正序 (旧->新)
        
        # 5. 确保是数值类型 (AkShare 有时返回字符串)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    @staticmethod
    def download_us_stock(symbol: str = "NVDA", days: int = 365) -> str:
        """
        [下载器] 获取美股数据，清洗后保存到本地。
        返回: 保存的 CSV 路径
        """
        print(f"[DataProcessor] 正在通过 AkShare 下载 {symbol} 数据...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        try:
            # 调用 AkShare 接口 (adjust="qfq" 代表前复权，适合做技术分析)
            # 注意：ak.stock_us_daily 可能会比较慢，请耐心等待
            df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
            
            # --- 数据清洗与标准化 ---
            df_clean = DataProcessor.normalize_akshare_data(df)
            
            # --- 时间切片 (只取最近 N 天) ---
            start_date = datetime.now() - timedelta(days=days)
            df_clean = df_clean[df_clean.index >= start_date]
            
            if df_clean.empty:
                raise ValueError(f"下载成功但数据为空 (可能是时间范围 {days} 天内无数据)")

            # 保存为标准 CSV
            file_path = f"{OUTPUT_DIR}/{symbol}_raw.csv"
            df_clean.to_csv(file_path)
            print(f"[DataProcessor] 下载并清洗完成: {file_path} (包含 {len(df_clean)} 行)")
            
            # --- [修复点] 返回标准字典 ---
            return {
                "status": "success",
                "summary": f"数据下载成功。已保存至 {file_path}，包含 {len(df_clean)} 行数据。",
                "processed_path": file_path, # 关键：把路径传回去
                "images": []
            }
            
        except Exception as e:
            print(f"[Error] 下载失败: {e}")
            # --- [修复点] 返回标准错误字典 ---
            return {"status": "error", "error": str(e)}
            
        except Exception as e:
            print(f"[Error] 下载失败: {e}")
            return None

    @staticmethod
    def add_technical_features(df_path: str):
        """
        [特征工程] 读取清洗好的 CSV，计算指标。
        """
        try:
            if not df_path or not os.path.exists(df_path):
                return {"status": "error", "error": "文件路径无效"}

            # 读取数据 (因为之前已经 normalize 过了，这里读出来就是标准的)
            df = pd.read_csv(df_path, index_col='Date', parse_dates=True)
            
            # --- 计算指标 ---
            
            # 1. MA
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()

            # 2. RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 3. MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # 4. 布林带
            std20 = df['Close'].rolling(window=20).std()
            df['Boll_Upper'] = df['MA20'] + 2 * std20
            df['Boll_Lower'] = df['MA20'] - 2 * std20
            df['Boll_Width'] = (df['Boll_Upper'] - df['Boll_Lower']) / df['MA20']

            # --- 保存结果 ---
            df.dropna(inplace=True) # 去除计算产生的空值
            
            # 构造输出文件名 (例如 NVDA_raw.csv -> NVDA_processed.csv)
            base_name = os.path.basename(df_path).replace("_raw.csv", "")
            new_path = f"{OUTPUT_DIR}/{base_name}_processed.csv"
            df.to_csv(new_path)
            
            return {
                "status": "success",
                "summary": f"特征工程完成。计算了 RSI(最新:{df['RSI'].iloc[-1]:.2f}), MACD 等指标。",
                "processed_path": new_path,
                "images": []
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy import stats

# 【重要】防止弹窗
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "./output"

class RiskEvaluator:
    """Risk Evaluator Class - Statistical and Risk Analysis"""
    
    @staticmethod
    def run_monte_carlo(df_path: str, n_simulations=1000, days=30):
        """
        Monte Carlo Simulation - Simulate future price paths
        
        Args:
            df_path: CSV file path
            n_simulations: Number of simulations, default 1000
            days: Forecast days, default 30
            
        Returns:
            dict: Dictionary containing status, summary and image paths
        """
        try:
            # 1. Load data
            df = pd.read_csv(df_path, parse_dates=['Date'], index_col='Date')
            df.sort_index(inplace=True)
            
            # 2. Core logic - Monte Carlo simulation
            # Calculate log returns
            df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()
            
            if len(df) < 30:
                return {
                    "status": "error",
                    "error": f"Insufficient data, at least 30 trading days required"
                }
            
            current_price = df['Close'].iloc[-1]
            returns = df['Returns'].values
            
            # Calculate statistical parameters
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Execute simulation
            simulations = np.zeros((n_simulations, days))
            np.random.seed(42)  # Fixed random seed
            
            for i in range(n_simulations):
                random_returns = np.random.normal(mean_return, std_return, days)
                price_path = current_price * np.exp(np.cumsum(random_returns))
                simulations[i] = price_path
            
            # Calculate risk metrics
            final_prices = simulations[:, -1]
            
            # VaR calculation (95% confidence)
            confidence_level = 0.95
            var_95 = current_price - np.percentile(final_prices, 100 * (1 - confidence_level))
            var_percentage = (var_95 / current_price) * 100
            
            # Confidence interval
            ci_lower = np.percentile(final_prices, 2.5)
            ci_upper = np.percentile(final_prices, 97.5)
            
            # 3. Plotting (save to OUTPUT_DIR)
            plt.figure(figsize=(12, 8))
            
            # Plot simulation paths (only some to avoid overcrowding)
            for i in range(min(100, n_simulations)):
                plt.plot(range(days), simulations[i], 
                        color='blue', alpha=0.05, linewidth=0.5)
            
            # Plot mean path
            mean_path = np.mean(simulations, axis=0)
            plt.plot(range(days), mean_path, 
                    color='red', linewidth=2, label='Mean Path')
            
            # Plot confidence interval
            lower_bound = np.percentile(simulations, 2.5, axis=0)
            upper_bound = np.percentile(simulations, 97.5, axis=0)
            plt.fill_between(range(days), lower_bound, upper_bound, 
                            color='red', alpha=0.2, label='95% Confidence Interval')
            
            # Current price line
            plt.axhline(y=current_price, color='green', linestyle='--', 
                       linewidth=2, label=f'Current Price: ${current_price:.2f}')
            
            # Chart settings
            plt.title(f'Monte Carlo Simulation - {n_simulations} {days}-Day Price Paths', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Trading Days')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add VaR annotation
            plt.text(days*0.7, current_price*0.9, 
                    f'VaR(95%) = -${var_95:.2f}\\n({var_percentage:.2f}%)',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
            
            # Ensure output directory exists
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"monte_carlo_{timestamp}.png"
            save_path = f"{OUTPUT_DIR}/{img_name}"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # Must close
            
            # 4. Return standard dictionary
            return {
                "status": "success",
                "summary": (
                    f"Monte Carlo simulation completed. Based on {len(df)} trading days of historical data, "
                    f"simulated {n_simulations} future {days}-day price paths.\\n"
                    f"Risk Analysis Results:\\n"
                    f"• VaR(95%): In the next {days} days, there's a 95% probability that maximum loss won't exceed ${var_95:.2f} ({var_percentage:.2f}% of current price)\\n"
                    f"• 95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]\\n"
                    f"• Historical Volatility: Daily return standard deviation {std_return:.4%}\\n"
                    f"• Current Price: ${current_price:.2f}"
                ),
                "images": [save_path]
            }
            
        except FileNotFoundError as e:
            return {
                "status": "error",
                "error": f"File not found: {str(e)}"
            }
        except KeyError as e:
            return {
                "status": "error",
                "error": f"Data column missing: {str(e)}. Please ensure data contains 'Date' and 'Close' columns"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Monte Carlo simulation failed: {str(e)}"
            }
    
    @staticmethod
    def run_distribution_test(df_path: str):
        """
        Return Distribution Test - Analyze return distribution characteristics
        
        Args:
            df_path: CSV file path
            
        Returns:
            dict: Dictionary containing status, summary and image paths
        """
        try:
            # 1. Load data
            df = pd.read_csv(df_path, parse_dates=['Date'], index_col='Date')
            df.sort_index(inplace=True)
            
            # 2. Core logic - Distribution test
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df = df.dropna()
            
            if len(df) < 30:
                return {
                    "status": "error",
                    "error": f"Insufficient data, at least 30 trading days required"
                }
            
            returns = df['Returns'].values * 100  # Convert to percentage
            
            # Calculate statistical indicators
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Normality test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            # 3. Plotting (save to OUTPUT_DIR)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Histogram vs Normal Distribution
            n_bins = min(50, int(np.sqrt(len(returns))))
            ax1.hist(returns, bins=n_bins, density=True, 
                    alpha=0.7, color='blue', edgecolor='black', 
                    label='Actual Distribution')
            
            # Normal distribution curve
            x = np.linspace(returns.min(), returns.max(), 1000)
            normal_pdf = stats.norm.pdf(x, loc=mean_return, scale=std_return)
            ax1.plot(x, normal_pdf, 'r-', linewidth=2, 
                    label=f'Normal Distribution\\nμ={mean_return:.2f}%, σ={std_return:.2f}%')
            
            ax1.set_title('Return Distribution vs Normal Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Daily Returns (%)')
            ax1.set_ylabel('Probability Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Q-Q Plot
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Theoretical Quantiles')
            ax2.set_ylabel('Sample Quantiles')
            ax2.grid(True, alpha=0.3)
            
            # Add JB test results on Q-Q plot
            ax2.text(0.05, 0.95, f'JB Statistic: {jb_stat:.2f}\\np-value: {jb_pvalue:.4f}',
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 4. Save image
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"distribution_test_{timestamp}.png"
            save_path = f"{OUTPUT_DIR}/{img_name}"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # Must close
            
            # 5. Analyze distribution characteristics
            skew_analysis = "Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Approximately symmetric"
            kurtosis_analysis = "Leptokurtic (fat-tailed)" if kurtosis > 1 else "Platykurtic (thin-tailed)" if kurtosis < -1 else "Mesokurtic (normal-like)"
            normality_test = "Does not follow normal distribution" if jb_pvalue < 0.05 else "Approximately follows normal distribution"
            
            # 6. Return standard dictionary
            return {
                "status": "success",
                "summary": (
                    f"Return distribution test completed. Based on {len(df)} trading days of historical return data.\\n"
                    f"Key Findings:\\n"
                    f"1. Basic Statistics:\\n"
                    f"   • Mean Daily Return: {mean_return:.4f}%\\n"
                    f"   • Volatility (Std Dev): {std_return:.4f}%\\n"
                    f"2. Distribution Characteristics:\\n"
                    f"   • Skewness: {skewness:.4f} ({skew_analysis})\\n"
                    f"   • Kurtosis: {kurtosis:.4f} ({kurtosis_analysis})\\n"
                    f"3. Normality Test:\\n"
                    f"   • Jarque-Bera Statistic: {jb_stat:.2f} (p-value: {jb_pvalue:.4f})\\n"
                    f"   • Conclusion: {normality_test}\\n"
                    f"\\nAnalysis shows: The asset return distribution is {skew_analysis.lower()}, "
                    f"exhibits {kurtosis_analysis.lower()} characteristics, and {normality_test.lower()}."
                ),
                "images": [save_path]
            }
            
        except FileNotFoundError as e:
            return {
                "status": "error",
                "error": f"File not found: {str(e)}"
            }
        except KeyError as e:
            return {
                "status": "error",
                "error": f"Data column missing: {str(e)}. Please ensure data contains 'Date' and 'Close' columns"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Distribution test failed: {str(e)}"
            }


# tools.py  (Member B - Professional Version)
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _ts_path(prefix: str, ext: str = "png") -> str:
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{int(time.time() * 1000)}.{ext}")
    return p.replace("\\", "/")  # 统一路径分隔符，利于LaTeX和跨平台


def _load_processed(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    # 优先识别 Date 列
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    else:
        # 兼容 index 本身就是日期的情况
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df


def _safe_clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _make_binary_label(df: pd.DataFrame, look_ahead: int, ret_threshold: float = 0.0) -> pd.DataFrame:
    """
    生成二分类标签：
    future_return = Close(t+look_ahead)/Close(t) - 1
    label = 1 if future_return > ret_threshold else 0
    若 ret_threshold > 0，意味着过滤掉一部分小波动（可选：也可以剔除中间段，但二分类通常不剔除）
    """
    out = df.copy()
    out["FUTURE_CLOSE"] = out["Close"].shift(-look_ahead)
    out["FUTURE_RET"] = out["FUTURE_CLOSE"] / out["Close"] - 1.0
    out["TARGET"] = (out["FUTURE_RET"] > ret_threshold).astype(int)
    return out


def _plot_feature_importance(feature_cols: List[str], importances: np.ndarray) -> str:
    order = np.argsort(importances)[::-1]
    topk = min(10, len(feature_cols))
    idx = order[:topk]

    plt.figure(figsize=(10, 5))
    plt.bar(range(topk), importances[idx])
    plt.xticks(range(topk), [feature_cols[i] for i in idx], rotation=30, ha="right")
    plt.title("RandomForest Feature Importance (Top)")
    plt.tight_layout()

    img_path = _ts_path("rf_feature_importance")
    plt.savefig(img_path, dpi=160)
    plt.close()
    return img_path


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float)

    # 行归一化（按真实类别归一化，便于看召回）
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(cm_pct, vmin=0, vmax=1)  # 用百分比做色阶更统一
    plt.title("Confusion Matrix (Row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    labels = ["Down/0", "Up/1"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    # 标注：count + percent
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = cm_pct[i, j]
            # 颜色对比：深色背景用白字
            text_color = "white" if pct > 0.5 else "black"
            plt.text(
                j, i,
                f"{count}\n{pct:.1%}",
                ha="center", va="center",
                color=text_color,
                fontsize=11
            )

    # 细网格线（看起来更像表）
    plt.gca().set_xticks(np.arange(-.5, 2, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, 2, 1), minor=True)
    plt.grid(which="minor", linestyle="-", linewidth=1)
    plt.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    img_path = _ts_path("rf_confusion_matrix")
    plt.savefig(img_path, dpi=180)
    plt.close()
    return img_path



from sklearn.metrics import roc_curve, auc

def _plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """
    更稳定的ROC绘图：
    - 若 y_test 只有一个类别：输出概率直方图替代（ROC不可定义）
    - 否则：用 roc_curve(drop_intermediate=False) + step plot，避免“倒三角”观感
    """
    y_test = np.asarray(y_test).astype(int)
    unique = np.unique(y_test)

    # 单类：ROC 不存在，改画概率直方图
    if len(unique) < 2:
        prob = model.predict_proba(X_test)[:, 1]
        plt.figure(figsize=(6, 5))
        plt.hist(prob, bins=25, alpha=0.85)
        plt.title("Probability Histogram (ROC unavailable: single-class test set)")
        plt.xlabel("P(Up)")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        img_path = _ts_path("rf_prob_hist")
        plt.savefig(img_path, dpi=160)
        plt.close()
        return img_path

    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.step(fpr, tpr, where="post", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    img_path = _ts_path("rf_roc_curve")
    plt.savefig(img_path, dpi=160)
    plt.close()
    return img_path



class PricePredictor:
    @staticmethod
    def run_rf_prediction(
        df_path: str,
        look_ahead: int = 1,
        ret_threshold: float = 0.0,
        n_splits: int = 5
    ) -> Dict:
        """
        - TimeSeriesSplit 做回测统计
        - 图：Feature Importance + Confusion Matrix + ROC/概率直方图
        - 返回严格：status/summary/images
        """
        try:
            df = _safe_clean_numeric(_load_processed(df_path))

            if "Close" not in df.columns:
                raise ValueError("processed 数据缺少关键列: Close")

            candidate_features = [
                "MA5", "MA20", "RSI",
                "MACD", "MACD_Signal", "MACD_Hist",
                "Boll_Width", "Volume"
            ]
            feature_cols = [c for c in candidate_features if c in df.columns]
            if len(feature_cols) < 4:
                raise ValueError(f"可用特征列太少：{feature_cols}。请确认 processed.csv 是否包含技术指标列。")

            df_labeled = _make_binary_label(df, look_ahead=look_ahead, ret_threshold=ret_threshold)
            data = df_labeled[feature_cols + ["TARGET"]].dropna()

            # 样本数下限：确保至少能做 2 折切分 + 一点测试集
            if len(data) < 120:
                raise ValueError(f"有效样本不足（{len(data)}），建议提供更长历史窗口或降低 look_ahead/阈值。")

            X = data[feature_cols].values
            y = data["TARGET"].values.astype(int)

            # Baseline：多数类准确率
            p_up = float(np.mean(y))
            baseline = max(p_up, 1.0 - p_up)

            # 动态折数：至少2折，且不超过 n_splits
            max_splits_by_data = max(2, len(data) // 60)
            splits = min(n_splits, max_splits_by_data)
            if splits < 2:
                splits = 2  # 保险
            tscv = TimeSeriesSplit(n_splits=splits)

            cv_metrics = {"acc": [], "bacc": [], "f1": [], "auc": []}

            last_train_idx, last_test_idx = None, None

            # CV 回测
            for train_idx, test_idx in tscv.split(X):
                last_train_idx, last_test_idx = train_idx, test_idx

                model = RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample"
                )
                model.fit(X[train_idx], y[train_idx])

                pred = model.predict(X[test_idx])
                prob = model.predict_proba(X[test_idx])[:, 1]

                cv_metrics["acc"].append(accuracy_score(y[test_idx], pred))
                cv_metrics["bacc"].append(balanced_accuracy_score(y[test_idx], pred))
                cv_metrics["f1"].append(f1_score(y[test_idx], pred, zero_division=0))
                try:
                    cv_metrics["auc"].append(roc_auc_score(y[test_idx], prob))
                except Exception:
                    cv_metrics["auc"].append(np.nan)

            if last_train_idx is None or last_test_idx is None:
                raise ValueError("TimeSeriesSplit 未能生成有效切分，请检查数据长度。")

            # 用最后一折训练得到“展示模型”
            X_train, X_test = X[last_train_idx], X[last_test_idx]
            y_train, y_test = y[last_train_idx], y[last_test_idx]

            final_model = RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
            final_model.fit(X_train, y_train)
            pred = final_model.predict(X_test)
            prob = final_model.predict_proba(X_test)[:, 1]

            # 最后一折点估计
            acc = accuracy_score(y_test, pred)
            bacc = balanced_accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)
            try:
                auc_val = roc_auc_score(y_test, prob)
            except Exception:
                auc_val = float("nan")

            # 图：重要性 + 混淆矩阵 + ROC(安全版/step版)
            img1 = _plot_feature_importance(feature_cols, final_model.feature_importances_)
            img2 = _plot_confusion_matrix(y_test, pred)
            img3 = _plot_roc_curve(final_model, X_test, y_test)

            # Top 特征
            importances = final_model.feature_importances_
            order = np.argsort(importances)[::-1]
            top5 = ", ".join([f"{feature_cols[i]}({importances[i]:.3f})" for i in order[:5]])

            # CV 均值±std
            def _mean_std(arr):
                arr = np.array(arr, dtype=float)
                return float(np.nanmean(arr)), float(np.nanstd(arr))

            acc_m, acc_s = _mean_std(cv_metrics["acc"])
            bacc_m, bacc_s = _mean_std(cv_metrics["bacc"])
            f1_m, f1_s = _mean_std(cv_metrics["f1"])
            auc_m, auc_s = _mean_std(cv_metrics["auc"])

            summary = (
                f"随机森林方向预测（look_ahead={look_ahead}, ret_threshold={ret_threshold:.4f}）完成。"
                f"样本上涨比例={p_up:.2%}，多数类基线Accuracy={baseline:.2%}。"
                f"最后一折：Accuracy={acc:.2%}, BalancedAcc={bacc:.2%}, F1={f1:.3f}, ROC-AUC={auc_val:.3f}。"
                f"时间序列CV(均值±标准差)：Acc={acc_m:.2%}±{acc_s:.2%}, "
                f"BAcc={bacc_m:.2%}±{bacc_s:.2%}, F1={f1_m:.3f}±{f1_s:.3f}, AUC={auc_m:.3f}±{auc_s:.3f}。"
                f"最重要特征Top5：{top5}。"
            )

            return {"status": "success", "summary": summary, "images": [img1, img2, img3]}

        except Exception as e:
            return {"status": "error", "error": f"run_rf_prediction failed: {str(e)}"}


    @staticmethod
    def run_regression(df_path: str) -> Dict:
        """
        趋势回归（研报常用）：
        - LinearRegression 对 Close 做趋势拟合
        - 输出 slope + R² + 图
        """
        try:
            df = _safe_clean_numeric(_load_processed(df_path))
            if "Close" not in df.columns:
                raise ValueError("processed 数据缺少 Close 列，无法回归。")

            data = df[["Close"]].dropna()
            if len(data) < 60:
                raise ValueError(f"有效样本太少（{len(data)}），建议至少60条。")

            x = np.arange(len(data)).reshape(-1, 1)
            y = data["Close"].values

            model = LinearRegression()
            model.fit(x, y)
            y_hat = model.predict(x)

            r2 = r2_score(y, y_hat)
            slope = float(model.coef_[0])

            # 画图：散点 + 拟合线
            plt.figure(figsize=(10, 5))
            plt.scatter(data.index, y, s=10, alpha=0.6)
            plt.plot(data.index, y_hat, linewidth=2)
            plt.title("Close Price Trend (Linear Regression)")
            plt.xticks(rotation=30)
            plt.tight_layout()

            img = _ts_path("price_regression")
            plt.savefig(img, dpi=160)
            plt.close()

            trend = "上升" if slope > 0 else "下降" if slope < 0 else "横盘"
            summary = f"回归拟合完成：趋势为「{trend}」（slope={slope:.6f}），R²={r2:.3f}。已生成趋势拟合图。"

            return {"status": "success", "summary": summary, "images": [img]}

        except Exception as e:
            return {"status": "error", "error": f"run_regression failed: {str(e)}"}


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')

# 【重要】防止弹窗
import matplotlib

matplotlib.use('Agg')

OUTPUT_DIR = "./output"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 【修改点 1】更改字体设置
# 移除 SimHei，使用标准字体以支持英文显示
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class MarketRegime:
    @staticmethod
    def run_kmeans_regime(df_path: str, n_clusters=3):
        """
        执行K-Means市场状态聚类分析
        """
        try:
            # 1. 加载数据
            df = pd.read_csv(df_path, index_col=0, parse_dates=True)

            # 2. 计算收益率和波动率
            df['Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)

            # 移除NaN值
            features = df[['Return', 'Volatility']].dropna()

            # 3. 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # 4. K-Means聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # 5. 分析结果
            df_features = features.copy()
            df_features['Cluster'] = labels
            df_features['Cluster'] = df_features['Cluster'].astype(int)

            # 计算每个聚类的统计信息
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = df_features[df_features['Cluster'] == i]
                cluster_stats[i] = {
                    'mean_return': cluster_data['Return'].mean() * 100,
                    'mean_volatility': cluster_data['Volatility'].mean(),
                    'days_count': len(cluster_data),
                    'frequency': len(cluster_data) / len(df_features) * 100
                }

            # 6. 确定当前状态（最近一天）
            current_return = df_features['Return'].iloc[-1] * 100
            current_volatility = df_features['Volatility'].iloc[-1]
            current_cluster = df_features['Cluster'].iloc[-1]

            # 【修改点 2】将聚类命名逻辑改为英文，以便图例显示为英文
            cluster_names = {}
            for i, stats in cluster_stats.items():
                return_val = stats['mean_return']
                vol_val = stats['mean_volatility']

                if vol_val > 2.5:
                    if return_val < -1:
                        cluster_names[i] = "High Vol Bear"  # 高波动大跌
                    elif return_val > 2:
                        cluster_names[i] = "High Vol Bull"  # 高波动大涨
                    else:
                        cluster_names[i] = "High Vol Chop"  # 高波动震荡
                else:
                    if return_val < -0.5:
                        cluster_names[i] = "Low Vol Bear"   # 低波动下跌
                    elif return_val > 0.5:
                        cluster_names[i] = "Low Vol Bull"   # 低波动上涨
                    else:
                        cluster_names[i] = "Neutral/Calm"   # 中波动温和

            current_state_name = cluster_names.get(current_cluster, "Unknown")

            # 7. 绘制散点图
            plt.figure(figsize=(12, 8))

            colors = ['red', 'green', 'blue', 'orange', 'purple'][:n_clusters]
            for i in range(n_clusters):
                cluster_data = df_features[df_features['Cluster'] == i]
                # 【修改点 3】Label改为英文格式
                plt.scatter(cluster_data['Volatility'], cluster_data['Return'] * 100,
                            c=colors[i], alpha=0.6, s=50,
                            label=f'Regime {i}: {cluster_names.get(i, "Unknown")}')

            # 标记当前点
            plt.scatter(current_volatility, current_return,
                        c='black', s=200, marker='*', edgecolors='yellow',
                        label=f'Current ({current_state_name})')

            # 【修改点 4】坐标轴和标题改为英文
            plt.xlabel('Volatility (Annualized)', fontsize=12)
            plt.ylabel('Return (%)', fontsize=12)
            plt.title('Market Regime Analysis (K-Means)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"kmeans_regime_{timestamp}.png"
            save_path = f"{OUTPUT_DIR}/{img_name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 8. 生成分析摘要 (保持中文便于阅读，但状态名会显示为英文)
            analysis_period = f"{df.index[0].date()} 至 {df.index[-1].date()}"

            summary = f"""
K-Means市场状态聚类分析报告
==================================================
分析周期: {analysis_period}
聚类数量: {n_clusters}
当前状态: 【{current_state_name}】 ({df.index[-1].date()})
当前价格: ${df['Close'].iloc[-1]:.2f}
当前收益率: {current_return:.2f}%
当前波动率: {current_volatility:.4f}

📊 聚类详情:
"""
            for i in range(n_clusters):
                stats = cluster_stats[i]
                summary += f"状态{i} ({cluster_names.get(i, '未知')}):\n"
                summary += f"  • 平均收益率: {stats['mean_return']:.2f}%\n"
                summary += f"  • 平均波动率: {stats['mean_volatility']:.4f}\n"
                summary += f"  • 持续天数: {stats['days_count']}天\n"
                summary += f"  • 出现频率: {stats['frequency']:.1f}%\n\n"

            market_mean_return = df_features['Return'].mean() * 100
            market_std_return = df_features['Return'].std() * 100
            market_mean_vol = df_features['Volatility'].mean()
            market_std_vol = df_features['Volatility'].std()

            summary += f"""
📈 市场特征统计:
• 平均日收益率: {market_mean_return:.3f}%
• 收益率标准差: {market_std_return:.3f}%
• 平均波动率: {market_mean_vol:.4f}
• 波动率标准差: {market_std_vol:.4f}

"""

            return {
                "status": "success",
                "summary": summary,
                "images": [save_path],
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"K-Means聚类分析失败: {str(e)}"
            }


class TimeSeriesMiner:
    @staticmethod
    def run_seasonal_decomposition(df_path: str, period=20):
        """
        执行时间序列分解分析
        """
        try:
            # 1. 加载数据
            df = pd.read_csv(df_path, index_col=0, parse_dates=True)

            # 2. 时间序列分解
            close_prices = df['Close'].dropna()

            # 使用加法模型
            decomposition = seasonal_decompose(close_prices, model='additive', period=period)

            # 3. 计算统计信息
            trend = decomposition.trend.dropna()
            recent_trend = trend.iloc[-min(10, len(trend)):]
            slope = np.polyfit(range(len(recent_trend)), recent_trend.values, 1)[0]

            seasonal = decomposition.seasonal.dropna()

            resid = decomposition.resid.dropna()
            resid_std = resid.std()
            recent_resid_std = resid.iloc[-min(30, len(resid)):].std()

            # 4. 计算贡献度
            total_variation = np.var(close_prices)
            trend_contrib = np.var(trend) / total_variation * 100 if total_variation > 0 else 0
            seasonal_contrib = np.var(seasonal) / total_variation * 100 if total_variation > 0 else 0
            resid_contrib = np.var(resid) / total_variation * 100 if total_variation > 0 else 0

            # 5. 生成趋势判断
            if abs(slope) < 0.1:
                trend_strength = "微弱"
                recent_trend_direction = "震荡"
            elif abs(slope) < 0.5:
                trend_strength = "温和"
                recent_trend_direction = "下降" if slope < 0 else "上升"
            else:
                trend_strength = "强烈"
                recent_trend_direction = "下降" if slope < 0 else "上升"

            # 6. 绘制图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 图表1：分解图
            fig1 = plt.figure(figsize=(15, 10))

            # 【修改点 5】分解图的所有中文标签改为英文
            # 子图1：原始序列
            ax1 = plt.subplot(411)
            ax1.plot(close_prices.index, close_prices, 'b-', linewidth=1.5)
            ax1.set_ylabel('Price', fontsize=10) # 价格
            ax1.set_title('Time Series Decomposition Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # 子图2：趋势
            ax2 = plt.subplot(412)
            ax2.plot(trend.index, trend, 'g-', linewidth=2)
            ax2.set_ylabel('Trend', fontsize=10) # 趋势
            ax2.grid(True, alpha=0.3)

            # 子图3：周期
            ax3 = plt.subplot(413)
            ax3.plot(seasonal.index, seasonal, 'r-', linewidth=1)
            ax3.set_ylabel('Seasonal', fontsize=10) # 周期
            ax3.grid(True, alpha=0.3)

            # 子图4：残差
            ax4 = plt.subplot(414)
            ax4.plot(resid.index, resid, 'k-', linewidth=0.8, alpha=0.7)
            ax4.axhline(y=2 * resid_std, color='r', linestyle='--', alpha=0.5, label='2σ')
            ax4.axhline(y=-2 * resid_std, color='r', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Residual', fontsize=10) # 残差
            ax4.set_xlabel('Date', fontsize=10)     # 日期
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            plt.tight_layout()

            # 保存分解图
            img1_name = f"seasonal_decomposition_{timestamp}.png"
            save_path1 = f"{OUTPUT_DIR}/{img1_name}"
            plt.savefig(save_path1, dpi=300, bbox_inches='tight')
            plt.close()

            # 图表2：贡献度饼图
            plt.figure(figsize=(10, 8))
            contributions = [trend_contrib, seasonal_contrib, resid_contrib]
            
            # 【修改点 6】饼图标签改为英文
            labels = ['Trend', 'Seasonal', 'Residual'] 
            colors = ['#4CAF50', '#2196F3', '#FF9800']

            plt.pie(contributions, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            plt.title('Component Contribution Analysis', fontsize=14, fontweight='bold')

            img2_name = f"decomposition_contrib_{timestamp}.png"
            save_path2 = f"{OUTPUT_DIR}/{img2_name}"
            plt.savefig(save_path2, dpi=300, bbox_inches='tight')
            plt.close()

            # 7. 生成分析摘要 (保持中文)
            analysis_period = f"{close_prices.index[0].date()} 至 {close_prices.index[-1].date()}"

            summary = f"""
时间序列分解分析报告
==================================================
分析周期: {analysis_period}
分解周期: {period} 天
模型类型: 加法模型 (additive)

📈 趋势分析:
• 趋势强度: {trend_strength} (贡献度: {trend_contrib:.1f}%)
• 近期趋势: {recent_trend_direction} (斜率: {slope:.4f})
• 当前趋势值: ${trend.iloc[-1]:.2f}

🔄 周期分析:
• 周期特征: {'无明显周期性' if seasonal_contrib < 5 else '有明显周期性'} (贡献度: {seasonal_contrib:.1f}%)
• 周期振幅: {seasonal.max() - seasonal.min():.2f} (范围: {seasonal.min():.2f} 到 {seasonal.max():.2f})
• 当前周期效应: {seasonal.iloc[-1]:.2f}

📊 残差分析:
• 波动特征: {'残差波动稳定' if recent_resid_std < resid_std * 1.2 else '残差波动增加'}
• 残差标准差: {resid_std:.2f}
• 近期标准差: {recent_resid_std:.2f}
• 异常点数量: {len(resid[abs(resid) > 2 * resid_std])}个 (> ±{2 * resid_std:.2f})

📋 分解贡献总结:
• 趋势分量贡献: {trend_contrib:.1f}%
• 周期分量贡献: {seasonal_contrib:.1f}%
• 残差分量贡献: {resid_contrib:.1f}%
• 主要驱动因素: {['残差', '周期', '趋势'][np.argmax([resid_contrib, seasonal_contrib, trend_contrib])]}

"""

            return {
                "status": "success",
                "summary": summary,
                "images": [save_path1, save_path2],
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"时间序列分解失败: {str(e)}"
            }
