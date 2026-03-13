# ==========================================
# 工具函数：数据处理、CSS 加载等
# ==========================================

import pandas as pd
from pathlib import Path
from .config import EMOTION_NAMES, EMOTION_STYLES, COARSE_MAP


def load_css(css_path: str) -> str:
    """
    加载 CSS 文件内容
    
    Args:
        css_path: CSS 文件路径
        
    Returns:
        CSS 内容字符串
    """
    path = Path(css_path)
    if path.exists():
        return path.read_text(encoding='utf-8')
    return ""


def load_csv_with_encoding(file_path, encodings=None):
    """
    尝试多种编码读取 CSV 文件
    
    Args:
        file_path: 文件路径或文件对象
        encodings: 编码列表，默认使用常见中文编码
        
    Returns:
        DataFrame 或 None
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "big5", "cp936", "latin1"]
    
    last_error = None
    
    for enc in encodings:
        try:
            if hasattr(file_path, 'seek'):
                # 文件对象需要重置指针
                file_path.seek(0)
                df = pd.read_csv(file_path, encoding=enc)
            else:
                df = pd.read_csv(file_path, encoding=enc)
            return df
        except Exception as e:
            last_error = e
            continue
    
    return None, last_error


def validate_dataframe(df, required_columns=None):
    """
    验证 DataFrame 是否包含必需的列
    
    Args:
        df: DataFrame
        required_columns: 必需列名列表，默认为 ['text']
        
    Returns:
        (bool, str): 是否有效和错误信息
    """
    if required_columns is None:
        required_columns = ['text']
    
    if df is None:
        return False, "DataFrame 为空"
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"缺少必需的列：{', '.join(missing_cols)}"
    
    return True, ""


def format_batch_results(results: list) -> pd.DataFrame:
    """
    格式化批量分析结果
    
    Args:
        results: 预测结果列表
        
    Returns:
        DataFrame
    """
    formatted = []
    for r in results:
        row = {
            "text": r.get("text", ""),
            "fine_emotion": r["fine_name"],
            "coarse_emotion": r["coarse"],
            "confidence": r["confidence"],
            **{f"prob_{k}": v for k, v in r["all_probabilities"].items()},
        }
        formatted.append(row)
    
    return pd.DataFrame(formatted)


def get_example_texts() -> list:
    """
    获取示例文本列表
    
    Returns:
        示例文本列表
    """
    return [
        ("开心示例", "今天终于拿到了心仪的 offer，感觉所有的努力都值得了！"),
        ("悲伤示例", "听到这个坏消息，我心里非常难过。"),
        ("惊讶示例", "居然中了大奖，太意外了！"),
    ]


def emotion_id_by_name(emotion_name: str) -> int:
    """
    根据情感名称获取情感 ID
    
    Args:
        emotion_name: 情感名称
        
    Returns:
        情感 ID，找不到返回 0
    """
    for k, v in EMOTION_NAMES.items():
        if v == emotion_name:
            return int(k)
    return 0


def get_coarse_badge_class(coarse: str) -> str:
    """
    获取粗粒度情感的 CSS 类名
    
    Args:
        coarse: 粗粒度情感标签（正面/负面/中性）
        
    Returns:
        CSS 类名
    """
    if coarse == "正面":
        return "positive"
    if coarse == "负面":
        return "negative"
    return "neutral"