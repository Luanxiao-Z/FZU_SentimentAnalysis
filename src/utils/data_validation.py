# ==========================================
# 数据验证和格式化工具
# ==========================================

import pandas as pd
from ..config import EMOTION_NAMES, EMOTION_STYLES, COARSE_MAP


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
            "fine_emotion": r.get("fine_name") or r.get("fine_emotion", ""),
            "coarse_emotion": r.get("coarse") or r.get("coarse_emotion", ""),
            "confidence": r.get("confidence", 0.0),
            **{f"prob_{k}": v for k, v in r.get("all_probabilities", {}).items()},
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


def validate_input_text(text: str) -> tuple[bool, str]:
    """
    验证输入文本是否有效
    
    Args:
        text: 输入文本
        
    Returns:
        (bool, str): 是否有效和错误信息
    """
    if not text:
        return False, "输入文本不能为空"
    if not isinstance(text, str):
        return False, "输入必须是字符串类型"
    if len(text.strip()) == 0:
        return False, "输入文本不能为空白字符"
    return True, ""
