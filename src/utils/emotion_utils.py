# ==========================================
# 情感标签工具函数
# ==========================================

from ..config import EMOTION_NAMES, EMOTION_STYLES, COARSE_MAP


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
