# ==========================================
# 文本处理工具：句子分割等
# ==========================================

import re


def split_chinese_sentences(text: str) -> list:
    """
    将中文文本分割成句子
    
    支持多种中文标点符号（句号、感叹号、问号等）进行句子分割
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
    """
    # 去除换行符，将文本合并为连续的一行
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    
    # 中文句子结束标点：。！？!? (包括中英文)
    sentence_end_pattern = r'([.。！？!?])'
    
    sentences = []
    
    # 使用正则表达式分割句子
    parts = re.split(sentence_end_pattern, text)
    
    # 重组句子和标点
    current_sentence = ""
    for i, part in enumerate(parts):
        if not part:
            continue
            
        # 如果是标点符号
        if re.match(sentence_end_pattern, part):
            current_sentence += part
            sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += part
    
    # 处理最后剩余的部分
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences
