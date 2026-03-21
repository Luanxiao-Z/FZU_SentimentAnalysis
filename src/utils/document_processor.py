# ==========================================
# 文档处理工具：将文档转换为句子 DataFrame
# ==========================================

import pandas as pd
from .file_io import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx,
    extract_text_from_md,
)
from .text_processing import split_chinese_sentences


def process_document_to_sentences(file_content, file_type: str) -> pd.DataFrame:
    """
    处理文档文件为句子 DataFrame
    
    Args:
        file_content: 文件内容（字节或文件对象）
        file_type: 文件类型 ('pdf', 'txt', 'docx', 'md')
        
    Returns:
        包含 'text' 列的 DataFrame
    """
    import tempfile
    import os
    
    # 创建临时文件
    suffix_map = {'pdf': '.pdf', 'txt': '.txt', 'docx': '.docx', 'md': '.md'}
    suffix = suffix_map.get(file_type, '')
    
    # 处理不同类型的输入
    if isinstance(file_content, bytes):
        # 字节数据：直接写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
    elif hasattr(file_content, 'seek'):
        # 文件对象：重置指针并读取内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_content.seek(0)
            tmp.write(file_content.read())
            tmp_path = tmp.name
    else:
        # Path对象或字符串路径：直接使用
        tmp_path = str(file_content)
    
    try:
        # 提取文本
        if file_type == 'pdf':
            text = extract_text_from_pdf(tmp_path)
        elif file_type == 'txt':
            text = extract_text_from_txt(tmp_path)
        elif file_type == 'docx':
            text = extract_text_from_docx(tmp_path)
        elif file_type == 'md':
            text = extract_text_from_md(tmp_path)
        else:
            raise ValueError(f"不支持的文件类型：{file_type}")
        
        # 划分句子
        sentences = split_chinese_sentences(text)
        
        # 创建 DataFrame
        df = pd.DataFrame({'text': sentences})
        
        return df
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass
