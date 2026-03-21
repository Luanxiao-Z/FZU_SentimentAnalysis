# -*- coding: utf-8 -*-
"""
文档处理器单元测试
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd

# 需要从 src 目录导入，以确保与项目结构一致
from src.utils.document_processor import process_document_to_sentences

def test_process_document_to_sentences_pdf():
    """测试处理PDF文档"""
    # 使用真实的文件路径进行测试
    from pathlib import Path
    real_file_path = Path("E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_documents/doc_sample.pdf")
    
    # 验证文件存在性
    if not real_file_path.exists():
        pytest.skip(f"跳过测试，因为真实文件不存在: {real_file_path}")

    result_df = process_document_to_sentences(real_file_path, 'pdf')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) > 0

def test_process_document_to_sentences_txt():
    """测试处理TXT文档"""
    # 直接使用真实文件进行测试
    from pathlib import Path
    real_file_path = Path("E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_documents/doc_sample.txt")
    
    if not real_file_path.exists():
        pytest.skip(f"跳过测试，因为真实文件不存在: {real_file_path}")

    result_df = process_document_to_sentences(real_file_path, 'txt')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) > 0

def test_process_document_to_sentences_docx():
    """测试处理DOCX文档"""
    # 使用真实的文件路径进行测试
    from pathlib import Path
    real_file_path = Path("E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_documents/doc_sample.docx")
    
    if not real_file_path.exists():
        pytest.skip(f"跳过测试，因为真实文件不存在: {real_file_path}")

    result_df = process_document_to_sentences(real_file_path, 'docx')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) > 0

def test_process_document_to_sentences_md():
    """测试处理Markdown文档"""
    # 使用真实的文件路径进行测试
    from pathlib import Path
    real_file_path = Path("E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_documents/doc_sample.md")
    
    if not real_file_path.exists():
        pytest.skip(f"跳过测试，因为真实文件不存在: {real_file_path}")

    result_df = process_document_to_sentences(real_file_path, 'md')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) > 0