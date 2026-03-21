# -*- coding: utf-8 -*-
"""
文档处理器单元测试
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd

# 需要从 src 目录导入，以确保与项目结构一致
from src.utils.document_processor import process_document_to_sentences

@patch('src.utils.file_io.extract_text_from_pdf')
def test_process_document_to_sentences_pdf(mock_extract_text):
    """测试处理PDF文档"""
    # 模拟 PDF 提取文本
    mock_extract_text.return_value = "这是第一句。这是第二句。"

    # 创建一个模拟的文件对象（例如 bytesIO）
    mock_file = b"fake pdf content"

    result_df = process_document_to_sentences(mock_file, 'pdf')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) == 2
    assert "这是第一句。" in result_df['text'].values
    assert "这是第二句。" in result_df['text'].values

@patch('src.utils.file_io.extract_text_from_txt')
def test_process_document_to_sentences_txt(mock_extract_text):
    """测试处理TXT文档"""
    # 模拟 TXT 提取文本
    mock_extract_text.return_value = "这是一段纯文本内容。没有标点。但有换行\n这里是下一行。"

    mock_file = b"fake txt content"
    result_df = process_document_to_sentences(mock_file, 'txt')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) > 0
    # 因为没有标点，所以可能只有一行或按换行分割
    assert any("这是一段纯文本内容。" in text for text in result_df['text'])

@patch('src.utils.file_io.extract_text_from_docx')
def test_process_document_to_sentences_docx(mock_extract_text):
    """测试处理DOCX文档"""
    # 模拟 DOCX 提取文本
    mock_extract_text.return_value = "这是一个Word文档。包含多个段落。"

    mock_file = b"fake docx content"
    result_df = process_document_to_sentences(mock_file, 'docx')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) >= 1
    assert "这是一个Word文档。" in result_df['text'].values or "包含多个段落。" in result_df['text'].values

@patch('src.utils.file_io.extract_text_from_md')
def test_process_document_to_sentences_md(mock_extract_text):
    """测试处理Markdown文档"""
    # 模拟 MD 提取文本
    mock_extract_text.return_value = "# 标题\n\n这是一篇关于AI的文章。它讨论了未来发展。"

    mock_file = b"fake md content"
    result_df = process_document_to_sentences(mock_file, 'md')

    # 断言
    assert isinstance(result_df, pd.DataFrame)
    assert 'text' in result_df.columns
    assert len(result_df) == 2
    assert "这是一篇关于AI的文章。" in result_df['text'].values
    assert "它讨论了未来发展。" in result_df['text'].values