# -*- coding: utf-8 -*-
"""
ASR 处理器单元测试
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
from pathlib import Path

# 需要从 src 目录导入，以确保与项目结构一致
from src.utils.asr_processor import audio_to_text, AsrError, AsrSettings

@pytest.fixture
def mock_asr_dependencies():
        """为ASR测试提供统一的mock依赖项"""
        with patch('src.utils.asr_processor._load_settings') as mock_load_settings,\
             patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono') as mock_normalize,\
             patch('src.utils.asr_processor.segment_wav') as mock_segment,\
             patch('src.utils.asr_processor._baidu_get_access_token') as mock_get_token,\
             patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text') as mock_baidu_asr:
            # 模拟加载设置已经由mock_load_settings等对象在fixture中处理
            
            # 预先配置通用的模拟行为
            mock_normalize.return_value = 'mock_normalized_wav.wav'
            mock_get_token.return_value = 'fake_token'
            
            yield mock_baidu_asr, mock_get_token, mock_segment, mock_normalize, mock_load_settings

def test_audio_to_text_wav(mock_asr_dependencies):
    """测试 WAV 音频转文本"""
    mock_baidu_asr, mock_get_token, mock_segment, mock_normalize, mock_load_settings = mock_asr_dependencies
    # 模拟函数行为
    mock_segment.return_value = ['E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_audio/test_copy/test_normal.wav']
    mock_baidu_asr.return_value = '识别的文本内容'

    # 执行测试，传入一个虚拟路径
    result = audio_to_text("dummy_path.wav")

    # 断言
    assert isinstance(result, str)
    assert len(result) > 0
    # 验证关键函数是否被调用
    mock_normalize.assert_called_once()
    mock_segment.assert_called_once()
    mock_baidu_asr.assert_called_once()

def test_audio_to_text_mp3(mock_asr_dependencies):
    """测试 MP3 音频转文本"""
    mock_baidu_asr, mock_get_token, mock_segment, mock_normalize, mock_load_settings = mock_asr_dependencies
    # 模拟函数行为
    mock_normalize.return_value = 'mock_normalized_wav.wav'
    mock_segment.return_value = ['E:/智能系统综合设计/FZU_SentimentAnalysis/tests/fixtures/sample_audio/test_copy/test_normal.mp3']
    mock_get_token.return_value = 'fake_token'
    mock_baidu_asr.return_value = '识别的文本内容'

    # 执行测试，传入一个MP3路径
    result = audio_to_text("dummy_path.mp3")

    # 断言
    assert isinstance(result, str)
    assert len(result) > 0
    # 验证 normalize 函数被调用（表明MP3被处理）
    mock_normalize.assert_called_once()

def test_audio_to_text_missing_config_error(mock_asr_dependencies):
    """测试缺少配置时抛出异常"""
    mock_baidu_asr, mock_get_token, mock_segment, mock_normalize, mock_load_settings = mock_asr_dependencies
    # 模拟未找到配置文件
    mock_load_settings.side_effect = FileNotFoundError("Config not found")

    with pytest.raises(AsrError, match="请求百度 ASR 失败（配置文件缺失）"):
        audio_to_text("dummy_path.wav")