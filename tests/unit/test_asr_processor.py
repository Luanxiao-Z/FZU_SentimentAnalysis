# -*- coding: utf-8 -*-
"""
ASR 处理器单元测试
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
from pathlib import Path

# 需要从 src 目录导入，以确保与项目结构一致
from src.utils.asr_processor import audio_to_text, AsrError, AsrSettings

@patch('src.utils.asr_processor._load_settings')
@patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono')
@patch('src.utils.asr_processor.segment_wav')
@patch('src.utils.asr_processor._baidu_get_access_token')
@patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text')
def test_audio_to_text_wav(
    mock_baidu_asr, 
    mock_get_token, 
    mock_segment, 
    mock_normalize, 
    mock_load_settings,
):
    """测试 WAV 音频转文本"""
    # 模拟加载设置
    mock_settings = AsrSettings(
        provider="baidu",
        timeout=18.0,
        dev_pid=1537,
        chunk_seconds=45,
        target_sample_rate=16000,
        target_channels=1,
        baidu_api_key="fake_api_key",
        baidu_secret_key="fake_secret_key"
    )
    mock_load_settings.return_value = mock_settings

    # 模拟函数行为
    mock_normalize.return_value = 'mock_normalized_wav.wav'
    mock_segment.return_value = ['mock_segmented_chunk1.wav']
    mock_get_token.return_value = 'fake_token'
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

@patch('src.utils.asr_processor._load_settings')
@patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono')
@patch('src.utils.asr_processor.segment_wav')
@patch('src.utils.asr_processor._baidu_get_access_token')
@patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text')
def test_audio_to_text_mp3(
    mock_baidu_asr, 
    mock_get_token, 
    mock_segment, 
    mock_normalize, 
    mock_load_settings,
):
    """测试 MP3 音频转文本"""
    # 模拟加载设置
    mock_settings = AsrSettings(
        provider="baidu",
        timeout=18.0,
        dev_pid=1537,
        chunk_seconds=45,
        target_sample_rate=16000,
        target_channels=1,
        baidu_api_key="fake_api_key",
        baidu_secret_key="fake_secret_key"
    )
    mock_load_settings.return_value = mock_settings

    # 模拟函数行为
    mock_normalize.return_value = 'mock_normalized_wav.wav'
    mock_segment.return_value = ['mock_segmented_chunk1.wav']
    mock_get_token.return_value = 'fake_token'
    mock_baidu_asr.return_value = '识别的文本内容'

    # 执行测试，传入一个MP3路径
    result = audio_to_text("dummy_path.mp3")

    # 断言
    assert isinstance(result, str)
    assert len(result) > 0
    # 验证 normalize 函数被调用（表明MP3被处理）
    mock_normalize.assert_called_once()

@patch('src.utils.asr_processor._load_settings')
@patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono')
@patch('src.utils.asr_processor.segment_wav')
@patch('src.utils.asr_processor._baidu_get_access_token')
@patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text')
def test_audio_to_text_long_audio(
    mock_baidu_asr, 
    mock_get_token, 
    mock_segment, 
    mock_normalize, 
    mock_load_settings,
):
    """测试长音频分段处理"""
    # 模拟加载设置
    mock_settings = AsrSettings(
        provider="baidu",
        timeout=18.0,
        dev_pid=1537,
        chunk_seconds=1,  # 设置为1秒以强制触发分段
        target_sample_rate=16000,
        target_channels=1,
        baidu_api_key="fake_api_key",
        baidu_secret_key="fake_secret_key"
    )
    mock_load_settings.return_value = mock_settings

    # 模拟函数行为
    mock_normalize.return_value = 'mock_normalized_wav.wav'
    mock_segment.return_value = ['chunk1.wav', 'chunk2.wav']
    mock_get_token.return_value = 'fake_token'
    # 使用side_effect模拟多次调用返回不同结果
    mock_baidu_asr.side_effect = ['第一段文本', '第二段文本']

    # 执行测试
    result = audio_to_text(sample_audio_path)

    # 断言
    assert isinstance(result, str)
    assert '第一段文本' in result
    assert '第二段文本' in result
    # 验证百度ASR函数被调用了两次
    assert mock_baidu_asr.call_count == 2

@patch('src.utils.asr_processor._load_settings')
def test_audio_to_text_missing_config_error(mock_load_settings):
    """测试缺少配置时抛出异常"""
    # 模拟未找到配置文件
    mock_load_settings.side_effect = FileNotFoundError("Config not found")

    with pytest.raises(AsrError, match="请求百度 ASR 失败（配置文件缺失）"):
        audio_to_text("dummy_path.wav")