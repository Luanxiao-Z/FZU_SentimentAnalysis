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
def test_audio_to_text_wav(mock_load_settings, sample_audio_path):
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

    # 模拟内部函数调用，直接返回已知存在的sample_audio_path的字符串形式
    with patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono', return_value=str(sample_audio_path)), \
         patch('src.utils.asr_processor.segment_wav', return_value=[str(sample_audio_path)]), \
         patch('src.utils.asr_processor._baidu_get_access_token', return_value='fake_token'), \
         patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text', return_value='识别的文本内容'):

        result = audio_to_text(sample_audio_path)


        assert isinstance(result, str)
        assert len(result) > 0

@patch('src.utils.asr_processor._load_settings')
def test_audio_to_text_mp3(mock_load_settings, sample_audio_path):
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

    # 模拟文件扩展名检查和转换
    mp3_path = Path(str(sample_audio_path).replace('.wav', '.mp3'))
    # 直接mock掉normalize_audio_to_wav_16k_mono，返回已知存在的sample_audio_path
    with patch('pathlib.Path.suffix', new_callable=PropertyMock) as mock_suffix, \
         patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono', return_value=str(sample_audio_path)), \
         patch('src.utils.asr_processor.segment_wav', return_value=[str(sample_audio_path)]), \
         patch('src.utils.asr_processor._baidu_get_access_token', return_value='fake_token'), \
         patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text', return_value='识别的文本内容'):
        mock_suffix.return_value = '.mp3'

        result = audio_to_text(mp3_path)

        assert isinstance(result, str)
        assert len(result) > 0

@patch('src.utils.asr_processor._load_settings')
def test_audio_to_text_long_audio(mock_load_settings, sample_audio_path):
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

    # 模拟长音频（例如60秒）被分割成多个片段
    with patch('src.utils.asr_processor.normalize_audio_to_wav_16k_mono', return_value=str(sample_audio_path)), \
         patch('src.utils.asr_processor.segment_wav', return_value=[str(sample_audio_path), str(sample_audio_path)]), \
         patch('src.utils.asr_processor._baidu_get_access_token', return_value='fake_token'), \
         patch('src.utils.asr_processor._baidu_asr_wav_bytes_to_text', side_effect=['第一段文本', '第二段文本']):

        result = audio_to_text(sample_audio_path)

        assert isinstance(result, str)
        assert '第一段文本' in result
        assert '第二段文本' in result

@patch('src.utils.asr_processor._load_settings')
def test_audio_to_text_missing_config_error(mock_load_settings):
    """测试缺少配置时抛出异常"""
    # 模拟未找到配置文件
    mock_load_settings.side_effect = FileNotFoundError("Config not found")

    with pytest.raises(AsrError, match="请求百度 ASR 失败（配置文件缺失）"):
        audio_to_text("dummy_path.wav")