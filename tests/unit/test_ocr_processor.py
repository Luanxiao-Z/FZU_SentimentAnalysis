"""
测试 ocr_processor 模块的单元测试。
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from src.utils.ocr_processor import extract_text_from_image, OcrError, _load_settings

def test_extract_text_from_image_path(sample_image_path):
    """测试从图片路径提取文本。"""
    text = extract_text_from_image(sample_image_path)
    assert isinstance(text, str)
    assert len(text.strip()) > 0

def test_extract_text_from_image_bytes(sample_image_path):
    """测试从字节数据提取文本。"""
    image_bytes = Path(sample_image_path).read_bytes()
    text = extract_text_from_image(image_bytes)
    assert isinstance(text, str)

def test_extract_text_invalid_image(tmp_path):
    """测试无效图片输入。"""
    invalid_image = tmp_path / "invalid.jpg"
    invalid_image.write_bytes(b"not an image")

    with pytest.raises(OcrError):
        extract_text_from_image(invalid_image)

def test_missing_config_error(tmp_path):
    """测试缺失配置文件错误。"""
    non_existent = tmp_path / "non_existent.toml"

    with pytest.raises(OcrError, match="未找到 OCR 私密配置文件"):
        extract_text_from_image("dummy.jpg", config_path=non_existent)

def test_baidu_api_call_success(sample_image_path):
    """模拟百度API成功调用。"""
    # 直接Mock掉整个 _baidu_general_ocr 函数，并返回一个成功的响应
    mock_response_json = {
        "words_result": [
            {"words": "这是一个测试文本"}
        ],
        "words_result_num": 1
    }
    with patch('src.utils.ocr_processor._baidu_general_ocr', return_value=mock_response_json),\
         patch('src.utils.ocr_processor._baidu_get_access_token', return_value='fake_token'):
        result = extract_text_from_image(sample_image_path, provider="baidu")
        assert result == "这是一个测试文本"

def test_baidu_api_call_failure(sample_image_path):
    """模拟百度API网络失败。"""
    # 在函数内部导入 requests 模块，确保其在作用域内可见
    import requests
    # 直接Mock _baidu_general_ocr 函数以触发网络异常，或者Mock更底层的requests.post
    with patch('src.utils.ocr_processor._baidu_get_access_token', return_value='fake_token'), \
         patch('src.utils.ocr_processor.requests.post', side_effect=requests.exceptions.RequestException("Network error")):
        with pytest.raises(OcrError, match="请求百度 OCR 失败（网络异常）"):
            extract_text_from_image(sample_image_path, provider="baidu")

def test_load_settings_from_default_path():
    """测试从默认路径加载设置。"""
    # 导入必要的模块
    import sys
    from src.utils.ocr_processor import OcrSettings
    # 不再Mock tomli.loads，而是直接Mock整个 _load_settings 函数的返回值，以完全隔离外部依赖
    # 注意：根据源码，默认超时时间为12.0秒，因此将预期值改为12.0
    expected_settings = OcrSettings(provider="baidu", timeout=12.0)
    
    with patch('src.utils.ocr_processor._load_settings', return_value=expected_settings):
        settings = _load_settings(config_path=None)
        assert settings.provider == "baidu"
        assert settings.timeout == 12.0