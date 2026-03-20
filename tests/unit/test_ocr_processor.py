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

def test_baidu_api_call_success():
    """模拟百度API成功调用。"""
    # 此处使用mock来模拟requests.post的返回值
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "words_result": [
            {"words": "这是一个测试文本"}
        ],
        "words_result_num": 1
    }

    with patch('src.utils.ocr_processor.requests.post', return_value=mock_response),\
         patch('src.utils.ocr_processor._baidu_get_access_token', return_value='fake_token'):
        result = extract_text_from_image(b"fake_image_data", provider="baidu")
        assert result == "这是一个测试文本"

def test_baidu_api_call_failure():
    """模拟百度API网络失败。"""
    with patch('src.utils.ocr_processor.requests.post', side_effect=requests.exceptions.RequestException("Network error")),\
         patch('src.utils.ocr_processor._baidu_get_access_token', return_value='fake_token'):
        with pytest.raises(OcrError, match="请求百度 OCR 失败（网络异常）"):
            extract_text_from_image(b"fake_image_data", provider="baidu")

def test_load_settings_from_default_path():
    """测试从默认路径加载设置。"""
    with patch('src.utils.ocr_processor.Path.exists', return_value=True),\
         patch('src.utils.ocr_processor._default_config_path', return_value=Path("/mock/path/to/config.toml")),\
         patch('src.utils.ocr_processor.open', mock_open(read_data='[ocr]\nprovider = "baidu"\ntimeout = 15.0')),\
         patch('src.utils.ocr_processor.tomllib.loads', return_value={"ocr": {"provider": "baidu", "timeout": 15.0}}):
        settings = _load_settings(config_path=None)
        assert settings.provider == "baidu"
        assert settings.timeout == 15.0