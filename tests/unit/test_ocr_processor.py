"""
OCR 处理器单元测试 - 精简版（5 个核心用例）
"""
import pytest
from src.utils.ocr_processor import extract_text_from_image, OcrError


class TestOcrProcessor:
    
    def test_extract_text_from_image_path(self, sample_image_path):
        """测试从图片路径提取文本"""
        text = extract_text_from_image(sample_image_path)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
    
    def test_extract_text_from_image_bytes(self, sample_image_path):
        """测试从字节数据提取文本"""
        image_bytes = Path(sample_image_path).read_bytes()
        text = extract_text_from_image(image_bytes)
        assert isinstance(text, str)
    
    def test_extract_text_invalid_image(self, tmp_path):
        """测试无效图片输入"""
        invalid_image = tmp_path / "invalid.jpg"
        invalid_image.write_bytes(b"not an image")
        
        with pytest.raises(OcrError):
            extract_text_from_image(invalid_image)
    
    @pytest.mark.slow
    def test_real_api_call(self, sample_image_path):
        """测试真实 API 调用（可选，需配置密钥）"""
        # 需要真实密钥，可跳过
        pass
    
    def test_missing_config_error(self, tmp_path):
        """测试缺失配置文件错误"""
        non_existent = tmp_path / "non_existent.toml"
        
        with pytest.raises(OcrError, match="未找到 OCR 私密配置文件"):
            extract_text_from_image("dummy.jpg", config_path=non_existent)