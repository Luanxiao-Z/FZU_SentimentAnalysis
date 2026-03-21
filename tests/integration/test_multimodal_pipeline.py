"""
多模态集成测试
"""
import pytest
from src.utils.ocr_processor import extract_text_from_image


class TestMultimodalPipeline:

    def test_image_to_emotion_positive(self, sample_image_path, loaded_model):
        """测试图片→文本→正面情感完整流程"""
        # 步骤 1: OCR 提取文字
        text = extract_text_from_image(sample_image_path)

        # 步骤 2: 情感分析
        result = loaded_model.predict(text)

        # 验证
        assert result["coarse"] == "正面"
        assert result["fine_name"] == "开心"

    def test_image_to_emotion_negative(self, loaded_model):
        """测试负面文字图片→情感分析"""
        # 使用包含负面文字的图片
        img_path = "tests/fixtures/sample_images/sad_text.jpg"
        text = extract_text_from_image(img_path)
        result = loaded_model.predict(text)

        assert result["coarse"] == "负面"

    def test_image_to_emotion_neutral(self, loaded_model):
        """测试中性文字图片→情感分析"""
        img_path = "tests/fixtures/sample_images/neutral_text.jpg"
        text = extract_text_from_image(img_path)
        result = loaded_model.predict(text)

        assert result["coarse"] == "中性"

    def test_empty_image_handling(self, tmp_path, loaded_model):
        """测试空白图片处理"""
        from PIL import Image
        from io import BytesIO

        # 创建空白图片
        img = Image.new('RGB', (800, 600), color='white')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)

        text = extract_text_from_image(buffer)

        # 空白图片应返回空字符串
        assert text.strip() == ""