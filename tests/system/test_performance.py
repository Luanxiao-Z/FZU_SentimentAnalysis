"""
性能基准测试
"""
import pytest
import time

from utils import extract_text_from_image


class TestPerformance:

    def test_model_inference_time(self, loaded_model):
        """测试模型单次推理时间"""
        text = "这是一个测试句子。"

        start = time.perf_counter()
        loaded_model.predict(text)
        end = time.perf_counter()

        inference_time = (end - start) * 1000  # 毫秒

        # CPU 环境下应小于 500ms
        assert inference_time < 500, f"推理时间过长：{inference_time}ms"
        print(f"✓ 单次推理时间：{inference_time:.2f}ms")

    def test_batch_processing_time(self, loaded_model):
        """测试批量处理性能"""
        texts = ["测试句子"] * 50

        start = time.perf_counter()
        loaded_model.predict_batch(texts)
        end = time.perf_counter()

        total_time = end - start
        avg_time = (total_time / len(texts)) * 1000

        # 平均每条应小于 200ms
        assert avg_time < 200, f"平均处理时间过长：{avg_time}ms"
        print(f"✓ 批量处理平均耗时：{avg_time:.2f}ms/条")

    @pytest.mark.slow
    def test_ocr_response_time(self, sample_image_path):
        """测试 OCR 响应时间（真实 API）"""
        start = time.perf_counter()
        extract_text_from_image(sample_image_path)
        end = time.perf_counter()

        ocr_time = end - start

        # 网络请求应在 10 秒内完成
        assert ocr_time < 10.0, f"OCR 响应时间过长：{ocr_time}s"
        print(f"✓ OCR 响应时间：{ocr_time:.2f}s")