"""
多模态集成测试

根据 src/multimodal_pipeline.py 的实现，本测试用例集覆盖了其所有公共接口。
- 测试统一入口函数 `multimodal_emotion_predict` 的三种输入模式（图片、音频、视频）
- 验证各模式下参数的正确传递和错误处理
- 使用 mock 来隔离外部依赖（OCR, ASR），专注于流程逻辑
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.asr_processor import AsrError

# 导入待测模块
from src.multimodal_pipeline import multimodal_emotion_predict


class TestMultimodalPipeline:

    @pytest.fixture
    def mock_predictor(self):
        """为 _EmotionPredictor 类创建一个 mock fixture"""
        with patch("src.multimodal_pipeline._get_default_predictor") as mock_get_pred:
            mock_pred = MagicMock()
            mock_pred.predict.return_value = {
                "fine_id": 0,
                "fine_name": "开心",
                "coarse": "正面",
                "confidence": 0.95,
                "all_probabilities": {"开心": 0.95},
                "top3": [("开心", 0.95)]
            }
            mock_get_pred.return_value = mock_pred
            yield mock_pred

    @patch("src.multimodal_pipeline.extract_text_from_image", return_value="这是一个测试图片。")
    def test_image_input_success(self, mock_ocr, sample_image_path, mock_predictor):
        """测试图片输入成功路径"""
        # 准备
        expected_text = "这是一个测试图片。"

        # 执行
        result = multimodal_emotion_predict(image=sample_image_path)

        # 断言
        mock_ocr.assert_called_once()
        assert Path(mock_ocr.call_args[0][0]) == sample_image_path
        mock_predictor.predict.assert_called_once_with(expected_text)
        assert result["source"] == "image_ocr"
        assert result["input_text"] == expected_text

    @pytest.mark.usefixtures("sample_audio_path")
    @patch("src.multimodal_pipeline.audio_to_text", return_value="这是一段音频转录。")
    def test_audio_input_success(self, mock_audio_to_text, sample_audio_path, mock_predictor):
        """测试音频输入成功路径"""
        # 准备
        expected_transcript = "这是一段音频转录。"

        # 执行
        result = multimodal_emotion_predict(audio=str(sample_audio_path))

        # 断言
        mock_audio_to_text.assert_called_once_with(str(sample_audio_path), input_suffix_hint=None, config_path=None, keep_temp=False)
        mock_predictor.predict.assert_called_once_with(expected_transcript)
        assert result["source"] == "audio_asr"
        assert result["input_text"] == expected_transcript

    def test_video_input_success(self, tmp_path, mock_predictor):
        """测试视频输入成功路径"""
        # 准备
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake_mp4_data")
        expected_transcript = "这是从视频中提取的转录。"

        # 使用嵌套的 with 语句，确保 mock 在断言时仍然有效
        with patch("src.utils.video_processor.video_to_transcript", return_value=expected_transcript) as mock_vtt:
            # 执行
            result = multimodal_emotion_predict(video_path=str(video_path))
            # 断言
            mock_vtt.assert_called_once_with(str(video_path), asr_func=None, cleanup_audio=True)
        mock_predictor.predict.assert_called_once_with(expected_transcript)
        assert result["source"] == "video_asr"
        assert result["input_text"] == expected_transcript

    def test_invalid_multiple_inputs(self, sample_image_path, mock_predictor):
        """测试同时提供多个输入时抛出 ValueError"""
        with pytest.raises(ValueError, match="必须且只能提供一个输入"):
            multimodal_emotion_predict(image=sample_image_path, audio="fake.wav")

    def test_invalid_no_input(self, mock_predictor):
        """测试不提供任何输入时抛出 ValueError"""
        with pytest.raises(ValueError, match="必须且只能提供一个输入"):
            multimodal_emotion_predict()
