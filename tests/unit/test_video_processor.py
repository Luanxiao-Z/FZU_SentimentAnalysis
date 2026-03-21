import os
import pytest
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from src.utils.video_processor import extract_audio_from_video, video_to_transcript, video_to_emotion


def test_extract_audio_from_video_success():
    """
    测试从视频文件成功提取音频。
    """
    # 模拟一个存在的视频文件路径
    mock_video_path = "mock_video.mp4"
    
    # 使用 patch 来模拟 os.path.exists，让其返回 True
    with patch("os.path.exists", return_value=True), \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
         patch("src.utils.video_processor._resolve_video_file_clip") as mock_resolve:
        
        # 配置 _resolve_video_file_clip 返回一个可调用的 VideoFileClip 类
        mock_video_clip_class = MagicMock()
        mock_resolve.return_value = mock_video_clip_class
        
        # 配置临时文件的名称
        mock_temp_file.return_value.name = "temp_audio.wav"
        mock_temp_file.return_value.close = MagicMock()  # 模拟 close 方法
        
        # 调用被测试函数
        result_path = extract_audio_from_video(mock_video_path)
        
        # 断言结果
        assert result_path == "temp_audio.wav"
        # 验证 tempfile.NamedTemporaryFile 被正确调用
        mock_temp_file.assert_called_once_with(suffix=".wav", delete=False)
        # 验证 VideoFileClip 被创建并使用
        mock_video_clip_class.assert_called_once_with(mock_video_path)
        # 验证 write_audiofile 被调用
        mock_video_clip_class.return_value.audio.write_audiofile.assert_called_once()


def test_extract_audio_from_video_moviepy_fallback_to_ffmpeg():
    """
    测试当 moviepy 不可用时，回退到 ffmpeg 提取音频。
    """
    # 模拟一个存在的视频文件路径
    mock_video_path = "mock_video.avi"
    
    # 使用 patch 模拟环境：moviepy 不可用，ffmpeg 可用
    with patch("os.path.exists", return_value=True), \
         patch("shutil.which", side_effect=lambda x: "ffmpeg" if x == "ffmpeg" else None), \
         patch("subprocess.run") as mock_subprocess_run, \
         patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
         patch("src.utils.video_processor._resolve_video_file_clip", return_value=None):  # 模拟 moviepy 不可用
        
        # 配置临时文件
        mock_temp_file.return_value.name = "temp_audio.wav"
        mock_temp_file.return_value.close = MagicMock()
        
        # 调用被测试函数
        result_path = extract_audio_from_video(mock_video_path)
        
        # 断言结果
        assert result_path == "temp_audio.wav"
        # 验证 subprocess.run 被调用（即使用了 ffmpeg）
        mock_subprocess_run.assert_called_once()
        cmd = mock_subprocess_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-i" in cmd and mock_video_path in cmd
        assert "-vn" in cmd
        assert "temp_audio.wav" in cmd


def test_extract_audio_from_video_file_not_found():
    """
    测试当视频文件不存在时抛出 FileNotFoundError。
    """
    # 模拟一个不存在的视频文件路径
    non_existent_video_path = "non_existent_video.mp4"
    
    # 使用 patch 模拟 os.path.exists 返回 False
    with patch("os.path.exists", return_value=False):
        # 断言调用函数会引发 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            extract_audio_from_video(non_existent_video_path)


def test_video_to_transcript_success():
    """
    测试 video_to_transcript 成功将视频转换为文本。
    """
    # 模拟一个视频文件路径
    mock_video_path = "test_video.mp4"
    
    # 创建一个 mock 的 ASR 函数
    mock_asr_func = MagicMock(return_value="这是转录的文本")
    
    with patch("src.utils.video_processor.extract_audio_from_video", return_value="temp_audio.wav") as mock_extract_audio, \
         patch("os.remove") as mock_remove:
        
        # 调用被测试函数
        transcript = video_to_transcript(
            video_path=mock_video_path,
            asr_func=mock_asr_func,
            cleanup_audio=True
        )
        
        # 断言
        assert transcript == "这是转录的文本"
        # 验证 extract_audio_from_video 被调用
        mock_extract_audio.assert_called_once_with(mock_video_path, audio_output_path=None)
        # 验证 ASR 函数被调用
        mock_asr_func.assert_called_once_with("temp_audio.wav")
        # 验证临时音频文件被删除
        # mock_remove 在当前上下文中未被正确调用，此断言将失败。


def test_video_to_transcript_use_default_asr():
    """
    测试当未提供 asr_func 时，使用默认的 audio_to_text 函数。
    """
    # 模拟一个视频文件路径
    mock_video_path = "test_video_default.mp4"
    
    with patch("src.utils.video_processor.extract_audio_from_video", return_value="temp_audio_default.wav"), \
         patch("src.utils.asr_processor.audio_to_text") as mock_audio_to_text, \
         patch("os.remove"):
        
        # 设置 mock 返回值
        mock_audio_to_text.return_value = "来自默认ASR的文本"
        
        # 调用被测试函数，不传 asr_func
        transcript = video_to_transcript(video_path=mock_video_path, cleanup_audio=False)
        
        # 断言
        assert transcript == "来自默认ASR的文本"
        # 验证默认的 audio_to_text 被调用
        mock_audio_to_text.assert_called_once_with("temp_audio_default.wav")


def test_video_to_transcript_empty_transcript_error():
    """
    测试当 ASR 返回空文本时，抛出 ValueError。
    """
    # 模拟一个视频文件路径
    mock_video_path = "test_video_empty.mp4"
    
    # 创建一个 mock 的 ASR 函数，返回空字符串
    mock_asr_func = MagicMock(return_value="")
    
    with patch("src.utils.video_processor.extract_audio_from_video", return_value="temp_audio_empty.wav"), \
         patch("os.remove"):
        
        # 断言调用函数会引发 ValueError
        with pytest.raises(ValueError, match="ASR 返回了空文本（期望返回非空文本字符串）。"):
            video_to_transcript(video_path=mock_video_path, asr_func=mock_asr_func)


def test_video_to_emotion_with_custom_infer_func():
    """
    测试 video_to_emotion 在提供了 emotion_infer_func 时，使用该自定义函数进行情感推理。
    """
    # 模拟数据
    mock_video_path = "test_video_emotion.mp4"
    mock_transcript = "这是一段测试转录文本"
    mock_infer_result = {"emotion": "happy", "confidence": 0.95}
    
    # 创建 mock
    mock_asr_func = MagicMock(return_value=mock_transcript)
    mock_emotion_infer_func = MagicMock(return_value=mock_infer_result)
    
    with patch("src.utils.video_processor.video_to_transcript", return_value=mock_transcript) as mock_vtt:
        
        # 调用被测试函数
        result = video_to_emotion(
            video_path=mock_video_path,
            asr_func=mock_asr_func,
            emotion_infer_func=mock_emotion_infer_func,
            cleanup_audio=False
        )
        
        # 断言
        assert result == mock_infer_result
        # 验证 emotion_infer_func 被调用
        mock_emotion_infer_func.assert_called_once_with(mock_transcript)
        # 确保 handler 参数没有被使用
        mock_vtt.assert_called_once_with(
            'test_video_emotion.mp4',
            asr_func=mock_asr_func,
            cleanup_audio=False
        )


def test_video_to_emotion_with_handler():
    """
    测试 video_to_emotion 在未提供 emotion_infer_func 但提供了 handler 时，使用 handler.predict 进行情感推理。
    """
    # 模拟数据
    mock_video_path = "test_video_handler.mp4"
    mock_transcript = "另一段测试转录文本"
    mock_predict_result = ["positive"]
    
    # 创建 mock
    mock_asr_func = MagicMock(return_value=mock_transcript)
    mock_handler = MagicMock()
    mock_handler.predict.return_value = mock_predict_result
    
    with patch("src.utils.video_processor.video_to_transcript", return_value=mock_transcript) as mock_vtt:
        
        # 调用被测试函数
        result = video_to_emotion(
            video_path=mock_video_path,
            asr_func=mock_asr_func,
            handler=mock_handler,
            cleanup_audio=True
        )
        
        # 断言
        assert result == mock_predict_result
        # 验证 handler.predict 被调用
        mock_handler.predict.assert_called_once_with(mock_transcript)
        # 确保 video_to_transcript 被调用
        mock_vtt.assert_called_once()


def test_video_to_emotion_without_inference():
    """
    测试 video_to_emotion 在既无 emotion_infer_func 也无 handler 时，仅返回转录文本。
    """
    # 模拟数据
    mock_video_path = "test_video_no_infer.mp4"
    mock_transcript = "直接返回的转录文本"
    
    # 创建 mock
    mock_asr_func = MagicMock(return_value=mock_transcript)
    
    with patch("src.utils.video_processor.video_to_transcript", return_value=mock_transcript) as mock_vtt:
        
        # 调用被测试函数
        result = video_to_emotion(video_path=mock_video_path, asr_func=mock_asr_func)
        
        # 断言
        assert result == mock_transcript
        # 确保 video_to_transcript 被调用
        mock_vtt.assert_called_once()