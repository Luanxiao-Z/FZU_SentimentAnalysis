"""
统一调度（多模态）情感分析流水线

当前实现：
1) OCR 图片 -> 文本 -> BERT 推理 -> 情感结果
2) 视频/音频链路：留出可插拔 ASR 回调接口，尚未接入时会抛出清晰异常
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, Union, overload

from .model_handler import EmotionModelHandler
from .utils.ocr_processor import extract_text_from_image

ImageInput = Union[str, Path, bytes, BinaryIO]
AsrFunc = Callable[[str], str]  # asr_func(audio_path: str) -> transcript


def ocr_image_to_emotion(
    handler: EmotionModelHandler,
    image: ImageInput,
    *,
    provider: str | None = None,
    preprocess: bool = True,
    lang: str = "CHN_ENG",
    timeout: float | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    图片 -> OCR -> 文本 -> 细粒度情感推理（BERT）

    Returns:
        dict: 包含 source/input_text + EmotionModelHandler.predict() 的字段
    """
    text = extract_text_from_image(
        image,
        provider=provider,
        preprocess=preprocess,
        lang=lang,
        timeout=timeout,
        config_path=config_path,
    )
    text = (text or "").strip()
    if not text:
        raise ValueError("OCR 未识别到有效文本（空文本）。")

    emotion_result = handler.predict(text)
    return {"source": "ocr", "input_text": text, **emotion_result}


def video_to_emotion(
    handler: EmotionModelHandler,
    video_path: str | Path,
    *,
    asr_func: AsrFunc | None = None,
    cleanup_audio: bool = True,
) -> dict[str, Any]:
    """
    视频 -> 音频 -> ASR -> 文本 -> 情感推理

    注意：
    - 本项目的 ASR 目前未实现，因此需要你传入可工作的 `asr_func`
      或先补齐 `src/utils/asr_processor.py`。
    """
    # 延迟导入：避免在只使用 OCR 时强依赖 moviepy/ffmpeg
    from .utils.video_processor import video_to_transcript

    if asr_func is None:
        from .utils.asr_processor import transcribe_audio

        def asr_func(audio_path: str) -> str:
            return transcribe_audio(audio_path)

    transcript = video_to_transcript(
        str(video_path),
        asr_func=asr_func,
        cleanup_audio=cleanup_audio,
    )
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("ASR 未返回有效文本（空文本）。")

    emotion_result = handler.predict(transcript)
    return {"source": "video", "input_text": transcript, **emotion_result}


def multimodal_emotion_predict(
    handler: EmotionModelHandler,
    *,
    image: ImageInput | None = None,
    video_path: str | Path | None = None,
    asr_func: AsrFunc | None = None,
    ocr_kwargs: Optional[dict[str, Any]] = None,
    video_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    统一入口：
    - 提供 `image`：走 OCR -> BERT
    - 提供 `video_path`：走 视频 -> ASR -> BERT
    """
    if (image is None) == (video_path is None):
        raise ValueError("必须且只能提供一个输入：`image` 或 `video_path`。")

    ocr_kwargs = ocr_kwargs or {}
    video_kwargs = video_kwargs or {}

    if image is not None:
        return ocr_image_to_emotion(handler, image, **ocr_kwargs)

    # video
    merged_video_kwargs = dict(video_kwargs)
    if asr_func is not None:
        merged_video_kwargs["asr_func"] = asr_func
    return video_to_emotion(handler, video_path, **merged_video_kwargs)


__all__ = [
    "ocr_image_to_emotion",
    "video_to_emotion",
    "multimodal_emotion_predict",
]

