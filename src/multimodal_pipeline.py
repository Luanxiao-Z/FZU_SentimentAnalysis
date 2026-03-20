"""
统一调度（多模态）情感分析流水线

支持输入：
1) 图片：OCR -> 文本 -> `EmotionModelHandler.predict()` -> 情感
2) 音频：ASR -> 文本 -> `EmotionModelHandler.predict()` -> 情感
3) 视频：视频提取音频 -> ASR -> 文本 -> `EmotionModelHandler.predict()` -> 情感
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, Union

from .model_handler import EmotionModelHandler
from .utils.ocr_processor import extract_text_from_image
from .utils.asr_processor import audio_to_text

ImageInput = Union[str, Path, bytes, BinaryIO]
AudioInput = Union[str, Path, bytes, BinaryIO]
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
    """图片 -> OCR -> 文本 -> 情感推理"""
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
    return {"source": "image_ocr", "input_text": text, **emotion_result}


def audio_to_emotion(
    handler: EmotionModelHandler,
    audio: AudioInput,
    *,
    input_suffix_hint: str | None = None,
    config_path: str | Path | None = None,
    keep_temp: bool = False,
) -> dict[str, Any]:
    """音频 -> ASR -> 文本 -> 情感推理"""
    transcript = audio_to_text(
        audio,
        input_suffix_hint=input_suffix_hint,
        config_path=config_path,
        keep_temp=keep_temp,
    )
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("ASR 未返回有效文本（空文本）。")

    emotion_result = handler.predict(transcript)
    return {"source": "audio_asr", "input_text": transcript, **emotion_result}


def video_to_emotion(
    handler: EmotionModelHandler,
    video_path: str | Path,
    *,
    asr_config_path: str | Path | None = None,
    asr_keep_temp: bool = False,
    cleanup_audio: bool = True,
) -> dict[str, Any]:
    """视频 -> 音频提取 -> ASR -> 文本 -> 情感推理"""
    # 延迟导入：避免只跑 OCR 时就强依赖 moviepy/ffmpeg
    from .utils.video_processor import video_to_transcript

    asr_func: AsrFunc | None = None
    if asr_config_path is not None or asr_keep_temp:
        def asr_func(audio_path: str) -> str:
            return audio_to_text(audio_path, config_path=asr_config_path, keep_temp=asr_keep_temp)

    transcript = video_to_transcript(
        str(video_path),
        asr_func=asr_func,
        cleanup_audio=cleanup_audio,
    )
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("ASR 未返回有效文本（空文本）。")

    emotion_result = handler.predict(transcript)
    return {"source": "video_asr", "input_text": transcript, **emotion_result}


def multimodal_emotion_predict(
    handler: EmotionModelHandler,
    *,
    image: ImageInput | None = None,
    audio: AudioInput | None = None,
    video_path: str | Path | None = None,
    # OCR 参数
    ocr_kwargs: Optional[dict[str, Any]] = None,
    # ASR 参数（audio 的 config_path/keep_temp；video 的 asr_config_path/asr_keep_temp）
    asr_kwargs: Optional[dict[str, Any]] = None,
    cleanup_audio: bool = True,
) -> dict[str, Any]:
    """
    统一入口：只能提供以下三者之一
    - `image`
    - `audio`
    - `video_path`
    """
    provided = [image is not None, audio is not None, video_path is not None]
    if sum(1 for x in provided if x) != 1:
        raise ValueError("必须且只能提供一个输入：`image` 或 `audio` 或 `video_path`。")

    ocr_kwargs = ocr_kwargs or {}
    asr_kwargs = asr_kwargs or {}

    if image is not None:
        return ocr_image_to_emotion(handler, image, **ocr_kwargs)

    if audio is not None:
        return audio_to_emotion(handler, audio, **asr_kwargs)

    # video
    return video_to_emotion(
        handler,
        video_path,
        asr_config_path=asr_kwargs.get("config_path"),
        asr_keep_temp=bool(asr_kwargs.get("keep_temp", False)),
        cleanup_audio=cleanup_audio,
    )


__all__ = ["ocr_image_to_emotion", "audio_to_emotion", "video_to_emotion", "multimodal_emotion_predict"]

