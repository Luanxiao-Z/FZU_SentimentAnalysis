"""
统一调度（多模态）情感分析流水线

支持输入：
1) 图片：OCR -> 文本 -> BERT 推理 -> 情感
2) 音频：ASR -> 文本 -> BERT 推理 -> 情感
3) 视频：视频提取音频 -> ASR -> 文本 -> BERT 推理 -> 情感
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, Union

import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from .config import COARSE_MAP, EMOTION_NAMES, MODEL_CONFIG, MODEL_PATH, NUM_LABELS
from .utils.asr_processor import audio_to_text
from .utils.ocr_processor import extract_text_from_image

ImageInput = Union[str, Path, bytes, BinaryIO]
AudioInput = Union[str, Path, bytes, BinaryIO]
AsrFunc = Callable[[str], str]  # asr_func(audio_path: str) -> transcript


class _EmotionPredictor:
    """直接从训练产物目录加载模型并完成推理（不依赖 model_handler.py）"""

    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=MODEL_CONFIG["truncation"],
            max_length=MODEL_CONFIG["max_length"],
            padding=MODEL_CONFIG["padding"],
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probabilities).item()
        confidence = probabilities[pred_id].item()

        all_probs = {EMOTION_NAMES[i]: probabilities[i].item() for i in range(int(NUM_LABELS))}
        coarse = COARSE_MAP[pred_id]
        top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "fine_id": pred_id,
            "fine_name": EMOTION_NAMES[pred_id],
            "coarse": coarse,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "top3": top3,
        }


@lru_cache(maxsize=2)
def _get_default_predictor(model_path: str) -> _EmotionPredictor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return _EmotionPredictor(tokenizer=tokenizer, model=model, device=device)


def ocr_image_to_emotion(
    predictor: _EmotionPredictor,
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

    emotion_result = predictor.predict(text)
    return {"source": "image_ocr", "input_text": text, **emotion_result}


def audio_to_emotion(
    predictor: _EmotionPredictor,
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

    emotion_result = predictor.predict(transcript)
    return {"source": "audio_asr", "input_text": transcript, **emotion_result}


def video_to_emotion(
    predictor: _EmotionPredictor,
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

    emotion_result = predictor.predict(transcript)
    return {"source": "video_asr", "input_text": transcript, **emotion_result}


def multimodal_emotion_predict(
    handler: Any | None = None,  # 保留参数兼容旧调用；不会调用 model_handler.py
    *,
    image: ImageInput | None = None,
    audio: AudioInput | None = None,
    video_path: str | Path | None = None,
    # OCR 参数
    ocr_kwargs: Optional[dict[str, Any]] = None,
    # ASR 参数（audio 的 config_path/keep_temp；video 的 asr_config_path/asr_keep_temp）
    asr_kwargs: Optional[dict[str, Any]] = None,
    cleanup_audio: bool = True,
    model_path: str | Path | None = None,
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

    # 统一调度内部直接加载训练产物模型
    p = model_path or MODEL_PATH
    predictor = _get_default_predictor(str(p))

    if image is not None:
        return ocr_image_to_emotion(predictor, image, **ocr_kwargs)

    if audio is not None:
        return audio_to_emotion(predictor, audio, **asr_kwargs)

    # video
    return video_to_emotion(
        predictor,
        video_path,
        asr_config_path=asr_kwargs.get("config_path"),
        asr_keep_temp=bool(asr_kwargs.get("keep_temp", False)),
        cleanup_audio=cleanup_audio,
    )


__all__ = ["ocr_image_to_emotion", "audio_to_emotion", "video_to_emotion", "multimodal_emotion_predict"]

