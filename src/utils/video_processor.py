# 【成员】： 102300415张伟健（视频处理与模型对接负责人）
# 任务：
# 1.使用moviepy或ffmpeg库，从上传的视频文件中提取音轨。
# 2.将提取的音频流传递给ASR接口，实现“视频→音频→文本”的转换流水线。
# 3.编写统一调度脚本multimodal_pipeline.py，将提取的文本送入现有的BERT模型进行推理，返回情感结果。
# 产出：
# utils/video_processor.py、核心推理调度接口。


import os
import shutil
import subprocess
from moviepy.editor import VideoFileClip  # type: ignore
import tempfile  # 用于处理临时文件


def extract_audio_from_video(
    video_path: str,
    audio_output_path: str | None = None,
    codec: str = "pcm_s16le",
    sample_rate: int = 16000,
):
    """
    从视频文件中提取音轨，返回音频文件的路径（临时文件或指定路径）。
    :param video_path: 视频文件的路径（如 .mp4, .avi 等）
    :param audio_output_path: 音频输出路径（若为 None，自动创建临时文件）
    :param codec: 目标音频编解码器（用于 moviepy/ffmpeg 兼容）
    :param sample_rate: 目标采样率（Hz）
    :return: 音频文件的路径（字符串）
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    created_audio = False
    try:
        # 处理音频输出路径：如果未指定，创建临时文件（后缀为 .wav）
        if audio_output_path is None:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_output_path = temp_audio.name
            temp_audio.close()  # 关闭临时文件，让后续操作写入
            created_audio = True

        # 加载视频文件
        video = VideoFileClip(video_path)
        try:
            if video.audio is None:
                raise ValueError("该视频不包含音频轨道，无法提取。")

            # 提取音频并保存（优先使用 moviepy）
            video.audio.write_audiofile(
                audio_output_path,
                codec=codec,
                fps=sample_rate,
            )
        finally:
            video.close()  # 关闭视频，释放资源

        return audio_output_path
    except Exception as e:
        # moviepy 提取失败时，尝试 ffmpeg 兜底（若环境具备 ffmpeg）
        try:
            if shutil.which("ffmpeg") is None:
                raise

            if audio_output_path is None:
                # 理论上到这里不会发生，但保持健壮性
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio_output_path = temp_audio.name
                temp_audio.close()
                created_audio = True

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le" if codec == "pcm_s16le" else codec,
                "-ar",
                str(sample_rate),
                audio_output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_output_path
        except Exception:
            # 如果 ffmpeg 也失败，则保留原始 moviepy 异常信息
            if created_audio and audio_output_path and os.path.exists(audio_output_path):
                os.remove(audio_output_path)
            raise e


def video_to_transcript(
    video_path: str,
    asr_func=None,
    cleanup_audio: bool = True,
):
    """
    将视频转换为文本：视频 -> 音频 -> ASR 得到文本。

    由于你提到 `asr` 脚本暂时未实现，这里采用“可插拔”的方式：
    - 你可以传入 `asr_func(audio_path) -> str` 来接入 ASR。
    - 如果未传入 `asr_func`，则抛出清晰异常，提醒你当前链路需要 ASR 实现。

    :param video_path: 视频文件路径
    :param asr_func: 回调函数，签名需为 asr_func(audio_path: str) -> str
    :param cleanup_audio: 若为 True，完成后删除临时音频文件
    :return: ASR 文本（字符串）
    """
    if asr_func is None:
        raise NotImplementedError(
            "当前 `asr` 尚未实现。请在调用 `video_to_transcript` 时提供 `asr_func(audio_path)->str`，"
            "或先实现对应 ASR 接口后再接入。"
        )

    audio_path = None
    try:
        audio_path = extract_audio_from_video(video_path, audio_output_path=None)
        transcript = asr_func(audio_path)
        if transcript is None:
            raise ValueError("ASR 返回了 None（期望返回文本字符串）。")
        return transcript
    finally:
        if cleanup_audio and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def video_to_emotion(
    video_path: str,
    asr_func=None,
    emotion_infer_func=None,
    cleanup_audio: bool = True,
):
    """
    视频 -> 文本 -> 情感结果（推理函数可插拔）。

    目前本文件只负责“视频->文本”的链路。情感推理（BERT）请通过 `emotion_infer_func(text)->any` 接入。

    :param video_path: 视频文件路径
    :param asr_func: 回调函数，签名 asr_func(audio_path)->str
    :param emotion_infer_func: 回调函数，签名 emotion_infer_func(text)->情感结果
    :param cleanup_audio: 是否清理临时音频
    :return: emotion_infer_func(transcript) 的结果；若未提供 emotion_infer_func，则返回 transcript
    """
    transcript = video_to_transcript(video_path, asr_func=asr_func, cleanup_audio=cleanup_audio)
    if emotion_infer_func is None:
        return transcript
    return emotion_infer_func(transcript)


# ------------------- 测试逻辑（手动替换回调） -------------------
if __name__ == "__main__":
    # 你可以在这里提供一个假的 asr_func 用于验证“提取音频 -> 调用回调”的流程。
    test_video_path = "v.mp4"

    if not os.path.exists(test_video_path):
        print(f"错误：测试视频 {test_video_path} 不存在！请检查路径。")
    else:
        def _dummy_asr(audio_path: str) -> str:
            # 仅用于演示：真实项目中请替换为你的 ASR 接口。
            return f"[dummy transcript from {os.path.basename(audio_path)}]"

        text = video_to_transcript(test_video_path, asr_func=_dummy_asr)
        print(f"转录结果：{text}")