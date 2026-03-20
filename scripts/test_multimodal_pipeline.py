"""
多模态统一调度脚本快速测试

默认行为：
1) 自动扫描 `src/utils/` 目录下的图片与视频
2) 加载 BERT 情感模型
3) 分别测试：图片(OCR) -> BERT情感、视频(视频->ASR->BERT) -> BERT情感

你可以把测试用的图片/视频先放进 `src/utils/`，运行完再删掉。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal_pipeline import multimodal_emotion_predict


def _latest_file_in_dir(directory: Path, exts: Iterable[str]) -> Optional[Path]:
    exts_l = {e.lower() for e in exts}
    candidates: list[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in exts_l:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test multimodal emotion pipeline")
    parser.add_argument("--image", type=str, default=None, help="测试图片路径（不填则自动扫描 src/utils 下图片）")
    parser.add_argument("--video", type=str, default=None, help="测试视频路径（不填则自动扫描 src/utils 下视频）")
    parser.add_argument(
        "--ocr-provider",
        type=str,
        default="baidu",
        help="OCR provider：baidu / aliyun（本项目默认 baidu）",
    )
    parser.add_argument("--keep-temp", action="store_true", help="ASR 分段/临时文件是否保留（调试用）")
    parser.add_argument("--cleanup-audio", action="store_true", help="视频链路是否清理临时音频 wav（建议不选，保持默认删除；选了会 cleanup=True）")
    parser.add_argument(
        "--no-cleanup-audio",
        action="store_true",
        help="视频链路不删除临时音频 wav（调试用，优先级高于 --cleanup-audio）",
    )
    args = parser.parse_args()

    utils_dir = PROJECT_ROOT / "src" / "utils"

    # 默认找最新文件
    image_path = Path(args.image) if args.image else _latest_file_in_dir(utils_dir, [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    video_path = Path(args.video) if args.video else _latest_file_in_dir(utils_dir, [".mp4", ".mov", ".avi", ".mkv", ".webm"])

    if image_path is None and video_path is None:
        raise SystemExit(
            f"未找到测试输入：请通过 --image/--video 指定路径，或在 `src/utils/` 放入图片/视频。\n"
            f"扫描目录：{utils_dir}"
        )

    # cleanup_audio：默认 True（删除临时音频）
    cleanup_audio = True
    if args.no_cleanup_audio:
        cleanup_audio = False
    elif args.cleanup_audio:
        cleanup_audio = True

    if image_path is not None:
        print(f"\n========== 测试图片：{image_path} ==========")
        try:
            res = multimodal_emotion_predict(
                image=image_path,
                ocr_kwargs={"provider": args.ocr_provider},
            )
            print(res)
        except Exception as e:
            print(f"[图片测试失败] {type(e).__name__}: {e}")

    if video_path is not None:
        print(f"\n========== 测试视频：{video_path} ==========")
        try:
            res = multimodal_emotion_predict(
                video_path=video_path,
                asr_kwargs={"keep_temp": bool(args.keep_temp)},
                cleanup_audio=cleanup_audio,
            )
            print(res)
        except Exception as e:
            print(f"[视频测试失败] {type(e).__name__}: {e}")

    print("\n========== 测试结束 ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

