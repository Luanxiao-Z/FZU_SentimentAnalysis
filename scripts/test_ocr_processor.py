"""
OCR 接口快速自测脚本

用法示例（在项目根目录）：
  python scripts/test_ocr_processor.py --image "data/demo.png"

说明：
  - 需要先配置 `config/ocr_secrets.toml`（该文件不会提交到 GitHub）
  - 默认 provider=baidu，可用 --provider 覆盖
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 让脚本支持在项目根目录直接运行：
# python scripts/test_ocr_processor.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ocr_processor import extract_text_from_image, OcrError


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR Processor quick test")
    parser.add_argument("--image", required=True, help="图片路径（JPG/PNG）")
    parser.add_argument("--provider", default=None, help="OCR provider：baidu / aliyun")
    parser.add_argument("--no-preprocess", action="store_true", help="关闭图片预处理（用于对比）")
    parser.add_argument("--lang", default="CHN_ENG", help="语言类型（百度 language_type），默认 CHN_ENG")
    parser.add_argument("--timeout", type=float, default=None, help="请求超时（秒），默认读取配置文件")
    parser.add_argument(
        "--config",
        default=None,
        help="私密配置文件路径（TOML），默认使用 config/ocr_secrets.toml",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERR] 图片不存在：{image_path}")
        return 2

    try:
        text = extract_text_from_image(
            str(image_path),
            provider=args.provider,
            preprocess=not args.no_preprocess,
            lang=args.lang,
            timeout=args.timeout,
            config_path=args.config,
        )
    except OcrError as e:
        print(f"[OCR ERROR] provider={e.provider} msg={e}")
        if e.details is not None:
            print(f"[DETAILS] {e.details}")
        return 1

    print("========== OCR RESULT ==========")
    print(text)
    print("========== END ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

