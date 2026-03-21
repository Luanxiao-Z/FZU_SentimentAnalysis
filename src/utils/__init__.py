# ==========================================
# 工具函数包初始化
# ==========================================

"""
工具函数模块，提供以下功能：
- 文件 I/O 操作（CSV、Excel、PDF、DOCX、Markdown、TXT）
- 文本处理（句子分割）
- 数据验证和格式化
- 情感标签工具函数
"""

from .file_io import (
    load_css,
    load_csv_with_encoding,
    load_excel_with_encoding,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_md,
    extract_text_from_txt, read_csv_file, write_csv_file,
)

from .text_processing import split_chinese_sentences

from .data_validation import (
    validate_dataframe,
    format_batch_results,
    get_example_texts,
)

from .document_processor import process_document_to_sentences

from .emotion_utils import (
    emotion_id_by_name,
    get_coarse_badge_class,
)

from .ocr_processor import extract_text_from_image, OcrError

from .asr_processor import audio_to_text, AsrError

__all__ = [
    # 文件 I/O
    'load_css',
    'load_csv_with_encoding',
    'load_excel_with_encoding',
    'extract_text_from_pdf',
    'extract_text_from_docx',
    'extract_text_from_md',
    'extract_text_from_txt',
    'read_csv_file',
    'write_csv_file',
    
    # 文本处理
    'split_chinese_sentences',
    
    # 数据验证
    'validate_dataframe',
    'format_batch_results',
    'get_example_texts',
    
    # 文档处理
    'process_document_to_sentences',
    
    # 情感工具
    'emotion_id_by_name',
    'get_coarse_badge_class',

    # OCR
    'extract_text_from_image',
    'OcrError',

    # ASR
    'audio_to_text',
    'AsrError',
]
