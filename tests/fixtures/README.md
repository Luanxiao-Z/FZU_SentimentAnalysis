# 测试样本文件说明

此目录用于存放单元测试所需的各类样本文件。每个子目录存在相应的文件。

## 1. sample_audio (音频样本)
- **用途**: 为 `src/utils/asr_processor.py` 模块的单元测试提供音频输入。
- **文件要求**:
  - 格式: `.wav`, `.mp3`
  - 内容: 清晰、可识别的中文语音片段。
  - 命名: 
    - `test_normal.wav`: 包含一段清晰普通话的WAV音频。
    - `test_silence.mp3`: 一个几乎无声的MP3文件，用于测试静音或极低音量场景。
    - `test_long_60s.wav`: 一个约60秒的长音频，用于测试分段处理逻辑。

## 2. sample_csv (CSV表格样本)
- **用途**: 为批量文本分析功能提供结构化数据输入。
- **文件要求**:
  - 格式: `.csv` (UTF-8编码)
  - 内容: 至少包含一列名为 `text` 的字段，用于存放待分析的文本。
  - 命名: 
    - `small_batch.csv`: 小批量数据（例如10条记录），用于快速测试。
    - `large_batch.csv`: 大批量数据（例如1000条记录），用于集成测试。
    - `diverse_emotions.csv`: 包含多种情感类别（开心、悲伤、生气等）的文本，用于覆盖性测试。
    - `edge_cases.csv`: 包含特殊字符、超长文本、空行等边界情况的文件，用于健壮性测试。

## 3. sample_images (图片样本)
- **用途**: 为 `src/utils/ocr_processor.py` 模块的单元测试提供图像输入。
- **文件要求**:
  - 格式: `.jpg`, `.png`
  - 内容: 图片上需有清晰可读的中文文本。
  - 命名: 
    - `text_simple.jpg`: 纯文字、背景简单的图片，OCR应能轻松识别。
    - `text_complex.png`: 文字与复杂背景混合的图片，用于测试OCR鲁棒性。
    - `test_empty.jpg`: 不包含任何文字的纯图片，用于测试无文本情况下的返回值。
    - `test_happy.jpg`: 一个包含开心文字的图片。
    - `test_sad.jpg`: 一个包含悲伤文字的图片。
    - `test_neutral.jpg`: 一个包含中性文字的图片。

## 4. sample_video (视频样本)
- **用途**: 为多模态分析中的视频处理功能提供输入。
- **文件要求**:
  - 格式: `.mp4`, `.avi`
  - 内容: 视频中应包含需要通过ASR提取的音频部分。
  - 命名: 
    - `video_short.mp4`: 一个短小的视频，其音频内容与 `sample_audio/test_normal.wav` 相同，用于验证视频到音频再到文本的完整流程。
    - `video_avi.avi` : 一个AVI格式的视频，用于测试视频处理逻辑。

## 5. sample_documents (文档样本)
- **用途**: 为 `src/utils/document_processor.py` 模块的单元测试提供文档输入。
- **文件要求**:
  - 格式: `.pdf`, `.docx`, `.txt`, `.md`
  - 内容: 文档内需包含至少两句话以上的中文文本，以测试句子分割功能。
  - 命名: 
    - `doc_sample.pdf`: 一个标准的PDF文档。
    - `doc_sample.docx`: 一个Word文档。
    - `doc_sample.txt`: 一个纯文本文档。
    - `doc_sample.md`: 一个Markdown格式的文档。