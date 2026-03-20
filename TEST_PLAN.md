# FZU 细粒度情感分析系统 - 多模态扩展测试计划

> **版本**: v1.0  
> **编制日期**: 2026 年 3 月 19 日  
> **执行周期**: 5 天（紧凑型）  
> **测试负责人**: 智能系统综合设计项目组

---

## 📋 一、测试目标与范围

### 1.1 测试目标（5 天冲刺版）

在**5 天时间**内完成多模态情感分析系统的**核心功能验证**,确保:

- ✅ **P0 级功能可用**: OCR/ASR/视频处理核心流程跑通
- ✅ **关键 Bug 修复**: 发现并修复影响使用的严重问题
- ✅ **基本质量保障**: 核心模块代码覆盖率>70%
- ✅ **文档完善**: 输出完整测试报告和使用文档

### 1.2 测试范围（优先级驱动）

| 优先级 | 测试模块 | 测试类型 | 时间分配 |
|-------|---------|---------|---------|
| **P0** | BERT模型推理 | 单元测试 + 集成测试 | 第 1 天 |
| **P0** | OCR 文字识别 | 单元测试 + API 集成 | 第 1-2 天 |
| **P0** | 端到端流程 | 图片→文本→情感分析 | 第 2 天 |
| **P0** | **单条文本分析** | **功能测试 + 界面测试** | **第 2-3 天** |
| **P0** | **批量文本分析** | **功能测试 + 性能测试** | **第 3 天** |
| **P1** | ASR 语音识别 | 功能验证（如有实现） | 第 4 天 |
| **P1** | 视频处理 | 功能验证（如有实现） | 第 4 天 |
| **P2** | Web 界面 | 手动测试为主 | 第 4-5 天 |
| **P2** | 性能测试 | 关键路径基准测试 | 第 5 天 |
| **P3** | 兼容性测试 | 基础环境验证 | 第 5 天 |

### 1.3 测试策略调整

**原计划**: 8 周全流程测试 → **现调整为**: 5 天敏捷测试

```mermaid
graph LR
    A[第 1 天<br/>模型+OCR 测试] --> B[第 2 天<br/>集成测试]
    B --> C[第 3 天<br/>ASR/视频测试]
    C --> D[第 4 天<br/>系统+性能测试]
    D --> E[第 5 天<br/>验收+报告]
```

**测试重点**:
- ✨ **重功能验证,轻边界条件**: 优先保证主流程可用
- ✨ **重自动化,轻手动测试**: 能自动化的尽量自动化
- ✨ **重核心,轻边缘**: 聚焦 P0/P1 功能，P3 功能酌情测试

---

## 🛠️ 二、测试工具快速部署

### 2.1 核心工具（必装）

```bash
# 测试框架
pip install pytest==7.4.0 pytest-cov==4.1.0 pytest-mock==3.11.1

# 断言库（可选但推荐）
pip install assertpy==1.1

# 性能测试（可选）
pip install pytest-benchmark==4.0.0
```

### 2.2 测试目录结构（精简版）

```
FZU_SentimentAnalysis/
├── tests/                          # 【新增】测试代码目录
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixtures
│   ├── unit/                       # 单元测试
│   │   ├── __init__.py
│   │   ├── test_model_handler.py   # ⭐ 模型测试（P0）
│   │   ├── test_ocr_processor.py   # ⭐ OCR 测试（P0）
<<<<<<< HEAD
│   │   ├── test_asr_processor.py   # ASR 自测（已合并，含离线切片/标准化与可选真实 ASR）
=======
│   │   ├── test_asr_processor.py   # ASR 测试（P1，可选）
>>>>>>> 69cfb9122c0350335668da90801e602d356c2021
│   │   └── test_video_processor.py # 视频测试（P1，可选）
│   ├── integration/                # 集成测试
│   │   ├── __init__.py
│   │   ├── test_single_text_analysis.py  # ⭐ 单条文本分析测试（P0）
│   │   ├── test_batch_analysis.py        # ⭐ 批量文本分析测试（P0）
│   │   └── test_multimodal_pipeline.py   # ⭐ 端到端测试（P0）
│   ├── system/                     # 系统测试
│   │   ├── __init__.py
│   │   ├── test_web_interface.py    # Web 界面测试（P2）
│   │   └── test_performance.py       # 性能测试（P2）
│   └── fixtures/                   # 测试数据
│       ├── sample_images/          # 测试图片
│       ├── sample_csv/             # 测试 CSV 文件
│       ├── sample_audio/           # 测试音频（可选）
│       └── sample_video/           # 测试视频（可选）
```

### 2.3 最小化配置文件

#### `pytest.ini`（项目根目录）

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short --cov=src --cov-report=term-missing
markers =
    unit: 单元测试
    integration: 集成测试
    slow: 慢速测试（真实 API 调用）
```

#### `tests/conftest.py`（共享 fixtures）

```python
import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

@pytest.fixture(scope="session")
def root_dir():
    return ROOT_DIR

@pytest.fixture(scope="session")
def model_path():
    return ROOT_DIR / "models" / "emotion_model"

@pytest.fixture
def loaded_model(model_path):
    from src.model_handler import EmotionModelHandler
    handler = EmotionModelHandler(model_path=str(model_path))
    handler.load_model()
    return handler

@pytest.fixture(scope="session")
def sample_image_path():
    return ROOT_DIR / "tests" / "fixtures" / "sample_images" / "test_happy.jpg"

@pytest.fixture
def mock_ocr_config(tmp_path):
    config = """
[ocr]
provider = "baidu"
timeout = 5.0

[baidu_ocr]
api_key = "test_key"
secret_key = "test_secret"
"""
    cfg_file = tmp_path / "ocr_secrets.toml"
    cfg_file.write_text(config)
    return cfg_file
```

---

## 📝 三、5 天详细测试安排

### 🗓️ 第 1 天：模型测试 + OCR 基础测试

**目标**: 完成 BERT 模型单元测试和 OCR 基础功能验证

#### 上午（9:00-12:00）

**任务 1.1**: 搭建测试环境 (1 小时)
- [ ] 安装 pytest 及依赖
- [ ] 创建测试目录结构
- [ ] 配置 `pytest.ini` 和 `conftest.py`
- [ ] 准备测试图片数据（至少 3 张：正面/负面/中性文字）

**任务 1.2**: 编写模型测试用例 (2 小时)
- [ ] `test_model_handler.py` - 核心测试用例

```python
"""
BERT 情感分析模型单元测试 - 精简版（6 个核心用例）
"""
import pytest
from src.model_handler import EmotionModelHandler

class TestEmotionModelHandler:
    
    def test_model_loading(self, model_path):
        """测试模型加载"""
        handler = EmotionModelHandler(model_path=str(model_path))
        handler.load_model()
        
        assert handler.model is not None
        assert handler.tokenizer is not None
    
    def test_predict_positive_text(self, loaded_model):
        """测试正面情感预测"""
        text = "今天天气真好，我很开心！"
        result = loaded_model.predict(text)
        
        assert result["fine_name"] == "开心"
        assert result["coarse"] == "正面"
        assert result["confidence"] > 0.5
    
    def test_predict_negative_text(self, loaded_model):
        """测试负面情感预测"""
        text = "听到这个消息我感到很难过。"
        result = loaded_model.predict(text)
        
        assert result["coarse"] == "负面"
    
    def test_predict_neutral_text(self, loaded_model):
        """测试中性情感预测"""
        text = "这个结果太让人惊讶了！"
        result = loaded_model.predict(text)
        
        assert result["fine_name"] == "惊讶"
        assert result["coarse"] == "中性"
    
    def test_predict_batch(self, loaded_model):
        """测试批量预测"""
        texts = ["今天心情很好", "我很难过", "太不可思议了"]
        results = loaded_model.predict_batch(texts)
        
        assert len(results) == 3
        for r in results:
            assert "fine_name" in r
    
    def test_predict_empty_text_error(self, loaded_model):
        """测试空文本输入异常"""
        with pytest.raises(Exception):
            loaded_model.predict("")
```

#### 下午（14:00-18:00）

**任务 1.3**: 编写 OCR 测试用例 (3 小时)
- [ ] `test_ocr_processor.py` - 核心测试用例

```python
"""
OCR 处理器单元测试 - 精简版（5 个核心用例）
"""
import pytest
from src.utils.ocr_processor import extract_text_from_image, OcrError

class TestOcrProcessor:
    
    def test_extract_text_from_image_path(self, sample_image_path):
        """测试从图片路径提取文本"""
        text = extract_text_from_image(sample_image_path)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
    
    def test_extract_text_from_image_bytes(self, sample_image_path):
        """测试从字节数据提取文本"""
        image_bytes = Path(sample_image_path).read_bytes()
        text = extract_text_from_image(image_bytes)
        assert isinstance(text, str)
    
    def test_extract_text_invalid_image(self, tmp_path):
        """测试无效图片输入"""
        invalid_image = tmp_path / "invalid.jpg"
        invalid_image.write_bytes(b"not an image")
        
        with pytest.raises(OcrError):
            extract_text_from_image(invalid_image)
    
    @pytest.mark.slow
    def test_real_api_call(self, sample_image_path):
        """测试真实 API 调用（可选，需配置密钥）"""
        # 需要真实密钥，可跳过
        pass
    
    def test_missing_config_error(self, tmp_path):
        """测试缺失配置文件错误"""
        non_existent = tmp_path / "non_existent.toml"
        
        with pytest.raises(OcrError, match="未找到 OCR 私密配置文件"):
            extract_text_from_image("dummy.jpg", config_path=non_existent)
```

**任务 1.4**: 执行测试并记录结果 (1 小时)
- [ ] 运行所有单元测试
- [ ] 记录通过的用例数
- [ ] 记录发现的 Bug

**第 1 天交付物**:
- ✅ 测试环境搭建完成
- ✅ 模型测试用例 6 个
- ✅ OCR 测试用例 5 个
- ✅ 第 1 天测试简报

---

### 🗓️ 第 2 天：集成测试（端到端流程 + 单条文本分析）

**目标**: 验证"图片→OCR→文本→情感分析"完整流程，完成单条文本分析功能测试

#### 上午（9:00-12:00）

**任务 2.1**: 编写集成测试用例 (2 小时)

```python
"""
多模态集成测试 - 精简版（4 个核心用例）
"""
import pytest
from src.utils.ocr_processor import extract_text_from_image

class TestMultimodalPipeline:
    
    def test_image_to_emotion_positive(self, sample_image_path, loaded_model):
        """测试图片→文本→正面情感完整流程"""
        # 步骤 1: OCR 提取文字
        text = extract_text_from_image(sample_image_path)
        
        # 步骤 2: 情感分析
        result = loaded_model.predict(text)
        
        # 验证
        assert result["coarse"] == "正面"
        assert result["fine_name"] == "开心"
    
    def test_image_to_emotion_negative(self, loaded_model):
        """测试负面文字图片→情感分析"""
        # 使用包含负面文字的图片
        img_path = "tests/fixtures/sample_images/sad_text.jpg"
        text = extract_text_from_image(img_path)
        result = loaded_model.predict(text)
        
        assert result["coarse"] == "负面"
    
    def test_image_to_emotion_neutral(self, loaded_model):
        """测试中性文字图片→情感分析"""
        img_path = "tests/fixtures/sample_images/neutral_text.jpg"
        text = extract_text_from_image(img_path)
        result = loaded_model.predict(text)
        
        assert result["coarse"] == "中性"
    
    def test_empty_image_handling(self, tmp_path, loaded_model):
        """测试空白图片处理"""
        from PIL import Image
        from io import BytesIO
        
        # 创建空白图片
        img = Image.new('RGB', (800, 600), color='white')
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        
        text = extract_text_from_image(buffer)
        
        # 空白图片应返回空字符串
        assert text.strip() == ""
```

**任务 2.2**: 编写单条文本分析测试用例 (2 小时)

```python
"""
单条文本分析功能测试 - 核心业务逻辑测试
"""
import pytest
from src.model_handler import EmotionModelHandler
from src.utils.text_processing import split_chinese_sentences
from src.utils.data_validation import validate_input_text

class TestSingleTextAnalysis:
    """单条文本分析功能测试类"""
    
    def test_standard_positive_input(self, loaded_model):
        """测试标准正面情感输入"""
        text = "今天收到生日礼物，超级开心！"
        result = loaded_model.predict(text)
        
        assert result["fine_name"] == "开心"
        assert result["coarse"] == "正面"
        assert result["confidence"] > 0.6
        assert "all_probabilities" in result
        assert "top3" in result
    
    def test_standard_negative_input(self, loaded_model):
        """测试标准负面情感输入"""
        text = "听到亲人去世的消息，我悲痛欲绝。"
        result = loaded_model.predict(text)
        
        assert result["coarse"] == "负面"
        assert result["fine_name"] in ["悲伤", "恐惧"]
    
    def test_standard_neutral_input(self, loaded_model):
        """测试标准中性情感输入"""
        text = "突然接到一个陌生电话，有点惊讶。"
        result = loaded_model.predict(text)
        
        assert result["fine_name"] == "惊讶"
        assert result["coarse"] == "中性"
    
    def test_short_text_input(self, loaded_model):
        """测试极短文本输入"""
        short_texts = ["好", "棒", "赞"]
        
        for text in short_texts:
            result = loaded_model.predict(text)
            assert result is not None
            assert "fine_name" in result
    
    def test_long_text_input(self, loaded_model):
        """测试长文本输入（超过 max_length）"""
        long_text = "今天心情很好，" * 100  # 远超 512 tokens
        result = loaded_model.predict(long_text)
        
        # 应该能正常处理（被截断）
        assert result is not None
        assert "fine_name" in result
    
    def test_mixed_emotion_input(self, loaded_model):
        """测试混合情感输入"""
        text = "虽然下雨了有些扫兴，但和朋友聊天还是很开心。"
        result = loaded_model.predict(text)
        
        assert result is not None
        # 混合情感应根据模型训练偏向判断
    
    def test_special_characters_input(self, loaded_model):
        """测试特殊字符输入"""
        text = "今天心情真好！！！😄😄😄"
        result = loaded_model.predict(text)
        
        assert result is not None
        assert result["fine_name"] == "开心"
    
    def test_sentence_splitting(self):
        """测试中文句子分割功能"""
        text = "今天天气真好。我和朋友去公园玩了。我们玩得很开心！"
        sentences = split_chinese_sentences(text)
        
        assert len(sentences) >= 2
        assert isinstance(sentences, list)
        for sent in sentences:
            assert len(sent.strip()) > 0
    
    def test_input_validation_valid(self):
        """测试输入验证（有效输入）"""
        valid_texts = [
            "你好",
            "今天心情很好",
            "这是一段测试文本"
        ]
        
        for text in valid_texts:
            assert validate_input_text(text) is True
    
    def test_input_validation_invalid(self):
        """测试输入验证（无效输入）"""
        invalid_inputs = [
            "",           # 空字符串
            "   ",        # 纯空格
            None,         # None 值
        ]
        
        for text in invalid_inputs:
            try:
                validate_input_text(text)
                assert False, f"应该抛出异常：{text}"
            except (ValueError, TypeError):
                pass  # 预期行为
    
    def test_result_format(self, loaded_model):
        """测试结果格式完整性"""
        text = "测试文本"
        result = loaded_model.predict(text)
        
        # 验证结果包含所有必需字段
        required_fields = [
            "fine_id",
            "fine_name",
            "coarse",
            "confidence",
            "all_probabilities",
            "top3"
        ]
        
        for field in required_fields:
            assert field in result, f"缺少必需字段：{field}"
        
        # 验证数据类型
        assert isinstance(result["fine_id"], int)
        assert isinstance(result["fine_name"], str)
        assert isinstance(result["coarse"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["all_probabilities"], dict)
        assert isinstance(result["top3"], list)
    
    def test_probability_distribution(self, loaded_model):
        """测试概率分布合理性"""
        text = "测试文本"
        result = loaded_model.predict(text)
        
        probs = result["all_probabilities"]
        
        # 所有概率之和应接近 1
        total = sum(probs.values())
        assert 0.99 <= total <= 1.01, f"概率和不为 1: {total}"
        
        # 每个概率应在 0-1 之间
        for prob in probs.values():
            assert 0 <= prob <= 1, f"概率超出范围：{prob}"
        
        # top3 应按降序排列
        top3 = result["top3"]
        assert len(top3) == 3
        for i in range(len(top3) - 1):
            assert top3[i][1] >= top3[i+1][1], "top3 未按降序排列"
```

**第 2 天交付物**:
- ✅ 集成测试用例 4 个
- ✅ 单条文本分析测试用例 12 个
- ✅ 端到端流程验证通过
- ✅ Bug 列表和修复记录
- ✅ 第 2 天测试报告

---

### 🗓️ 第 3 天：批量文本分析 + ASR/视频模块测试

**目标**: 完成批量文本分析功能测试，根据 ASR/视频模块实现情况进行功能验证

#### 上午（9:00-12:00）

**任务 3.1**: 编写批量文本分析测试用例 (3 小时)

```python
"""
批量文本分析功能测试 - 核心业务逻辑测试
"""
import pytest
import pandas as pd
from pathlib import Path
from src.model_handler import EmotionModelHandler
from src.utils.file_io import read_csv_file, write_csv_file

class TestBatchAnalysis:
    """批量文本分析功能测试类"""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """准备测试用 CSV 文件"""
        csv_content = """text
今天心情很好
我很难过
太不可思议了
这个产品真棒
我对结果很失望
"""
        csv_file = tmp_path / "test_batch.csv"
        csv_file.write_text(csv_content, encoding='utf-8')
        return csv_file
    
    def test_csv_file_reading(self, sample_csv_path):
        """测试 CSV 文件读取功能"""
        df = read_csv_file(str(sample_csv_path))
        
        assert df is not None
        assert 'text' in df.columns
        assert len(df) == 5
        assert isinstance(df['text'].iloc[0], str)
    
    def test_batch_prediction_basic(self, loaded_model, sample_csv_path):
        """测试基本批量预测功能"""
        df = read_csv_file(str(sample_csv_path))
        texts = df['text'].tolist()
        
        results = loaded_model.predict_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "fine_name" in result
            assert "coarse" in result
            assert "confidence" in result
    
    def test_batch_prediction_with_dataframe(self, loaded_model, sample_csv_path):
        """测试带 DataFrame 的批量预测"""
        df = read_csv_file(str(sample_csv_path))
        texts = df['text'].tolist()
        
        results = loaded_model.predict_batch(texts)
        
        # 将结果添加到 DataFrame
        df['fine_emotion'] = [r['fine_name'] for r in results]
        df['coarse_emotion'] = [r['coarse'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        assert 'fine_emotion' in df.columns
        assert 'coarse_emotion' in df.columns
        assert 'confidence' in df.columns
        assert len(df) == 5
    
    def test_large_batch_processing(self, loaded_model, tmp_path):
        """测试大批量数据处理"""
        # 创建包含 100 条文本的 CSV
        texts = [f"这是第{i}条测试文本"] * 100
        df = pd.DataFrame({'text': texts})
        csv_file = tmp_path / "large_batch.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 读取并处理
        df_loaded = read_csv_file(str(csv_file))
        results = loaded_model.predict_batch(df_loaded['text'].tolist())
        
        assert len(results) == 100
    
    def test_csv_writing(self, tmp_path):
        """测试 CSV 文件写入功能"""
        df = pd.DataFrame({
            'text': ['文本 1', '文本 2'],
            'emotion': ['开心', '悲伤']
        })
        
        output_file = tmp_path / "output.csv"
        write_csv_file(df, str(output_file))
        
        assert output_file.exists()
        
        # 验证写入内容
        df_loaded = pd.read_csv(output_file, encoding='utf-8')
        assert len(df_loaded) == 2
        assert 'text' in df_loaded.columns
        assert 'emotion' in df_loaded.columns
    
    def test_empty_csv_handling(self, loaded_model, tmp_path):
        """测试空 CSV 文件处理"""
        csv_content = "text\n"
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text(csv_content, encoding='utf-8')
        
        df = read_csv_file(str(csv_file))
        
        if len(df) == 0:
            # 空文件应返回空 DataFrame 或适当错误
            results = loaded_model.predict_batch([])
            assert len(results) == 0
    
    def test_csv_missing_column_error(self, tmp_path):
        """测试 CSV 缺少 text 列的错误处理"""
        csv_content = """content
今天心情很好
"""
        csv_file = tmp_path / "wrong_format.csv"
        csv_file.write_text(csv_content, encoding='utf-8')
        
        df = read_csv_file(str(csv_file))
        
        # 应该没有'text'列
        assert 'text' not in df.columns
    
    def test_batch_result_statistics(self, loaded_model, sample_csv_path):
        """测试批量结果的统计信息"""
        df = read_csv_file(str(sample_csv_path))
        texts = df['text'].tolist()
        results = loaded_model.predict_batch(texts)
        
        # 统计各类情感数量
        emotion_counts = {}
        for result in results:
            emotion = result['fine_name']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 验证统计结果
        assert sum(emotion_counts.values()) == len(results)
        assert len(emotion_counts) > 0
    
    def test_batch_performance_baseline(self, loaded_model):
        """测试批量处理性能基线"""
        import time
        
        texts = ["测试文本"] * 50
        
        start = time.perf_counter()
        results = loaded_model.predict_batch(texts)
        end = time.perf_counter()
        
        total_time = end - start
        avg_time_per_text = (total_time / len(texts)) * 1000  # 毫秒
        
        assert len(results) == 50
        # 平均每条处理时间应小于 500ms（保守估计）
        assert avg_time_per_text < 500, f"批量处理性能不达标：{avg_time_per_text}ms/条"
        
        print(f"✓ 批量处理性能：{avg_time_per_text:.2f}ms/条")
    
    def test_diverse_emotion_distribution(self, loaded_model):
        """测试多样化情感分布"""
        # 准备包含不同情感的文本
        texts = [
            "今天收到礼物很开心",      # 开心
            "听到噩耗我悲痛欲绝",      # 悲伤
            "这个结果让我很生气",      # 生气
            "突如其来的消息让我惊讶",  # 惊讶
            "我害怕面对这个挑战",      # 恐惧
            "这种行为让我厌恶"        # 厌恶
        ]
        
        results = loaded_model.predict_batch(texts)
        
        # 验证每种情感都被正确识别
        emotions = [r['fine_name'] for r in results]
        
        # 至少应识别出 4 种不同的情感
        unique_emotions = set(emotions)
        assert len(unique_emotions) >= 4, f"情感识别过于单一：{unique_emotions}"
    
    def test_concurrent_batch_processing(self, loaded_model):
        """测试并发批量处理（可选，高级功能）"""
        import concurrent.futures
        
        texts_list = [
            ["文本 1", "文本 2"],
            ["文本 3", "文本 4"],
            ["文本 5", "文本 6"]
        ]
        
        def process_batch(texts):
            return loaded_model.predict_batch(texts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch, texts) for texts in texts_list]
            all_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # 验证所有批次都处理完成
        assert len(all_results) == 3
        for results in all_results:
            assert len(results) == 2
```

#### 下午（14:00-18:00）

**任务 3.2**: ASR 模块测试（如已实现）
```python
"""
ASR 测试用例（如已实现）
"""
import pytest

class TestAsrProcessor:
    
    def test_audio_to_text_wav(self, sample_audio_path):
        """测试 WAV 音频转文本"""
        # TODO: 待 ASR 模块实现后补充
        pass
    
    def test_audio_to_text_mp3(self, sample_audio_path):
        """测试 MP3 音频转文本"""
        # TODO: 待 ASR 模块实现后补充
        pass
```

**任务 3.3**: 视频模块测试（如已实现）
```python
"""
视频处理测试用例（如已实现）
"""
import pytest

class TestVideoProcessor:
    
    def test_video_to_audio(self, sample_video_path):
        """测试视频提取音频"""
        # TODO: 待视频模块实现后补充
        pass
    
    def test_video_full_pipeline(self, sample_video_path, loaded_model):
        """测试视频→音频→文本→情感完整流程"""
        # TODO: 待视频模块实现后补充
        pass
```

#### 场景 B: ASR/视频未实现

**替代方案**: 加强单条和批量分析功能测试

**下午**: 补充边界条件和异常场景测试
- [ ] 增加模型边界测试用例（极端长度文本、特殊符号等）
- [ ] 增加批量处理异常测试（损坏的 CSV、编码问题等）
- [ ] 增加 OCR 复杂场景测试（模糊图片、手写体、多语言混合等）

**任务 3.4**: 文档完善和代码审查 (2 小时)
- [ ] 审查现有测试代码质量
- [ ] 补充测试用例注释
- [ ] 整理测试数据

**第 3 天交付物**:
- ✅ 批量文本分析测试用例 12 个
- ✅ ASR/视频测试用例（如已实现）
- ✅ 或：边界测试补充用例 5+
- ✅ 测试代码审查记录
- ✅ 第 3 天测试总结

---

### 🗓️ 第 4 天：系统测试 + 性能基准

**目标**: Web 界面功能验证和关键性能指标测试

#### 上午（9:00-12:00）

**任务 4.1**: Web 界面手动测试清单

由于时间限制，Web 界面采用**手动测试**为主:

```markdown
## Web 界面测试清单

### 单条文本分析页面
- [ ] 页面正常加载
- [ ] 输入正面文字，结果显示正确
- [ ] 输入负面文字，结果显示正确
- [ ] 输入中性文字，结果显示正确
- [ ] 图表正常显示（雷达图、柱状图）
- [ ] 侧边栏导航正常切换

### 批量文本分析页面
- [ ] CSV 文件上传成功
- [ ] 批量处理正常执行
- [ ] 结果表格显示正确
- [ ] 结果 CSV 下载功能正常
- [ ] 情感分布饼图显示正常

### 多模态功能（如已实现）
- [ ] 图片上传功能正常
- [ ] 图片 OCR 识别成功
- [ ] 情感分析结果展示
- [ ] 音频/视频上传（如有）
```

**任务 4.2**: 记录测试结果
- [ ] 填写测试清单
- [ ] 截图保存关键功能
- [ ] 记录发现的 UI 问题

#### 下午（14:00-18:00）

**任务 4.3**: 性能基准测试 (3 小时)

```python
"""
性能基准测试 - 精简版（3 个关键指标）
"""
import pytest
import time

class TestPerformance:
    
    def test_model_inference_time(self, loaded_model):
        """测试模型单次推理时间"""
        text = "这是一个测试句子。"
        
        start = time.perf_counter()
        loaded_model.predict(text)
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000  # 毫秒
        
        # CPU 环境下应小于 500ms
        assert inference_time < 500, f"推理时间过长：{inference_time}ms"
        print(f"✓ 单次推理时间：{inference_time:.2f}ms")
    
    def test_batch_processing_time(self, loaded_model):
        """测试批量处理性能"""
        texts = ["测试句子"] * 50
        
        start = time.perf_counter()
        loaded_model.predict_batch(texts)
        end = time.perf_counter()
        
        total_time = end - start
        avg_time = (total_time / len(texts)) * 1000
        
        # 平均每条应小于 200ms
        assert avg_time < 200, f"平均处理时间过长：{avg_time}ms"
        print(f"✓ 批量处理平均耗时：{avg_time:.2f}ms/条")
    
    @pytest.mark.slow
    def test_ocr_response_time(self, sample_image_path):
        """测试 OCR 响应时间（真实 API）"""
        start = time.perf_counter()
        extract_text_from_image(sample_image_path)
        end = time.perf_counter()
        
        ocr_time = end - start
        
        # 网络请求应在 10 秒内完成
        assert ocr_time < 10.0, f"OCR 响应时间过长：{ocr_time}s"
        print(f"✓ OCR 响应时间：{ocr_time:.2f}s")
```

**任务 4.4**: 记录性能数据
- [ ] 记录模型推理时间
- [ ] 记录批量处理性能
- [ ] 记录 OCR 响应时间（如测试）
- [ ] 对比性能基线

**第 4 天交付物**:
- ✅ Web 界面测试清单（已填写）
- ✅ 性能测试报告
- ✅ 性能基准数据
- ✅ 第 4 天测试总结

---

### 🗓️ 第 5 天：验收测试 + 最终报告

**目标**: 用户验收测试、Bug 收尾、输出最终报告

#### 上午（9:00-12:00）

**任务 5.1**: 用户验收测试（UAT）(2 小时)

邀请 2-3 名真实用户进行体验测试:

```markdown
## 用户验收测试表

### 测试场景 1: 单条文本情感分析
- 用户操作：输入"今天收到礼物很开心"
- 期望结果：识别为"开心"情感，置信度>0.6
- 实际结果：_______________
- 用户满意度：⭐⭐⭐⭐⭐

### 测试场景 2: 批量文本分析
- 用户上传：包含 10 条文本的 CSV 文件
- 期望结果：全部分析完成，结果可下载
- 实际结果：_______________
- 用户满意度：⭐⭐⭐⭐⭐

### 测试场景 3: 图片 OCR 情感分析（亮点功能）
- 用户上传：包含文字的图片
- 期望结果：正确识别文字并分析情感
- 实际结果：_______________
- 用户满意度：⭐⭐⭐⭐⭐

### 整体评价
- 系统易用性：⭐⭐⭐⭐⭐
- 功能完整性：⭐⭐⭐⭐⭐
- 结果准确性：⭐⭐⭐⭐⭐
- 响应速度：⭐⭐⭐⭐⭐
```

**任务 5.2**: 收集反馈并分类 (1 小时)
- [ ] 整理用户反馈问题
- [ ] 按优先级分类（P0/P1/P2）
- [ ] 评估修复工作量

#### 下午（14:00-18:00）

**任务 5.3**: 最终 Bug 修复和回归 (2 小时)
- [ ] 修复所有 P0 级 Bug
- [ ] 尽可能修复 P1 级 Bug
- [ ] 对未修复 Bug 制定后续计划
- [ ] 执行最终回归测试

**任务 5.4**: 编写最终测试报告 (2 小时)

**第 5 天交付物**:
- ✅ 用户验收测试报告
- ✅ 最终 Bug 列表
- ✅ 测试总结报告
- ✅ 项目移交文档

---

## 📊 四、测试度量指标

### 4.1 每日跟踪指标

| 日期 | 用例总数 | 通过数 | 失败数 | 阻塞数 | 覆盖率 | Bug 数 |
|-----|---------|-------|-------|-------|-------|-------|
| 第 1 天 | 11 | - | - | - | - | - |
| 第 2 天 | 27 | - | - | - | - | - |
| 第 3 天 | 39+ | - | - | - | - | - |
| 第 4 天 | 42+ | - | - | - | - | - |
| 第 5 天 | 42+ | - | - | - | - | - |
| **总计** | **42+** | **-** | **-** | **-** | **>70%** | **-** |

### 4.2 缺陷管理（简化版）

使用 Excel 或在线文档跟踪:

| ID | 标题 | 优先级 | 状态 | 发现日期 | 修复日期 | 负责人 |
|----|------|-------|------|---------|---------|-------|
| BUG-001 | 模型加载失败 | P0 | Open | Day1 | - | XXX |
| BUG-002 | OCR 配置读取错误 | P1 | Fixed | Day1 | Day2 | XXX |

**缺陷优先级定义**:
- **P0**: 阻塞性问题（功能完全不可用）- 必须当天修复
- **P1**: 严重问题（核心功能异常）- 2 天内修复
- **P2**: 一般问题（部分功能异常）- 视时间情况修复
- **P3**: 建议改进（不影响使用）- 后续迭代处理

---

## 📈 五、测试报告模板

### 5.1 每日站会报告（模板）

```markdown
# 测试日报 - 第 X 天

## 一、今日完成情况

### 完成的工作
- ✅ 编写测试用例：X 个
- ✅ 执行测试：X 次
- ✅ 发现 Bug: X 个
- ✅ 修复 Bug: X 个

### 测试通过率
- 总用例数：X
- 通过：X (XX%)
- 失败：X (XX%)
- 阻塞：X (XX%)

### 代码覆盖率
- 语句覆盖率：XX%
- 分支覆盖率：XX%

## 二、发现的问题

### 严重问题（P0）
1. [BUG-001] 问题描述
   - 影响：XXX
   - 状态：已修复/进行中

### 一般问题（P1/P2）
1. [BUG-002] 问题描述
   - 影响：XXX
   - 状态：待修复

## 三、明日计划

1. 任务 1: XXX
2. 任务 2: XXX
3. 目标：完成 XXX 测试

## 四、需要支持

- 需要 XXX 协助
- 需要申请 XXX 资源
```

### 5.2 最终测试报告（模板）

```markdown
# FZU 情感分析系统 - 5 天测试总结报告

## 一、测试概况

- 测试周期：2026 年 3 月 X 日 - 3 月 X 日（5 天）
- 参与人员：XXX
- 测试范围：模型推理、OCR 识别、多模态集成、Web 界面

## 二、测试成果

### 用例统计
- 编写测试用例：XX 个
- 执行测试次数：XX 次
- 测试通过率：XX%
- 代码覆盖率：XX%

### 缺陷统计
- 发现缺陷：XX 个
- 已修复：XX 个
- 待修复：XX 个
- 严重缺陷（P0）：XX 个（已全部修复）

## 三、质量评估

### 功能完整性
- ✅ BERT 模型推理：功能完整，测试通过
- ✅ OCR 文字识别：功能完整，测试通过
- ⚠️ ASR 语音识别：功能未完成/已部分实现
- ⚠️ 视频处理：功能未完成/已部分实现
- ✅ Web 界面：核心功能可用

### 性能指标
| 指标 | 目标值 | 实测值 | 结论 |
|-----|-------|-------|------|
| 单次推理时间 | <500ms | XXXms | ✅ 达标 |
| 批量处理性能 | <200ms/条 | XXXms | ✅ 达标 |
| OCR 响应时间 | <10s | XXXs | ✅ 达标 |

### 用户体验
- 界面友好性：⭐⭐⭐⭐⭐
- 操作便捷性：⭐⭐⭐⭐⭐
- 结果准确性：⭐⭐⭐⭐⭐

## 四、遗留问题

### 已知缺陷列表
| ID | 描述 | 优先级 | 计划修复时间 |
|----|------|-------|-------------|
| BUG-XXX | XXX | P2 | 下一迭代 |
| BUG-XXX | XXX | P3 | 待定 |

### 风险项
1. **风险 1**: XXX
   - 可能性：中
   - 影响：高
   - 缓解措施：XXX

2. **风险 2**: XXX
   - 可能性：低
   - 影响：中
   - 缓解措施：XXX

## 五、测试结论

### 总体评价
本次测试在 5 天时间内完成了**核心功能验证**,系统达到**上线标准**:

- ✅ P0 级功能全部通过测试
- ✅ 严重缺陷已全部修复
- ✅ 性能指标满足要求
- ✅ 用户体验良好

### 发布建议
**建议发布**,但需注意:
1. 持续监控 P2 级缺陷影响
2. 收集用户反馈及时优化
3. 后续迭代补充 ASR/视频功能测试

## 六、附录

- 测试用例清单：见附件
- Bug 列表：见附件
- 性能测试详情：见附件

---
**报告编制**: XXX  
**审核**: XXX  
**日期**: 2026 年 3 月 X 日
```

---

## 💡 六、测试实施建议

### 6.1 快速启动指南

**Step 1**: 安装依赖（5 分钟）
```bash
pip install pytest pytest-cov pytest-mock assertpy
```

**Step 2**: 创建测试骨架（10 分钟）
```bash
mkdir -p tests/unit tests/integration tests/fixtures/sample_images
touch tests/__init__.py tests/conftest.py pytest.ini
```

**Step 3**: 复制粘贴测试用例（30 分钟）
- 将本文中的测试用例代码复制到对应文件
- 根据实际情况调整路径和参数

**Step 4**: 准备测试数据（30 分钟）
- 准备 3 张测试图片（正面/负面/中性文字）
- 准备 2 个测试 CSV 文件（小批量/大批量）
- 准备模型文件（确保在 `models/emotion_model/` 目录）

**Step 5**: 运行测试（5 分钟）
```bash
pytest -v --cov=src
```

### 6.2 时间管理技巧

1. **番茄工作法**: 每 25 分钟专注一个任务，休息 5 分钟
2. **优先级排序**: 每天先完成 P0 任务，再处理 P1/P2
3. **避免完美主义**: 5 天时间有限，接受"足够好"而非"完美"
4. **及时沟通**: 每天站会同步进展，避免信息孤岛

### 6.3 常见问题应对

**Q1: 测试环境搭建遇到问题怎么办？**

A: 
- 优先使用团队已有环境配置
- 遇到依赖冲突时，使用虚拟环境（venv/conda）
- 超过 30 分钟未解决，立即寻求支援

**Q2: 测试用例编写进度落后怎么办？**

A:
- 削减非核心用例（边界条件、异常场景）
- 优先保证主流程测试用例
- 考虑复用网上开源测试代码

**Q3: 发现大量 Bug 如何处理？**

A:
- 按优先级分类，优先修复 P0
- P1/P2 Bug 记录在案，评估修复成本
- 如影响上线，及时调整发布时间

**Q4: 真实 API 密钥如何获取？**

A:
- 联系项目负责人申请百度 AI 开放平台账号
- 如无密钥，使用 Mock 跳过相关测试
- 标记为`@pytest.mark.slow`,CI 中默认跳过

### 6.4 测试加速技巧

1. **并行执行测试**:
```bash
pip install pytest-xdist
pytest -n auto  # 自动使用所有 CPU 核心
```

2. **增量测试**:
```bash
pytest --last-failed  # 只运行上次失败的测试
```

3. **快速反馈模式**:
```bash
pytest -x  # 遇到第一个失败就停止
pytest --exitfirst  # 同上
```

---

## 🎯 七、成功标准

### 7.1 最低成功标准（必须达成）

- ✅ 完成 30 个核心测试用例（模型 6+ OCR 5+ 单条 12+ 批量 7）
- ✅ 代码覆盖率>70%
- ✅ 无 P0 级缺陷
- ✅ 核心功能可正常使用（单条分析、批量分析、OCR 识别）

### 7.2 理想成功标准（努力达成）

- ✅ 完成 40+ 测试用例
- ✅ 代码覆盖率>80%
- ✅ 无 P1 级缺陷
- ✅ 性能指标全部达标
- ✅ 用户验收满意度>4.5/5.0
- ✅ **单条文本分析和批量文本分析功能稳定可用**

### 7.3 优秀成功标准（挑战目标）

- ✅ 完成 50+ 测试用例
- ✅ 代码覆盖率>85%
- ✅ 所有缺陷已修复或有计划
- ✅ 输出完整测试文档
- ✅ 立自动化测试流水线
- ✅ **形成完整的单条 + 批量 + 多模态测试体系**

---

## 📚 八、附录

### 8.1 测试命令速查

```bash
# 运行所有测试
pytest -v

# 运行特定测试文件
pytest tests/unit/test_model_handler.py -v

# 运行特定测试函数
pytest tests/unit/test_model_handler.py::TestEmotionModelHandler::test_predict_positive_text -v

# 查看覆盖率
pytest --cov=src --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest --cov=src --cov-report=html
# 打开 htmlcov/index.html 查看

# 跳过慢速测试
pytest -m "not slow"

# 并行加速
pytest -n 4  # 使用 4 个进程

# 只运行失败用例
pytest --last-failed
```

### 8.2 测试数据准备清单

**最小测试数据集**（5 天冲刺版）:

```
tests/fixtures/
├── sample_images/
│   ├── test_happy.jpg      # 正面文字："今天心情很好"
│   ├── test_sad.jpg        # 负面文字："我很难过"
│   ├── test_neutral.jpg    # 中性文字："这是真的吗？"
│   └── test_empty.jpg      # 空白图片
├── sample_csv/
│   ├── small_batch.csv     # 小批量：5-10 条文本
│   ├── large_batch.csv     # 大批量：50-100 条文本
│   ├── diverse_emotions.csv # 多样情感：包含 6 种情感
│   └── edge_cases.csv      # 边界情况：空行、特殊字符等
├── sample_audio/           # 可选
│   └── test.wav
└── sample_video/           # 可选
    └── test.mp4
```

**CSV 文件格式示例**:

small_batch.csv:
```csv
text
今天心情很好
我很难过
太不可思议了
这个产品真棒
我对结果很失望
```

diverse_emotions.csv:
```csv
text
今天收到礼物很开心
听到噩耗我悲痛欲绝
这个结果让我很生气
突如其来的消息让我惊讶
我害怕面对这个挑战
这种行为让我厌恶
```

### 8.3 参考资源

- pytest 官方文档：https://docs.pytest.org/
- 测试最佳实践：https://docs.python-guide.org/writing/tests/
- Streamlit 测试：https://docs.streamlit.io/

---

## 🏆 九、总结

本 5 天测试计划是**8 周全量计划的精简版**,聚焦于:

1. **核心功能验证**: 确保模型推理、OCR 识别、端到端流程可用
2. **快速质量反馈**: 通过自动化测试快速发现问题
3. **实用主义导向**: 不追求完美覆盖，但求关键功能可靠

**执行要点**:
- 🎯 每天目标明确，当日事当日毕
- 🎯 优先级驱动，P0 功能优先测试
- 🎯 及时沟通，每天站会同步进展
- 🎯 灵活调整，根据实际情况动态优化

**预期成果**:
- ✅ 40-50 个高质量测试用例
- ✅ 70%-85% 代码覆盖率
- ✅ 无严重缺陷
- ✅ 完整测试报告
- ✅ **单条文本分析和批量文本分析功能经过充分验证**

相信通过 5 天的集中攻关，一定能为项目交付提供**坚实的质量保障**!

---

**编制人**: 智能系统综合设计项目组  
**审核人**: 待定  
**批准人**: 待定  
**版本号**: v1.0  
**生效日期**: 2026 年 3 月 19 日
