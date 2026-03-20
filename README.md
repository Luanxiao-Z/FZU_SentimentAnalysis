# FZU_SentimentAnalysis

## 基于BERT的细粒度情感分析系统

### 项目简介
本项目是一个基于深度学习的多模态细粒度情感分析系统，旨在实现对中文文本、文档、图片、音频和视频等多种模态内容的情感识别。系统采用 BERT 模型进行核心分析，并提供直观的可视化界面，支持单条文本、批量文件及多模态数据的处理与分析。

### 项目开发范围
1. 基于 BERT-base-Chinese 模型的 6 类细粒度情感分类
2. 支持单条文本输入的实时情感分析
3. 支持 CSV等多格式文件的批量情感分析
4. 情感分布统计与可视化（饼图、柱状图、雷达图等）
5. 模型训练与评估功能
6. 多模态分析功能

### 技术栈
- **模型层**：BERT-base-Chinese（Hugging Face Transformers）
- **深度学习框架**：PyTorch 2.0+
- **Web 界面**：Streamlit（快速构建数据应用）
- **可视化**：Plotly（交互式图表）
- **数据处理**：Pandas、NumPy、Scikit-learn
- **版本控制**：Git + GitHub

---

## 项目结构

```
FZU_SentimentAnalysis/
├── app.py                      # Streamlit 入口文件（仅包含页面配置和侧边栏导航）
├── .streamlit/                 # Streamlit 配置目录
│   └── config.toml            # Streamlit 运行配置
├── config/                   # 配置目录（包含私密配置模板）
│   └── ocr_secrets.example.toml # OCR 私密配置模板（复制为 ocr_secrets.toml 后填写真实密钥；真实文件不提交）
├── pages/                    # Streamlit 多页面目录
│   ├── 1_单条文本分析.py      # 单条文本分析页面
│   ├── 2_批量文本分析.py      # 批量文本分析页面
│   ├── 3_关于系统.py          # 关于系统页面
│   └── 4_多模态分析.py         # 多模态分析页面，支持图片、音频、视频等多模态数据处理
├── src/                      # 核心业务逻辑（与 Web 框架解耦）
│   ├── __init__.py           # 包初始化文件
│   ├── config.py             # 配置文件：模型路径、超参数等
│   ├── model_handler.py      # 模型操作类：加载、预测逻辑
│   └── utils/                # 工具函数模块（按功能拆分）
│       ├── __init__.py       # 工具函数包初始化
│       ├── asr_processor.py  # 语音识别处理器
│       ├── data_validation.py # 数据验证和格式化工具
│       ├── document_processor.py # 文档处理工具：文档转句子 DataFrame
│       ├── emotion_utils.py  # 情感标签工具函数
│       ├── file_io.py        # 文件 I/O 工具：CSV、Excel、PDF、DOCX、Markdown、TXT 读取
│       ├── ocr_processor.py  # OCR 图像文字识别处理器
│       ├── text_processing.py # 文本处理工具：句子分割
│       └── video_processor.py # 视频处理器
├── assets/                   # 静态资源
│   └── style.css             # 网页样式表
├── models/                   # 模型存储目录
│   └── emotion_model/        # 训练好的 BERT 模型文件
├── data/                     # 数据文件
│   ├── dataHandler/          # 数据处理脚本
│   │   ├── balance_distribution.py
│   │   ├── convert.py
│   │   ├── llm_expand.py
│   │   ├── merge.py
│   │   ├── process_emotions.py
│   │   └── translate_csv_baidu.py
│   ├── errorPreviewData/     # 错误案例预览数据
│   │   └── hard_cases.csv
│   ├── originalData/         # 原始数据集
│   │   ├── Goemotions.csv
│   │   ├── NLPcc2013-2014微博文本情感分类数据集.csv
│   │   └── OCEMOTION.csv
│   ├── data.csv
│   ├── data_balanced.csv
│   ├── data_expand.csv
│   └── result.csv
├── docs/                     # 项目文档
│   ├── Git 基础入门教程：从克隆到推送.md
│   ├── 指标数据02.md
│   └── 指标记录01.md
├── scripts/                  # 独立脚本
│   ├── check_label_balance.py # 检查标签平衡性脚本
│   └── train_emotion.py      # 模型训练脚本
├── requirements.txt          # 依赖包列表
└── README.md                 # 项目说明文档
```

---

## 安装与使用

### 1. 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+（可选，用于 GPU 加速）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 说明
本项目依赖以下主要库：
- **Web 框架**: Streamlit (>=1.28.0)
- **深度学习**: PyTorch (>=2.0.0), Transformers (>=4.30.0)
- **数据处理**: Pandas (>=2.0.0), Openpyxl (>=3.0.0)
- **可视化**: Plotly (>=5.14.0)
- **文档处理**: pdfplumber (>=0.10.0), python-docx (>=1.0.0), markdown (>=3.4.0)
- **图像/音频处理**: Pillow (>=10.2.0)

### 2.1 OCR（可选）私密配置

如果你要使用 OCR 能力（`src/utils/ocr_processor.py`），请：

1. 复制模板文件：

```bash
copy config\\ocr_secrets.example.toml config\\ocr_secrets.toml
```

2. 打开 `config/ocr_secrets.toml` 填入你的密钥。

说明：`config/ocr_secrets.toml` 已加入 `.gitignore`，不会被提交到仓库。

### 3. 准备模型
将训练好的 BERT 模型文件放置在 `models/emotion_model/` 目录下。该目录通常由训练脚本自动生成，包含以下文件：
- `model.safetensors` - 模型权重文件（安全张量格式）
- `config.json` - 模型配置文件
- `emotion_config.json` - 情感标签映射配置
- `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` 等 - 分词器相关文件
- `test_metrics.json` - 测试集评估指标

### 4. 运行应用
```bash
streamlit run app.py
```

### 5. 功能说明

#### 单条文本分析
- 支持实时输入中文文本进行情感分析
- 自动识别 6 种细粒度情感（开心、悲伤、生气、惊讶、恐惧、厌恶）
- 情感结果自动映射到 3 种粗粒度情感（正面、负面、中性）
- 提供多维度可视化展示，包括雷达图、柱状图等，直观呈现各情感概率分布

#### 批量文本分析
- 支持上传多种格式文件（CSV、Excel、PDF、TXT、DOCX、MD）进行批量情感分析
- 自动提取文档内容并拆分为句子进行逐条分析
- 生成细粒度和粗粒度情感分布的饼图统计
- 提供详细结果表格，支持导出为 CSV 文件

#### 多模态分析
- 支持上传图片、音频、视频及文档等多模态文件
- 图片可自动进行 OCR 文字识别，音频可进行语音转写
- 对提取的文字内容进行情感分析
- 提供全面的可视化展示，包括饼图、雷达图和柱状图
- 支持导出完整的分析结果

#### 关于系统
- 查看 6 种细粒度情感的详细说明与示例
- 了解细粒度到粗粒度情感的映射关系
- 技术栈介绍：BERT-base-Chinese 模型、PyTorch、Transformers、Streamlit、Plotly

---

## 情感分类体系

### 细粒度情感（6 类）
- 😊 **开心** - 积极、愉悦、满足
- 😢 **悲伤** - 失落、痛苦、泪坠
- 😠 **生气** - 愤怒、不满、恼火
- 😲 **惊讶** - 意外、震惊、诧异
- 😨 **恐惧** - 害怕、担心、焦虑
- 🤢 **厌恶** - 反感、鄙视、嫌弃

### 粗粒度情感（3 类）
- **正面** - 开心
- **负面** - 悲伤、生气、恐惧、厌恶
- **中性** - 惊讶

---

## 数据格式

### CSV 文件格式
CSV 文件需包含 `text` 列，示例：
```csv
text
今天天气真好，心情很愉快！
听到这个消息我感到很难过。
这个结果太让人惊讶了！
```

### 批量分析输出
分析结果包含以下字段：
- `text` - 原始文本
- `fine_emotion` - 细粒度情感标签
- `coarse_emotion` - 粗粒度情感标签
- `confidence` - 置信度
- `prob_*` - 各情感类别的概率

---

## 开发团队

© 2026 智能系统综合设计 | 细粒度情感分析系统

---

## License

MIT License
