# FZU_SentimentAnalysis

## 基于 BERT 的细粒度情感分析系统

### 项目简介
项目目标：构建一个基于 BERT-base-Chinese 的细粒度情感分析系统，实现对中文文本中 6 种基础情感的自动识别与可视化展示。

### 项目开发范围
1. 基于 BERT-base-Chinese 模型的 6 类细粒度情感分类
2. 支持单条文本输入的实时情感分析
3. 支持 CSV 文件的批量情感分析
4. 情感分布统计与可视化（饼图、柱状图、雷达图等）
5. 模型训练与评估功能

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
├── app.py                  # Streamlit 入口文件（仅包含页面配置和侧边栏导航）
├── pages/                  # Streamlit 多页面目录
│   ├── 1_单条文本分析.py    # 单条文本分析页面
│   ├── 2_批量文本分析.py    # 批量文本分析页面
│   └── 3_关于系统.py        # 关于系统页面
├── src/                    # 核心业务逻辑（与 Web 框架解耦）
│   ├── __init__.py         # 包初始化文件
│   ├── config.py           # 配置文件：模型路径、超参数等
│   ├── model_handler.py    # 模型操作类：加载、预测逻辑
│   └── utils.py            # 工具函数：数据处理、CSS 加载等
├── assets/                 # 静态资源
│   └── style.css           # 网页样式表
├── models/                 # 模型存储目录
│   └── emotion_model/      # 训练好的 BERT 模型文件
├── data/                   # 数据文件
│   └── dataHandler/        # 数据处理脚本
│       ├── convert.py
│       ├── merge.py
│       ├── process_emotions.py
│       └── translate_csv_baidu.py
├── scripts/                # 独立脚本
│   └── train_emotion.py    # 模型训练脚本
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明文档
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

### 3. 准备模型
将训练好的模型文件放置在 `models/emotion_model/` 目录下，应包含：
- `pytorch_model.bin` - 模型权重文件
- `config.json` - 模型配置文件
- `emotion_config.json` - 情感标签配置
- tokenizer 相关文件

### 4. 运行应用
```bash
streamlit run app.py
```

### 5. 功能说明

#### 单条文本分析
- 实时输入中文文本
- 自动识别 6 种细粒度情感（开心、悲伤、生气、惊讶、恐惧、厌恶）
- 映射到 3 种粗粒度情感（正面、负面、中性）
- 可视化展示：雷达图、柱状图显示各情感概率分布

#### 批量文本分析
- 上传包含 `text` 列的 CSV 文件
- 自动批量分析所有文本的情感
- 生成细粒度和粗粒度情感分布饼图
- 导出分析结果为 CSV 文件

#### 关于系统
- 查看 6 种细粒度情感的详细说明
- 了解粗粒度情感映射关系
- 技术栈介绍

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
