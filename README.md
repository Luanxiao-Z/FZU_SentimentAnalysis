# FZU_SentimentAnalysis
- main分支为正式项目代码
- Demo分支为项目可行性分析制作的原型项目代码
## 基于BERT的细粒度情感分析系统
项目目标：
构建一个基于BERT-base-Chinese的细粒度情感分析系统，实现对中文文本中6种基础情感的自动识别与可视化展示。
项目开发范围：
1. 基于BERT-base-Chinese模型的6类细粒度情感。
2. 分类支持单条文本输入的实时情感分析。
3. 支持CSV/Excel文件的批量情感分析。
4. 情感分布统计与可视化（饼图、柱状图、热力图等）。
5. 模型训练与评估功能。
- 模型层：BERT-base-Chinese（Hugging Face Transformers）
- 深度学习框架：PyTorch 2.0+
- Web界面：Streamlit（快速构建数据应用）
- 可视化：Plotly（交互式图表）
- 数据处理：Pandas、NumPy、Scikit-learn
- 版本控制：Git + GitHub
