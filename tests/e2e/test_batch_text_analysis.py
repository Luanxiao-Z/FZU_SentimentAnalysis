import pytest
from streamlit.testing.v1 import AppTest
import pandas as pd

def test_batch_analysis_csv():
    # 准备: 启动Streamlit应用
    at = AppTest.from_file("app.py")
    at.run()

    # 模拟上传一个包含情感句子的CSV文件
    csv_data = "text\n今天是个好日子！\n服务太差了，非常生气。\n天气一般，没什么感觉。"
    at.file_uploader[0].upload(csv_data.encode()).run()

    # 执行: 点击“开始批量分析”按钮
    at.button[0].click().run()

    # 断言: 验证结果表格中是否包含预期的情感
    result_df = at.dataframe[0].value  # 假设结果表格是第一个dataframe元素
    assert '开心' in result_df['fine_emotion'].values
    assert '生气' in result_df['fine_emotion'].values
    assert '中性' in result_df['coarse_emotion'].values