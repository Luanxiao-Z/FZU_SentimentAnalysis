import pytest
from streamlit.testing.v1 import AppTest

def test_single_text_analysis_positive():
    # 准备: 启动Streamlit应用
    at = AppTest.from_file("app.py")
    at.run()

    # 执行: 模拟用户输入和点击按钮操作
    at.text_input[0].input("今天天气真好，心情非常棒！").run()
    at.button[0].click().run()  # 点击“开始分析”按钮

    # 断言: 验证输出结果是否包含预期的关键字
    assert at.success[0].value == "分析完成。"
    assert any('开心' in str(element.value) for element in at.markdown)

def test_single_text_analysis_negative():
    # 准备: 启动Streamlit应用
    at = AppTest.from_file("app.py")
    at.run()

    # 执行: 模拟用户输入和点击按钮操作
    at.text_input[0].input("这服务太差了，气死我了！").run()
    at.button[0].click().run()  # 点击“开始分析”按钮

    # 断言: 验证输出结果是否包含预期的关键字
    assert at.success[0].value == "分析完成。"
    assert any('生气' in str(element.value) for element in at.markdown)