"""
测试 EmotionModelHandler 类的单元测试。
"""

import pytest
from unittest.mock import Mock, patch
from src.model_handler import EmotionModelHandler

def test_model_loading(model_path):
    """测试模型加载功能。"""
    handler = EmotionModelHandler(model_path=str(model_path))
    handler.load_model()

    assert handler.model is not None
    assert handler.tokenizer is not None

def test_predict_positive_text(loaded_model):
    """测试正面情感预测。"""
    text = "今天天气真好，我很开心！"
    result = loaded_model.predict(text)

    assert result["fine_name"] == "开心"
    assert result["coarse"] == "正面"
    assert result["confidence"] > 0.5

def test_predict_negative_text(loaded_model):
    """测试负面情感预测。"""
    text = "听到这个消息我感到很难过。"
    result = loaded_model.predict(text)

    assert result["coarse"] == "负面"

def test_predict_neutral_text(loaded_model):
    """测试中性情感预测。"""
    text = "这个结果太让人惊讶了！"
    result = loaded_model.predict(text)

    assert result["fine_name"] == "惊讶"
    assert result["coarse"] == "中性"

def test_predict_batch(loaded_model):
    """测试批量预测功能。"""
    texts = ["今天心情很好", "我很难过", "太不可思议了"]
    results = loaded_model.predict_batch(texts)

    assert len(results) == 3
    for r in results:
        assert "fine_name" in r

def test_predict_empty_text_error(loaded_model):
    """测试空文本输入异常处理。"""
    with pytest.raises(ValueError):
        loaded_model.predict("")