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
            "",  # 空字符串
            "   ",  # 纯空格
            None,  # None 值
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
            assert top3[i][1] >= top3[i + 1][1], "top3 未按降序排列"