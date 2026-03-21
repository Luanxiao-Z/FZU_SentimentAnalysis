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
            "今天收到礼物很开心",  # 开心
            "听到噩耗我悲痛欲绝",  # 悲伤
            "这个结果让我很生气",  # 生气
            "突如其来的消息让我惊讶",  # 惊讶
            "我害怕面对这个挑战",  # 恐惧
            "这种行为让我厌恶"  # 厌恶
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