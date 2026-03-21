"""
Shared pytest fixtures for the test suite.
"""

import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]

# 将项目根目录添加到 Python 路径，以便导入 src 模块
sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    """Add custom markers to pytest."""
    config.addinivalue_line(
        "markers", "unit: 单元测试（快速，隔离）"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试（较慢，使用外部资源）"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试（例如真实API调用）"
    )

def pytest_collection_modifyitems(items):
    """Add 'unit' marker to all tests in unit/ directory."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

@pytest.fixture(scope="session")
def root_dir():
    return ROOT_DIR

@pytest.fixture(scope="session")
def model_path(root_dir):
    return root_dir / "models" / "emotion_model"

@pytest.fixture()
def loaded_model(model_path):
    from src.model_handler import EmotionModelHandler
    handler = EmotionModelHandler(model_path=str(model_path))
    handler.load_model()
    return handler

@pytest.fixture(scope="session")
def sample_image_path(root_dir):
    return root_dir / "tests" / "fixtures" / "sample_images" / "test_happy.jpg"

@pytest.fixture(scope="session")
def ocr_config_path(root_dir):
    """提供一个用于测试的OCR配置文件路径。"""
    # 可以创建一个临时的或预设的测试配置
    return root_dir / "config" / "ocr_secrets.example.toml"

@pytest.fixture(scope="session")
def asr_config_path(root_dir):
    """提供一个用于测试的ASR配置文件路径。"""
    return root_dir / "config" / "asr_secrets.example.toml"


@pytest.fixture(scope="session")
def sample_audio_path(root_dir):
    """提供一个用于测试的音频文件路径。"""
    return root_dir / "tests" / "fixtures" / "sample_audio" / "test.wav"