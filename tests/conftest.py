import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

@ pytest.fixture(scope="session")
def root_dir():
    return ROOT_DIR

@ pytest.fixture(scope="session")
def model_path():
    return ROOT_DIR / "models" / "emotion_model"

@ pytest.fixture
def loaded_model(model_path):
    from src.model_handler import EmotionModelHandler
    handler = EmotionModelHandler(model_path=str(model_path))
    handler.load_model()
    return handler

@ pytest.fixture(scope="session")
def sample_image_path():
    return ROOT_DIR / "tests" / "fixtures" / "sample_images" / "test_happy.jpg"

@ pytest.fixture
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