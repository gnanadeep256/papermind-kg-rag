import pytest
import os
from src.utils.config import load_config, ConfigNotFoundError, ConfigParseError

def test_load_config_success():
    """Test loading the default configuration file successfully."""
    # Ensure config path points to valid test setup
    config = load_config("configs/config.yaml")
    assert isinstance(config, dict)
    assert "project" in config
    assert config["project"]["name"] == "papermind-kg-rag"
    assert "arxiv" in config
    assert "categories" in config["arxiv"]

def test_load_config_missing_file():
    """Test loading a non-existent configuration file raises ConfigNotFoundError."""
    with pytest.raises(ConfigNotFoundError) as exc_info:
        load_config("configs/does_not_exist.yaml")
    assert "Configuration file not found at:" in str(exc_info.value)

def test_load_config_invalid_yaml(tmp_path):
    """Test loading an invalid YAML file raises ConfigParseError."""
    bad_config_file = tmp_path / "bad_config.yaml"
    with open(bad_config_file, "w", encoding="utf-8") as f:
        # Invalid YAML format
        f.write("project:\n  name: : invalid : colons")
        
    with pytest.raises(ConfigParseError) as exc_info:
        load_config(str(bad_config_file))
    assert "Failed to parse YAML configuration" in str(exc_info.value)

def test_load_config_empty_file(tmp_path):
    """Test loading an empty configuration file raises ConfigParseError."""
    empty_config_file = tmp_path / "empty_config.yaml"
    empty_config_file.touch()
    
    with pytest.raises(ConfigParseError) as exc_info:
        load_config(str(empty_config_file))
    assert "is empty" in str(exc_info.value)
