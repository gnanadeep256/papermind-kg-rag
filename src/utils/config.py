import os
from typing import Any, Dict
import yaml

class ConfigError(Exception):
    """Base exception class for configuration errors."""
    pass

class ConfigNotFoundError(ConfigError):
    """Exception raised when the configuration file is not found."""
    pass

class ConfigParseError(ConfigError):
    """Exception raised when the configuration file contains invalid YAML."""
    pass

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads and validates the YAML configuration file.

    Args:
        config_path: Path to the configuration file. Defaults to "configs/config.yaml".

    Returns:
        A dictionary containing the configuration values.

    Raises:
        ConfigNotFoundError: If the configuration file does not exist.
        ConfigParseError: If the configuration file cannot be parsed as valid YAML.
    """
    if not os.path.exists(config_path):
        raise ConfigNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ConfigError(f"An unexpected error occurred while loading configuration: {e}")

    if config_data is None:
        raise ConfigParseError(f"Configuration file at {config_path} is empty")
        
    if not isinstance(config_data, dict):
        raise ConfigParseError(f"Configuration file at {config_path} must be a dictionary")
        
    return config_data

