"""
Configuration loader for Mini OCR project.

This module provides utilities to load and access configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for loading and accessing YAML config files.

    Example:
        >>> config = Config.load("config.yaml")
        >>> print(config.get("model.name"))
        'unsloth/DeepSeek-OCR'
        >>> print(config.training.batch_size)
        2
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config object with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return cls(config_dict)

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "model.name")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("training.lora.rank")
            16
            >>> config.get("nonexistent.key", default=0)
            0
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getattr__(self, name: str) -> Any:
        """
        Access config sections as attributes.

        Args:
            name: Section name

        Returns:
            ConfigSection object or value

        Example:
            >>> config.model.name
            'unsloth/DeepSeek-OCR'
        """
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()


class ConfigSection:
    """
    Configuration section wrapper for dict-like access.

    Allows accessing nested config values as attributes.
    """

    def __init__(self, section_dict: Dict[str, Any]):
        """
        Initialize config section.

        Args:
            section_dict: Dictionary for this section
        """
        self._section = section_dict

    def __getattr__(self, name: str) -> Any:
        """
        Access section values as attributes.

        Args:
            name: Key name

        Returns:
            Value or nested ConfigSection
        """
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._section:
            value = self._section[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"Section has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default.

        Args:
            key: Key name
            default: Default value if key not found

        Returns:
            Value or default
        """
        value = self._section.get(key, default)
        if isinstance(value, dict):
            return ConfigSection(value)
        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert section to dictionary.

        Returns:
            Section dictionary
        """
        return self._section.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to config file, defaults to "config.yaml"

    Returns:
        Loaded Config object
    """
    if config_path is None:
        config_path = "config.yaml"
    return Config.load(config_path)


if __name__ == "__main__":
    # Example usage
    import sys

    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"\nModel name: {config.model.name}")
        print(f"Training batch size: {config.training.batch_size}")
        print(f"LoRA rank: {config.training.lora.rank}")
        print(f"Image size: {config.data_collator.image_size}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
