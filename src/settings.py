"""Application settings configuration module.

Defines the Settings class that manages environment-based
configuration using pydantic-settings.
"""
from pydantic_settings import (
    BaseSettings,  # pydantic-settings v2
    )


class Settings(BaseSettings):
    """Base configuration class for environment variables.

    Extend this class to define app-specific settings.
    """
    