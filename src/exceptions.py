"""Custom exception hierarchy for the Content Moderation API.

Defines structured application errors for training and prediction
phases with consistent HTTP status codes and retry behavior.
"""
from fastapi import status

class AppError(Exception):
    """Base class for all application-level exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "APP_ERROR"
    retryable = False
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

# ---- train ----
class TrainingError(AppError):
    """Raised for unexpected failures during model training."""
    code = "TRAINING_ERROR"

class TrainingResourceError(AppError):
    """Raised when a required resource for training is unavailable."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "TRAINING_RESOURCE_ERROR"
    retryable = True

# ---- predict ----
class InvalidInputError(AppError):
    """Raised when incoming prediction data fails validation."""
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "INVALID_INPUT"

class ArtifactNotFoundError(AppError):
    """Raised when a required model or artifact file is missing."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "ARTIFACT_MISSING"
    retryable = True

class ModelInitializationError(AppError):
    """Raised when the model cannot be initialized properly."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "MODEL_NOT_READY"
    retryable = True

class PredictionError(AppError):
    """Raised for generic prediction or inference failures."""
    code = "PREDICTION_ERROR"

class SchemaMismatchError(AppError):
    """Raised when prediction output shape differs from expectations."""
    code = "SCHEMA_MISMATCH"

class ConfigError(AppError):
    """Raised for malformed or invalid configuration files."""
    code = "CONFIG_ERROR"
