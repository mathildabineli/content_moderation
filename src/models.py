"""Pydantic models for the Content Moderation API.

This module defines data schemas used for request validation
and structured responses in prediction endpoints.
"""
from typing import List, Dict
from pydantic import BaseModel

class PredictInput(BaseModel):
    """Input schema for text moderation prediction.

    Contains a list of text samples to classify.
    """
    texts: List[str]

class PredictResponse(BaseModel):
    """Response schema for moderation prediction results.

    Includes labels, predictions, latency, and summary result.
    """
    labels: List[str]
    preds: List[int]
    latency_ms: float
    num_chunks: int
    result: Dict

__all__ = ["PredictInput", "PredictResponse"]
