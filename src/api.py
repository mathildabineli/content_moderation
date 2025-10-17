"""Content Moderation API — production with logging."""
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.inference import ClassifierService
from src.models import PredictInput, PredictResponse
from src.train_teacher import train_teacher_step
from src.utils import labels_with_pred1

from src.config import (
    ONNX_PATH, TOKENIZER_PATH, METADATA_PATH, THRESHOLDS_PATH,
    MAX_TEXTS, MAX_TEXT_LEN,
)

from src.exceptions import (
    AppError, TrainingError, TrainingResourceError,
    InvalidInputError, ArtifactNotFoundError, ModelInitializationError,
    PredictionError, SchemaMismatchError, ConfigError,
)

# ──────────────── Logging setup (minimal, prod-friendly) ────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("content_moderation.api")

log.info(
    (
        "API starting with LOG_LEVEL=%s | "
        "ONNX=%s | TOKENIZER=%s | META=%s | THR=%s | "
        "MAX_TEXTS=%d | MAX_TEXT_LEN=%d"
    ),
    LOG_LEVEL,
    ONNX_PATH,
    TOKENIZER_PATH,
    METADATA_PATH,
    THRESHOLDS_PATH,
    MAX_TEXTS,
    MAX_TEXT_LEN,
)


app = FastAPI(
    title="Content Moderation API",
    version="1.0.0",
    description=(
        "🚦 **Content Moderation API**\n\n"
        "This API provides endpoints for training and serving multilingual "
        "text classification model used for safe content moderation.\n\n"
        "**Features:**\n"
        "- Train teacher model with automatic ONNX export and threshold fitting.\n"
        "- Predict moderation labels from raw text inputs (CPU/GPU auto-fallback).\n"
        "### 🔍 Endpoints\n"
        "- `/train_teacher` — Train and export the teacher model.\n"
        "- `/predict` — Run inference on input texts.\n"
    ),
)

# ──────────────── Consistent error responses (global) ────────────────
def _err(content: dict, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=content)

def _env(message: str, code: str, retryable: bool = False, details: Optional[dict] = None) -> dict:
    return {"error": message, "code": code, "retryable": retryable, "details": details or {}}

@app.exception_handler(RequestValidationError)
async def _handle_req_validation(req: Request, exc: RequestValidationError):
    log.warning("Request validation error path=%s errors=%s", req.url.path, exc.errors())
    return _err(
    _env(
        "Invalid request payload",
        "VALIDATION_ERROR",
        False,
        {"errors": exc.errors()},
    ),
    status.HTTP_422_UNPROCESSABLE_ENTITY,
)

@app.exception_handler(ValidationError)
async def _handle_pydantic_validation(req: Request, exc: ValidationError):
    log.warning("Pydantic validation error path=%s errors=%s", req.url.path, exc.errors())
    return _err(_env("Validation failed", "VALIDATION_ERROR", False, {"errors": exc.errors()}),
                status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.exception_handler(AppError)
async def _handle_app_error(req: Request, exc: AppError):
    # 4xx → warn, 5xx → error
    lvl = logging.WARNING if 400 <= exc.status_code < 500 else logging.ERROR
    log.log(
    lvl,
    "AppError path=%s code=%s msg=%s details=%s",
    req.url.path,
    exc.code,
    exc.message,
    exc.details,
)
    return _err(_env(exc.message, exc.code, getattr(exc, "retryable", False), exc.details),
                exc.status_code)

@app.exception_handler(Exception)
async def _handle_unexpected(req: Request, exc: Exception):
    """Handle any unexpected, unhandled server errors."""
    log.exception("Unexpected error path=%s: %s", req.url.path, exc)
    return _err(
        _env("Unexpected server error", "UNEXPECTED_ERROR"),
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )

# ──────────────── Helpers ────────────────
def _validate_payload(inp: PredictInput):
    log.debug(
    "Validating payload: %d texts",
    len(inp.texts)
    if hasattr(inp, "texts") and isinstance(inp.texts, list)
    else -1,
)
    if not isinstance(inp.texts, list) or not inp.texts:
        raise InvalidInputError("`texts` must be a non-empty list of strings")
    if len(inp.texts) > MAX_TEXTS:
        raise InvalidInputError(
    f"`texts` list exceeds max size {MAX_TEXTS}",
    {"max_texts": MAX_TEXTS},
)
    for i, t in enumerate(inp.texts):
        if not isinstance(t, str):
            raise InvalidInputError(f"`texts[{i}]` must be a string", {"index": i})
        if len(t) > MAX_TEXT_LEN:
            raise InvalidInputError(f"`texts[{i}]` exceeds max length {MAX_TEXT_LEN}", {"index": i})

def _require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise ArtifactNotFoundError(f"Required artifact not found: {what}", {"path": str(path)})
    return path

def _load_metadata_and_thresholds() -> Tuple[List[str], np.ndarray]:
    log.debug("Loading metadata from %s", METADATA_PATH)
    try:
        meta = json.loads(_require_file(METADATA_PATH, "metadata.json").read_text())
    except FileNotFoundError as exc:
        raise ArtifactNotFoundError(
            "Required artifact not found: metadata.json",
            {"path": str(METADATA_PATH)},
        ) from exc
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in metadata: {e}") from e

    labels = meta.get("label_cols")
    if not isinstance(labels, list) or not labels:
        raise ConfigError("metadata.json missing 'label_cols' list")

    thr: np.ndarray
    if THRESHOLDS_PATH.exists():
        log.debug("Loading thresholds from %s", THRESHOLDS_PATH)
        try:
            thr_map = json.loads(Path(THRESHOLDS_PATH).read_text(encoding="utf-8"))
            thr = np.array([thr_map.get(label, 0.5) for label in labels], dtype=np.float32)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in thresholds: {e}") from e
    else:
        log.info("Thresholds file missing at %s; defaulting to 0.5 for all labels", THRESHOLDS_PATH)
        thr = np.full(len(labels), 0.5, dtype=np.float32)

    log.debug("Loaded labels=%d thresholds=OK", len(labels))
    return labels, thr

def _build_service() -> ClassifierService:
    log.debug("Checking artifacts: ONNX=%s | TOKENIZER=%s", ONNX_PATH, TOKENIZER_PATH)
    _require_file(ONNX_PATH, "ONNX model")
    if not Path(TOKENIZER_PATH).exists():
        raise ArtifactNotFoundError("Tokenizer directory not found", {"path": str(TOKENIZER_PATH)})
    try:
        log.info("Initializing ClassifierService...")
        svc = ClassifierService(onnx_path=str(ONNX_PATH), tokenizer_name=str(TOKENIZER_PATH))
        log.info("ClassifierService initialized successfully")
        return svc
    except Exception as e:
        # Could include type(e).__name__ if helpful
        raise ModelInitializationError(f"Failed to initialize model: {e}") from e

# ──────────────── Endpoints ────────────────
@app.post("/train_teacher")
def train_teacher():
    """Train the teacher model for content moderation.

    Triggers the supervised training routine and returns a success status.
    Raises TrainingError or TrainingResourceError on failure.
    """
    log.info("POST /train_teacher started")
    try:
        train_teacher_step()
        log.info("POST /train_teacher completed: success")
        return {"status": "success"}
    except OSError as e:
        log.error("Training resource error: %s", e)
        raise TrainingResourceError(f"Resource error during training: {e}") from e
    except Exception as e:
        log.exception("Training failed")
        raise TrainingError(str(e)) from e

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictInput) -> Dict:
    """Predict content moderation labels for given texts.

    Validates input, runs ONNX model inference, and returns label predictions.
    Raises validation, model, or schema-related errors if any step fails.
    """
    log.info("POST /predict started")
    # 1) Validate input (422)
    _validate_payload(payload)
    log.debug("Payload validated: %d texts", len(payload.texts))

    # 2) Build service + load metadata (503/500)
    service = _build_service()
    labels, thr = _load_metadata_and_thresholds()

    # 3) Inference (500)
    try:
        t0 = time.perf_counter()
        probs_chunks = service.predict(payload.texts)  # ndarray [chunks, num_labels]
        lat_ms = (time.perf_counter() - t0) * 1000.0
        log.info("Inference completed in %.2f ms", lat_ms)
    except Exception as e:
        log.exception("Prediction runtime error")
        raise PredictionError(str(e)) from e

    if not isinstance(probs_chunks, np.ndarray):
        log.error("Prediction returned non-numpy output: type=%s", type(probs_chunks))
        raise PredictionError("Model returned non-numpy predictions")

    # 4) Post-process (shape guard → 500)
    probs = probs_chunks.max(0)  # keep your 'max' mode
    log.debug("Post-process: probs shape=%s labels=%d", getattr(probs, "shape", None), len(labels))
    if probs.ndim != 1 or probs.shape[0] != len(labels):
        log.error(
            "Schema mismatch: got=%s expected=%s",
            getattr(probs, "shape", None),
            (len(labels),),
        ) 
        raise SchemaMismatchError(
            f"Prediction shape mismatch. got={tuple(probs.shape)} expected=({len(labels)},)",
            {"got": tuple(probs.shape), "expected": (len(labels),)},
        )

    preds = (probs >= thr).astype(int)
    result = labels_with_pred1({
        "labels": labels, "probs": probs, "preds": preds,
        "latency_ms": round(float(lat_ms), 2), "num_chunks": 1
    })

    log.info("POST /predict completed: labels=%d, preds_shape=%s", len(labels), preds.shape)
    return {
        "labels": labels,
        "preds": preds.tolist(),
        "latency_ms": round(float(lat_ms), 2),
        "num_chunks": 1,
        "result": result,
    }
