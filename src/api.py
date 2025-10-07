import json
import os
import time
from typing import Dict

import numpy as np
from fastapi import FastAPI

from src.inference import ClassifierService
from src.models import PredictInput
from src.train_teacher import train_teacher_step
from src.utils import labels_with_pred1

"""Services API for content moderation"""

app = FastAPI()


@app.post("/train_teacher")
def train_teacher():
    """
    train the teacher model for content moderation
    input : None
    output: None
    """
    train_teacher_step()
    return {"status": "success"}


@app.post("/predict")
def predict(input:PredictInput) -> Dict:
    """
    this function predict the class of the input text
    """
    service = ClassifierService(
        onnx_path="artifacts/teacher/best/teacher_fp32.onnx",
        tokenizer_name="/home/meligavincent/Desktop/content_moderation/src/artifacts/teacher/best/",
    )

    print("service created")
    preds = service.predict(input.texts)
    print("here are preds", preds, type(preds))

    t0 = time.perf_counter()
    probs_chunks = preds
    lat_ms = (time.perf_counter() - t0) * 1000

    meta = json.load(
        open(
            f"/home/meligavincent/Desktop/content_moderation/src/artifacts/teacher/best/metadata.json"
        )
    )
    labels = meta["label_cols"]
    thr = (
        json.load(
            open(
                f"/home/meligavincent/Desktop/content_moderation/src/artifacts/teacher/best/teacher_thresholds.json"
            )
        )
        if os.path.exists(
            f"/home/meligavincent/Desktop/content_moderation/src/artifacts/teacher/best/teacher_thresholds.json"
        )
        else dict(zip(labels, [0.5] * len(labels)))
    )

    mode = "max"

    probs = probs_chunks.mean(0) if mode == "mean" else probs_chunks.max(0)
    thr = (
        np.full(probs.shape[-1], 0.5)
        if (labels is None or thr is None)
        else np.array([thr.get(l, 0.5) for l in labels])
    )

    probs, preds = probs, (probs >= thr).astype(int)


    result = labels_with_pred1(
        {
            "labels": labels,
            "probs": probs,
            "preds": preds,
            "latency_ms": round(float(lat_ms), 2),
            "num_chunks": 1,
        }
    )

    return {
        "labels": labels,
        "preds": preds.tolist(),
        "latency_ms": round(float(lat_ms), 2),
        "num_chunks": 1,
        "result": result,
    }
