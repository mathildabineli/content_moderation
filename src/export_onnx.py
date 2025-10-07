import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CFG
from .utils import LOG


def export_teacher_onnx(model_dir: str, tokenizer: AutoTokenizer) -> str:
    class Wrap(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.eval()

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )["logits"]

    onnx_dir = f"{CFG.out_dir}/onnx"
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    ).eval()
    sample = tokenizer(
        "hello",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=CFG.max_len,
    )
    dyn_axes = {
        "input_ids": {0: "B", 1: "L"},
        "attention_mask": {0: "B", 1: "L"},
        "logits": {0: "B"},
    }
    out_fp = os.path.join(onnx_dir, "teacher_fp32.onnx")
    torch.onnx.export(
        Wrap(model),
        (sample["input_ids"], sample["attention_mask"]),
        out_fp,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dyn_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    LOG.info("Teacher ONNX exported: %s", out_fp)
    return out_fp
