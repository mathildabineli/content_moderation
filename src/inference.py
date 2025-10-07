import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from .config import CFG

class ClassifierService:
    def __init__(self, onnx_path: str, tokenizer_name: str):
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def predict(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=CFG.max_len)
        outputs = self.session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        return outputs[0]
