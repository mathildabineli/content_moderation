import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

from .config import CFG


class ClassifierService:
    def __init__(self, onnx_path: str, tokenizer_name: str):
        # self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=True
        )
        print("tokenizer loaded")

        try:
            # Perform a full check (including shape inference)
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model, full_check=True)
            print("The model is valid!")
            self.session = ort.InferenceSession(onnx_path)
            print("ready for inference")
            print("tokenizer path------", tokenizer_name)

        except onnx.checker.ValidationError as e:
            print(f"The model is invalid: {e}")

    def predict(self, texts: list[str]) -> np.ndarray:
        print("starting loading encoder")
        enc = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=CFG.max_len,
        )
        print("encoder loaded")
        outputs = self.session.run(
            None,
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        )
        print("inference", outputs[0].__str__())

        return outputs[0]
