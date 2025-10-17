"""Builder pattern for Hugging Face TrainingArguments configuration."""

from transformers import TrainingArguments
from .config import CFG


class TrainingArgsBuilder:
    """Fluent builder for constructing TrainingArguments."""

    def __init__(self, output_dir: str):
        """Initialize builder with output directory."""
        self.args = {"output_dir": output_dir}

    def with_epochs(self, n: int):
        """Set number of training epochs."""
        self.args["num_train_epochs"] = n
        return self

    def with_batch_size(self, b: int):
        """Set per-device batch size for training and evaluation."""
        self.args["per_device_train_batch_size"] = b
        self.args["per_device_eval_batch_size"] = b
        return self

    def with_lr(self, lr: float):
        """Set learning rate."""
        self.args["learning_rate"] = lr
        return self

    def with_wd(self, wd: float):
        """Set weight decay."""
        self.args["weight_decay"] = wd
        return self

    def build(self) -> TrainingArguments:
        """Finalize and create TrainingArguments with defaults."""
        self.args.update(
            {
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "macro_f1",
                "greater_is_better": True,
                "fp16": True if CFG.device == "cuda" else False,
                "warmup_ratio": 0.06,
                "lr_scheduler_type": "cosine",
                "save_total_limit": 2,
                "report_to": "none",
                "seed": CFG.seed,
            }
        )
        return TrainingArguments(**self.args)
