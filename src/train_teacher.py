"""Teacher model training pipeline for content moderation.

Handles dataset preparation, model training, ONNX export,
and threshold computation for the moderation teacher model.
"""
import json
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
)

from .config import CFG
from .data import build_hf_dataset, split_dataset
from .export_onnx import export_teacher_onnx
from .teacher import MetricStrategy, ModelFactory
from .thresholds import fit_thresholds_with_floor
from .training_builder import TrainingArgsBuilder
from .utils import LOG, ensure_dataset_exists, set_seed


def train_teacher_step():
    """Train the teacher moderation model end-to-end.

    Builds datasets, trains using Hugging Face Trainer,
    exports to ONNX, and saves fitted decision thresholds.
    """
    
    set_seed(CFG.seed)
    ensure_dataset_exists()

    train_df, val_df, _, labels, text_col = split_dataset()
    tokenizer = AutoTokenizer.from_pretrained(CFG.teacher_model, use_fast=True)

    dataset = DatasetDict(
        {
            "train": build_hf_dataset(tokenizer, train_df, text_col, labels),
            "validation": build_hf_dataset(tokenizer, val_df, text_col, labels),
        }
    )

    teacher = ModelFactory.create_teacher(len(labels))
    metric = MetricStrategy()

    # Builder Pattern pour TrainingArguments
    train_args = (
        TrainingArgsBuilder(f"{CFG.out_dir}/teacher")
        .with_epochs(CFG.t_epochs)
        .with_batch_size(CFG.t_bsz_train)
        .with_lr(CFG.t_lr)
        .with_wd(CFG.t_wd)
        .build()
    )

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=teacher,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.early_stop)],
    )

    trainer.train()
    trainer.save_model(f"{CFG.out_dir}/teacher")

    # Export ONNX
    export_teacher_onnx(f"{CFG.out_dir}/teacher", tokenizer)

    # Compute thresholds
    preds = trainer.predict(dataset["validation"]).predictions
    thresholds = fit_thresholds_with_floor(
        preds, dataset["validation"]["labels"], labels, floor=CFG.recall_floor
    )
    with open(f"{CFG.out_dir}/teacher_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f)
    LOG.info("Training completed and thresholds saved.")
