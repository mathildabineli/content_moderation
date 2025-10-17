"""Model definitions for the Content Moderation system.

Includes transformer-based teacher architecture, configuration
objects, factory creation, and evaluation metric strategy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

from .config import CFG


class MTConfig(PretrainedConfig):
    """Custom configuration for the moderation transformer model."""
    model_type = "singlehead_backbone"

    def __init__(
        self, backbone="microsoft/mdeberta-v3-base", num_labels=8, gamma_focal=2.0, **kw
    ):
        super().__init__(**kw)
        self.backbone = backbone
        self.num_labels = int(num_labels)
        self.gamma_focal = float(gamma_focal)


class MTModel(PreTrainedModel):
    """Single-head transformer classifier for multi-label moderation."""
    config_class = MTConfig

    def __init__(self, config: MTConfig):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(config.backbone)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Run a forward pass; returns logits and optional loss."""
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden = self.dropout(out.last_hidden_state[:, 0])
        logits = self.classifier(hidden)
        if labels is None:
            return {"logits": logits}
        bce = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = labels * p + (1 - labels) * (1 - p)
        loss = ((1 - pt).clamp_min(1e-6).pow(self.config.gamma_focal) * bce).mean()
        return {"loss": loss, "logits": logits}


class ModelFactory:
    """Factory for creating preconfigured model instances."""
    @staticmethod
    def create_teacher(num_labels: int) -> MTModel:
        """Instantiate the teacher model with predefined config."""
        cfg = MTConfig(
            backbone=CFG.teacher_model,
            num_labels=num_labels,
            gamma_focal=CFG.gamma_focal,
        )
        cfg.id2label = {i: f"L{i}" for i in range(num_labels)}
        cfg.label2id = {v: k for k, v in cfg.id2label.items()}
        return MTModel(cfg)


class MetricStrategy:
    """Evaluation metric strategy for model validation."""
    def __call__(self, eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        y_pred = (probs >= 0.5).astype(int)
        return {
            "macro_f1": float(
                f1_score(labels, y_pred, average="macro", zero_division=0)
            )
        }
