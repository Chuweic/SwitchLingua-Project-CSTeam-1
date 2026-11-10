
"""Causal dual-head transformer for code-switch prediction."""

import torch
from torch import nn
from transformers import AutoModel

from src.streaming_dataloader import (
    build_causal_mask,
    causal_mask_for_backbone,
    multitask_cross_entropy,
)


class CausalMultitaskModel(nn.Module):
    """Predict next-token switches and upcoming language burst lengths."""

    def __init__(
        self,
        model_name="xlm-roberta-base",
        switch_weight=1.0,
        duration_weight=1.0,
        dropout=0.1,
        backbone=None,
    ):
        super().__init__()

        # Shared multilingual transformer backbone.
        # An optional backbone allows lightweight unit testing.
        self.backbone = (
            backbone if backbone is not None
            else AutoModel.from_pretrained(model_name)
        )

        hidden_size = self.backbone.config.hidden_size

        # Two independent prediction heads.
        self.dropout = nn.Dropout(dropout)
        self.switch_head = nn.Linear(hidden_size, 2)
        self.duration_head = nn.Linear(hidden_size, 3)

        self.switch_weight = switch_weight
        self.duration_weight = duration_weight

    def forward(
        self,
        input_ids,
        attention_mask,
        causal_mask=None,
        switch_labels=None,
        duration_labels=None,
    ):
        # Prevent the transformer from attending to future tokens.
        if causal_mask is None:
            causal_mask = build_causal_mask(attention_mask)

        backbone_mask = causal_mask_for_backbone(causal_mask)

        # Extract contextual token representations.
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=backbone_mask,
            return_dict=True,
        ).last_hidden_state

        hidden = self.dropout(hidden)

        # Predict the next-token switch and duration at each position.
        switch_logits = self.switch_head(hidden)
        duration_logits = self.duration_head(hidden)

        outputs = {
            "switch_logits": switch_logits,
            "duration_logits": duration_logits,
        }

        # Calculate multitask loss when both labels are supplied.
        if (switch_labels is None) != (duration_labels is None):
            raise ValueError("Both label tensors must be provided together.")

        if switch_labels is not None:
            losses = multitask_cross_entropy(
                switch_logits,
                duration_logits,
                switch_labels,
                duration_labels,
                switch_weight=self.switch_weight,
                duration_weight=self.duration_weight,
            )
            outputs.update(losses)

        return outputs
