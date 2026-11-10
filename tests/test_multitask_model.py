
"""Tests for the causal dual-head multitask model."""

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from src.multitask_model import CausalMultitaskModel


@pytest.fixture(params=["xlmr", "mbert"])
def model(request):
    """Use tiny randomly initialized backbones for fast testing."""
    torch.manual_seed(42)

    config_args = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )

    if request.param == "xlmr":
        config = XLMRobertaConfig(**config_args, pad_token_id=1)
        backbone = XLMRobertaModel(config)
    else:
        config = BertConfig(**config_args, pad_token_id=0)
        backbone = BertModel(config)

    return CausalMultitaskModel(backbone=backbone)


def sample_batch():
    return {
        "input_ids": torch.tensor([[2, 10, 11, 12, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "switch_labels": torch.tensor([[-100, 0, 1, 0, -100]]),
        "duration_labels": torch.tensor([[-100, -1, 1, -1, -100]]),
    }


def test_output_shapes(model):
    model.eval()
    batch = sample_batch()

    with torch.no_grad():
        outputs = model(**batch)

    assert outputs["switch_logits"].shape == (1, 5, 2)
    assert outputs["duration_logits"].shape == (1, 5, 3)
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])


def test_gradient_flow(model):
    model.train()
    outputs = model(**sample_batch())
    outputs["loss"].backward()

    assert model.switch_head.weight.grad is not None
    assert model.duration_head.weight.grad is not None
    assert model.backbone.embeddings.word_embeddings.weight.grad is not None

    assert model.switch_head.weight.grad.abs().sum() > 0
    assert model.duration_head.weight.grad.abs().sum() > 0
    assert (
        model.backbone.embeddings.word_embeddings.weight.grad.abs().sum()
        > 0
    )


def test_duration_loss_masking(model):
    """Duration loss should only include positions with labels 0, 1, 2."""
    model.eval()
    batch = sample_batch()

    with torch.no_grad():
        outputs = model(**batch)

    expected_duration_loss = F.cross_entropy(
        outputs["duration_logits"][:, 2, :],
        torch.tensor([1]),
    )

    assert torch.allclose(
        outputs["duration_loss"],
        expected_duration_loss,
        atol=1e-6,
    )
    assert outputs["duration_count"].item() == 1
    assert outputs["switch_count"].item() == 3


def test_all_duration_labels_ignored(model):
    batch = sample_batch()
    batch["duration_labels"] = torch.full((1, 5), -1)

    outputs = model(**batch)

    assert outputs["duration_loss"].item() == 0
    assert outputs["duration_count"].item() == 0
    assert torch.isfinite(outputs["loss"])


def test_no_future_token_leakage(model):
    """Changing future tokens must not affect earlier predictions."""
    model.eval()
    batch = sample_batch()

    original_ids = batch["input_ids"].clone()
    modified_ids = original_ids.clone()

    # Change future tokens at positions 3 and 4.
    modified_ids[0, 3] = 75
    modified_ids[0, 4] = 76

    with torch.no_grad():
        original = model(
            input_ids=original_ids,
            attention_mask=batch["attention_mask"],
        )
        modified = model(
            input_ids=modified_ids,
            attention_mask=batch["attention_mask"],
        )

    # Positions 0, 1, 2 must remain unchanged.
    for head in ("switch_logits", "duration_logits"):
        assert torch.allclose(
            original[head][:, :3],
            modified[head][:, :3],
            atol=1e-6,
        ), f"Future-token leakage detected in {head}"


def test_partial_labels_rejected(model):
    batch = sample_batch()

    with pytest.raises(ValueError):
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            switch_labels=batch["switch_labels"],
        )
