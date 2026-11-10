
"""Real-data integration test for the dual-head transformer."""

import argparse

import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from src.multitask_model import CausalMultitaskModel
from src.switchlingua_labels import (
    load_fast_tokenizer,
    records_from_parquet,
)
from src.streaming_dataloader import create_streaming_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    # 1. Load real SwitchLingua examples.
    tokenizer = load_fast_tokenizer("xlm-roberta-base")

    records = records_from_parquet(
        "data/processed/validation.parquet",
        tokenizer,
        language_pairs=("Chinese-English", "Spanish-English"),
        max_rows_per_pair=5,
        max_utterances_per_row=2,
        punctuation_mode="keep",
    )

    # Select examples containing a switch within the first 128 tokens.
    records = [
        r for r in records
        if any(d >= 0 for d in r["duration_labels"][:128])
    ]
    assert records, "No switch examples found. Increase sample size."

    print("Loaded records:", len(records))
    print("Example:", records[0]["text"])
    print("Punctuation mode:", records[0]["punctuation_mode"])

    # 2. Build batch with existing DataLoader.
    loader = create_streaming_dataloader(
        records[:2],
        batch_size=2,
        pad_token_id=tokenizer.pad_token_id,
        max_length=128,
    )
    batch = next(iter(loader))

    print("Input shape:", batch["input_ids"].shape)
    print("Causal mask:", batch["causal_mask"].shape)

    # 3. Load backbone.
    if args.pretrained:
        model = CausalMultitaskModel(
            model_name="xlm-roberta-base"
        )
    else:
        # Lightweight model for fast integration testing.
        config = XLMRobertaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=514,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = CausalMultitaskModel(
            backbone=XLMRobertaModel(config)
        )

    # 4. Run forward pass with real labels.
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        causal_mask=batch["causal_mask"],
        switch_labels=batch["switch_labels"],
        duration_labels=batch["duration_labels"],
    )

    # 5. Verify shapes, loss and gradients.
    assert outputs["switch_logits"].shape == (
        *batch["input_ids"].shape, 2
    )
    assert outputs["duration_logits"].shape == (
        *batch["input_ids"].shape, 3
    )
    assert torch.isfinite(outputs["loss"])
    assert outputs["duration_count"].item() > 0

    outputs["loss"].backward()

    assert model.switch_head.weight.grad is not None
    assert model.duration_head.weight.grad is not None
    assert any(
        p.grad is not None
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    print("Switch logits:", outputs["switch_logits"].shape)
    print("Duration logits:", outputs["duration_logits"].shape)
    print("Switch loss:", outputs["switch_loss"].item())
    print("Duration loss:", outputs["duration_loss"].item())
    print("Total loss:", outputs["loss"].item())
    print("Valid switch positions:", outputs["switch_count"].item())
    print("Valid duration positions:", outputs["duration_count"].item())
    print("Integration test PASSED")


if __name__ == "__main__":
    main()
