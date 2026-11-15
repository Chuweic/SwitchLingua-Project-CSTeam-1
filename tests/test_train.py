import json

import torch
from transformers import BertConfig, BertModel

from src.multitask_model import CausalMultitaskModel
from src.streaming_dataloader import create_streaming_dataloader
from src.train import (
    binary_metrics,
    collect_predictions,
    summarize,
    train_model,
    tune_threshold,
)


def tiny_model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    return CausalMultitaskModel(backbone=BertModel(config))


def make_records(pair, count):
    # Token 20 is always followed by a switch into a two-token burst.
    records = []
    for index in range(count):
        filler = 10 + index % 5
        records.append({
            "input_ids": [2, filler, filler, 20, 30, 31, 3],
            "attention_mask": [1] * 7,
            "subword_language_ids": [None, "es", "es", "es", "en", "en", None],
            "switch_labels": [-100, 0, 0, 1, 0, -100, -100],
            "duration_labels": [-100, -1, -1, 0, -1, -100, -100],
            "language_pair": pair,
        })
    return records


def loader(records, shuffle=False):
    return create_streaming_dataloader(records, batch_size=8, shuffle=shuffle, pad_token_id=0)


def test_binary_metrics_matches_hand_computation():
    targets = torch.tensor([1, 1, 0, 0, 1])
    predictions = torch.tensor([1, 0, 1, 0, 1])
    metrics = binary_metrics(targets, predictions)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert abs(metrics["anticipatory_f1"] - 2 / 3) < 1e-9
    assert metrics["support"] == 5
    assert metrics["positive_support"] == 3


def test_tune_threshold_separates_classes():
    probabilities = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.35])
    targets = torch.tensor([0, 0, 0, 1, 1])
    threshold = tune_threshold(probabilities, targets)
    assert 0.20 < threshold <= 0.30
    assert binary_metrics(targets, probabilities >= threshold)["anticipatory_f1"] == 1.0


def test_collect_predictions_skips_ignored_positions():
    records = make_records("Spanish-English", 3)
    predictions = collect_predictions(tiny_model(), loader(records), torch.device("cpu"))
    assert predictions["switch_targets"].numel() == 3 * 4
    assert predictions["duration_targets"].numel() == 3
    assert len(predictions["switch_pairs"]) == 12
    assert set(predictions["duration_targets"].tolist()) == {0}


def test_training_learns_toy_task_and_reports_by_pair():
    train = make_records("Spanish-English", 32) + make_records("Arabic-English", 32)
    held_out = make_records("Spanish-English", 8) + make_records("Arabic-English", 8)
    model = tiny_model()
    device = torch.device("cpu")

    history = train_model(
        model,
        loader(train, shuffle=True),
        loader(held_out),
        device=device,
        epochs=8,
        learning_rate=5e-3,
        log_every=4,
    )
    result = summarize(collect_predictions(model, loader(held_out), device), history["threshold"])

    assert len(history["epochs"]) == 8
    assert history["steps"]
    assert history["epochs"][-1]["train_loss"] < history["epochs"][0]["train_loss"]
    assert 1 <= history["best_epoch"] <= 8
    assert result["overall"]["anticipatory_f1"] == 1.0
    assert result["overall"]["duration_accuracy"] == 1.0
    assert set(result["by_language_pair"]) == {"Arabic-English", "Spanish-English"}
    assert result["universality_std"] == 0.0
    json.dumps(result)  # the result must be directly serializable


def test_step_history_averages_full_windows_across_epochs():
    records = make_records("Spanish-English", 64)  # 8 steps per epoch
    history = train_model(
        tiny_model(),
        loader(records),
        loader(records[:8]),
        device=torch.device("cpu"),
        epochs=3,
        learning_rate=1e-3,
        log_every=3,  # windows straddle the epoch boundaries at steps 8 and 16
    )
    assert [entry["step"] for entry in history["steps"]] == list(range(3, 25, 3))
    from_windows = sum(entry["train_loss"] * 3 for entry in history["steps"])
    from_epochs = sum(entry["train_loss"] * 8 for entry in history["epochs"])
    assert abs(from_windows - from_epochs) < 1e-4
