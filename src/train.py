"""Train and evaluate the causal dual-head model (XLM-R or mBERT).

The row caps, seed, threshold selection, and result schema match
``src.naive_baseline`` so the trained models can be compared with the naive
baseline and with each other across punctuation policies.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.multitask_model import CausalMultitaskModel
from src.streaming_dataloader import create_streaming_dataloader, make_split_loaders
from src.switchlingua_labels import (
    IGNORE_INDEX,
    PUNCTUATION_MODES,
    SELECTED_LANGUAGE_PAIRS,
    load_fast_tokenizer,
    records_from_parquet,
)


BACKBONES = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    """Same keys as the naive baseline's switch metrics."""

    targets = targets.bool()
    predictions = predictions.bool()
    tp = int((targets & predictions).sum())
    fp = int((~targets & predictions).sum())
    fn = int((targets & ~predictions).sum())
    tn = int((~targets & ~predictions).sum())
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "anticipatory_f1": _divide(2 * precision * recall, precision + recall),
        "precision": precision,
        "recall": recall,
        "accuracy": _divide(tp + tn, tp + fp + fn + tn),
        "support": int(targets.numel()),
        "positive_support": int(targets.sum()),
    }


def tune_threshold(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Pick the validation threshold with the best F1 (ties: closest to 0.5)."""

    candidates = []
    for step in range(1, 100):
        threshold = step / 100
        f1 = binary_metrics(targets, probabilities >= threshold)["anticipatory_f1"]
        candidates.append((f1, -abs(threshold - 0.5), threshold))
    return max(candidates)[2]


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Run the model once and gather every valid switch and duration target."""

    model.eval()
    switch_probabilities: List[torch.Tensor] = []
    switch_targets: List[torch.Tensor] = []
    switch_pairs: List[str] = []
    duration_predictions: List[torch.Tensor] = []
    duration_targets: List[torch.Tensor] = []
    duration_pairs: List[str] = []
    totals = {"loss": 0.0, "switch_loss": 0.0, "duration_loss": 0.0}
    batches = 0

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            causal_mask=batch["causal_mask"].to(device),
            switch_labels=batch["switch_labels"].to(device),
            duration_labels=batch["duration_labels"].to(device),
        )
        for key in totals:
            totals[key] += float(outputs[key])
        batches += 1

        probabilities = torch.softmax(outputs["switch_logits"].float(), dim=-1)[..., 1].cpu()
        duration_argmax = outputs["duration_logits"].argmax(dim=-1).cpu()
        switch_valid = batch["switch_labels"].ne(IGNORE_INDEX)
        duration_valid = batch["duration_labels"].ge(0)  # excludes both -1 and -100

        for row, pair in enumerate(batch["language_pair"]):
            switch_probabilities.append(probabilities[row][switch_valid[row]])
            switch_targets.append(batch["switch_labels"][row][switch_valid[row]])
            switch_pairs.extend([pair] * int(switch_valid[row].sum()))
            duration_predictions.append(duration_argmax[row][duration_valid[row]])
            duration_targets.append(batch["duration_labels"][row][duration_valid[row]])
            duration_pairs.extend([pair] * int(duration_valid[row].sum()))

    return {
        "switch_probabilities": torch.cat(switch_probabilities),
        "switch_targets": torch.cat(switch_targets),
        "switch_pairs": switch_pairs,
        "duration_predictions": torch.cat(duration_predictions),
        "duration_targets": torch.cat(duration_targets),
        "duration_pairs": duration_pairs,
        "losses": {key: _divide(value, batches) for key, value in totals.items()},
    }


def _metrics(
    predictions: Mapping[str, Any],
    threshold: float,
    switch_keep: Optional[torch.Tensor] = None,
    duration_keep: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    probabilities = predictions["switch_probabilities"]
    targets = predictions["switch_targets"]
    duration_predicted = predictions["duration_predictions"]
    duration_targets = predictions["duration_targets"]
    if switch_keep is not None:
        probabilities, targets = probabilities[switch_keep], targets[switch_keep]
    if duration_keep is not None:
        duration_predicted, duration_targets = duration_predicted[duration_keep], duration_targets[duration_keep]

    metrics = binary_metrics(targets, probabilities >= threshold)
    metrics["duration_accuracy"] = _divide(
        int((duration_predicted == duration_targets).sum()), duration_targets.numel()
    )
    metrics["duration_support"] = int(duration_targets.numel())
    metrics["switch_rate"] = _divide(int(targets.sum()), targets.numel())
    return metrics


def summarize(predictions: Mapping[str, Any], threshold: float) -> Dict[str, Any]:
    """Overall and per-pair metrics in the naive baseline's result layout."""

    pairs = sorted(set(predictions["switch_pairs"]))
    by_pair = {}
    for pair in pairs:
        switch_keep = torch.tensor([value == pair for value in predictions["switch_pairs"]], dtype=torch.bool)
        duration_keep = torch.tensor([value == pair for value in predictions["duration_pairs"]], dtype=torch.bool)
        by_pair[pair] = _metrics(predictions, threshold, switch_keep, duration_keep)
    return {
        "overall": _metrics(predictions, threshold),
        "by_language_pair": by_pair,
        "universality_std": statistics.pstdev(
            metrics["anticipatory_f1"] for metrics in by_pair.values()
        ) if len(by_pair) > 1 else 0.0,
        "threshold": threshold,
        "losses": dict(predictions["losses"]),
    }


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_clip: float = 1.0,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Fine-tune, track convergence, and restore the best validation-F1 epoch."""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(warmup_ratio * total_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    step_history: List[Dict[str, float]] = []
    epoch_history: List[Dict[str, Any]] = []
    best = {"f1": -1.0, "epoch": 0, "threshold": 0.5, "state": None}
    step = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "switch_loss": 0.0, "duration_loss": 0.0}
        window = dict(running)
        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                causal_mask=batch["causal_mask"].to(device),
                switch_labels=batch["switch_labels"].to(device),
                duration_labels=batch["duration_labels"].to(device),
            )
            optimizer.zero_grad()
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            for key in running:
                value = float(outputs[key].detach())
                running[key] += value
                window[key] += value
            if step % log_every == 0:
                entry = {"step": step, "epoch": epoch}
                entry.update({f"train_{key}": value / log_every for key, value in window.items()})
                step_history.append(entry)
                window = {key: 0.0 for key in window}
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {entry['train_loss']:.4f} "
                    f"(switch {entry['train_switch_loss']:.4f}, duration {entry['train_duration_loss']:.4f})",
                    flush=True,
                )

        predictions = collect_predictions(model, validation_loader, device)
        threshold = tune_threshold(predictions["switch_probabilities"], predictions["switch_targets"])
        validation = _metrics(predictions, threshold)
        entry = {
            "epoch": epoch,
            "step": step,
            "seconds": time.time() - started,
            "validation_threshold": threshold,
            "validation_anticipatory_f1": validation["anticipatory_f1"],
            "validation_duration_accuracy": validation["duration_accuracy"],
        }
        entry.update({f"train_{key}": _divide(value, len(train_loader)) for key, value in running.items()})
        entry.update({f"validation_{key}": value for key, value in predictions["losses"].items()})
        epoch_history.append(entry)
        print(
            f"epoch {epoch} done: train loss {entry['train_loss']:.4f}, "
            f"val loss {entry['validation_loss']:.4f}, "
            f"val F1 {entry['validation_anticipatory_f1']:.4f} @ {threshold:.2f}, "
            f"val duration acc {entry['validation_duration_accuracy']:.4f}",
            flush=True,
        )

        if validation["anticipatory_f1"] > best["f1"]:
            best = {
                "f1": validation["anticipatory_f1"],
                "epoch": epoch,
                "threshold": threshold,
                "state": copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()}),
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "best_validation_anticipatory_f1": best["f1"],
        "epochs": epoch_history,
        "steps": step_history,
    }


def _load_split_records(args: argparse.Namespace, tokenizer: Any, punctuation_mode: str) -> Dict[str, List[Dict[str, Any]]]:
    limits = {
        "train": args.train_rows_per_pair,
        "validation": args.validation_rows_per_pair,
        "test": args.test_rows_per_pair,
    }
    return {
        split: records_from_parquet(
            str(Path(args.data_dir) / f"{split}.parquet"),
            tokenizer,
            language_pairs=args.pairs,
            max_rows_per_pair=limit,
            max_utterances_per_row=args.utterances_per_row,
            seed=args.seed,
            punctuation_mode=punctuation_mode,
        )
        for split, limit in limits.items()
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="xlmr")
    parser.add_argument(
        "--punctuation-mode",
        choices=PUNCTUATION_MODES,
        default="keep",
        help="Approach A (keep), B (drop), or C (neutral).",
    )
    parser.add_argument(
        "--cross-eval-mode",
        choices=PUNCTUATION_MODES,
        default=None,
        help="Also score the trained model on test labels built with another punctuation policy.",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default=None, help="Default: results/train_<backbone>_<mode>.json")
    parser.add_argument("--pairs", nargs="+", default=list(SELECTED_LANGUAGE_PAIRS))
    parser.add_argument("--train-rows-per-pair", type=int, default=500)
    parser.add_argument("--validation-rows-per-pair", type=int, default=150)
    parser.add_argument("--test-rows-per-pair", type=int, default=150)
    parser.add_argument("--utterances-per-row", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-checkpoint", default=None, help="Optional path for the best model weights.")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    model_name = BACKBONES[args.backbone]
    tokenizer = load_fast_tokenizer(model_name)
    print(f"backbone {model_name} | punctuation {args.punctuation_mode} | device {device}", flush=True)

    records = _load_split_records(args, tokenizer, args.punctuation_mode)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
    )
    train_loader, validation_loader, test_loader = make_split_loaders(
        records["train"],
        records["validation"],
        records["test"],
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )

    model = CausalMultitaskModel(
        model_name=model_name,
        switch_weight=args.switch_weight,
        duration_weight=args.duration_weight,
        dropout=args.dropout,
    )
    history = train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        log_every=args.log_every,
    )

    result = summarize(collect_predictions(model, test_loader, device), history["threshold"])
    result["definition"] = "causal dual-head transformer; threshold and epoch selected on validation"
    result["history"] = history

    if args.cross_eval_mode and args.cross_eval_mode != args.punctuation_mode:
        # Same rows and seed, labels from the other policy. The threshold is
        # re-selected on that policy's validation labels; weights are unchanged.
        other = _load_split_records(args, tokenizer, args.cross_eval_mode)
        other_validation = create_streaming_dataloader(other["validation"], **loader_kwargs)
        other_test = create_streaming_dataloader(other["test"], **loader_kwargs)
        validation_predictions = collect_predictions(model, other_validation, device)
        other_threshold = tune_threshold(
            validation_predictions["switch_probabilities"], validation_predictions["switch_targets"]
        )
        cross = summarize(collect_predictions(model, other_test, device), other_threshold)
        cross["label_punctuation_mode"] = args.cross_eval_mode
        result["cross_eval"] = cross

    result["run"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone": model_name,
        "tokenizer": model_name,
        "language_pairs": args.pairs,
        "weak_lid_labels": True,
        "punctuation_mode": args.punctuation_mode,
        "seed": args.seed,
        "rows_per_pair": {
            "train": args.train_rows_per_pair,
            "validation": args.validation_rows_per_pair,
            "test": args.test_rows_per_pair,
        },
        "utterances_per_row": args.utterances_per_row,
        "record_counts": {split: len(values) for split, values in records.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "switch_weight": args.switch_weight,
            "duration_weight": args.duration_weight,
            "dropout": args.dropout,
        },
        "device": str(device),
    }

    output = Path(args.output or f"results/train_{args.backbone}_{args.punctuation_mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.save_checkpoint:
        Path(args.save_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_checkpoint)

    overall = result["overall"]
    print(
        f"TEST  F1 {overall['anticipatory_f1']:.4f}  P {overall['precision']:.4f}  R {overall['recall']:.4f}  "
        f"duration acc {overall['duration_accuracy']:.4f}  universality σ {result['universality_std']:.4f}  "
        f"(best epoch {history['best_epoch']}, threshold {history['threshold']:.2f})"
    )
    print("saved", output)


if __name__ == "__main__":
    main()
