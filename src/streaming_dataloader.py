"""Streaming DataLoader, batch collation, and causal masking (Member C).

Each input record is the dictionary returned by
``src.switchlingua_labels.encode_and_label`` (Member B), optionally extended
with a ``language_pair`` string.  Labels are computed by Member B on the full
utterance; this module only controls what the model is allowed to *see*:
position ``t`` may attend to positions ``0..t`` and never to ``t+1`` or later.
"""

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.switchlingua_labels import IGNORE_INDEX


REQUIRED_FIELDS = (
    "input_ids",
    "attention_mask",
    "subword_language_ids",
    "switch_labels",
    "duration_labels",
)
UNKNOWN_LANGUAGE_PAIR = "unknown"


def _validate_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise KeyError(f"record is missing fields: {missing}")
    lengths = {field: len(record[field]) for field in REQUIRED_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"record fields must have the same length: {lengths}")
    if lengths["input_ids"] == 0:
        raise ValueError("record must contain at least one token")


def build_lid_vocab(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Map every language ID seen in ``records`` to a stable integer index."""

    languages = {
        language
        for record in records
        for language in record["subword_language_ids"]
        if language is not None
    }
    return {language: index for index, language in enumerate(sorted(languages))}


class SwitchLinguaStreamingDataset(Dataset):
    """Holds Member B outputs and converts them to per-utterance tensors.

    ``max_length`` truncates from the right *after* labelling, so the label at
    the last kept position still refers to the true next sub-word.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        lid_vocab: Optional[Mapping[str, int]] = None,
        max_length: Optional[int] = None,
    ):
        for record in records:
            _validate_record(record)
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        self.records = list(records)
        self.lid_vocab = dict(lid_vocab) if lid_vocab is not None else build_lid_vocab(self.records)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _lid_index(self, language: Optional[str]) -> int:
        if language is None:
            return IGNORE_INDEX
        if language not in self.lid_vocab:
            raise KeyError(f"language ID {language!r} is not in lid_vocab")
        return self.lid_vocab[language]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        end = self.max_length if self.max_length is not None else len(record["input_ids"])
        return {
            "input_ids": torch.tensor(record["input_ids"][:end], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"][:end], dtype=torch.long),
            "current_lid": torch.tensor(
                [self._lid_index(language) for language in record["subword_language_ids"][:end]],
                dtype=torch.long,
            ),
            "switch_labels": torch.tensor(record["switch_labels"][:end], dtype=torch.long),
            "duration_labels": torch.tensor(record["duration_labels"][:end], dtype=torch.long),
            "language_pair": record.get("language_pair", UNKNOWN_LANGUAGE_PAIR),
        }


def build_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a boolean ``[batch, 1, L, L]`` mask where ``True`` means "may attend".

    Query ``i`` may attend to key ``j`` only when ``j <= i`` and key ``j`` is a
    real (non-padding) token.  With right padding every query row keeps at
    least one visible key, so softmax never sees a fully masked row.
    """

    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len]")
    seq_len = attention_mask.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device))
    key_is_real = attention_mask.bool()[:, None, None, :]
    return causal[None, None, :, :] & key_is_real


def to_additive_mask(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a boolean "may attend" mask into 0 / large-negative scores."""

    additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return additive.masked_fill(~mask, torch.finfo(dtype).min)


def collate_streaming_batch(
    features: Sequence[Mapping[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Right-pad a list of dataset items into one batch with a causal mask."""

    if not features:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(features)
    seq_len = max(item["input_ids"].size(0) for item in features)

    def padded(value: int) -> torch.Tensor:
        return torch.full((batch_size, seq_len), value, dtype=torch.long)

    batch = {
        "input_ids": padded(pad_token_id),
        "attention_mask": padded(0),
        "current_lid": padded(IGNORE_INDEX),
        "switch_labels": padded(IGNORE_INDEX),
        "duration_labels": padded(IGNORE_INDEX),
    }
    for row, item in enumerate(features):
        length = item["input_ids"].size(0)
        for field in batch:
            batch[field][row, :length] = item[field]

    batch["causal_mask"] = build_causal_mask(batch["attention_mask"])
    batch["language_pair"] = [item["language_pair"] for item in features]
    return batch


def create_streaming_dataloader(
    records: Sequence[Mapping[str, Any]],
    batch_size: int = 32,
    shuffle: bool = False,
    pad_token_id: int = 0,
    lid_vocab: Optional[Mapping[str, int]] = None,
    max_length: Optional[int] = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Build a DataLoader over Member B records.

    Use the training set's ``lid_vocab`` for validation and test loaders so the
    language indices stay consistent across splits.
    """

    dataset = SwitchLinguaStreamingDataset(records, lid_vocab=lid_vocab, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda features: collate_streaming_batch(features, pad_token_id=pad_token_id),
        **dataloader_kwargs,
    )


def iter_prefix_windows(record: Mapping[str, Any], include_ignored: bool = False) -> Iterator[Dict[str, Any]]:
    """Yield explicit prefix-only samples ``(x_0..x_t) -> labels[t]``.

    Intended for streaming evaluation and demonstrations.  For training, feed
    the full sequence with ``causal_mask`` instead: it is equivalent and avoids
    the O(L^2) blow-up of materialising every prefix.
    """

    _validate_record(record)
    for position in range(len(record["input_ids"])):
        switch_label = record["switch_labels"][position]
        if switch_label == IGNORE_INDEX and not include_ignored:
            continue
        yield {
            "position": position,
            "prefix_input_ids": list(record["input_ids"][: position + 1]),
            "prefix_language_ids": list(record["subword_language_ids"][: position + 1]),
            "switch_label": switch_label,
            "duration_label": record["duration_labels"][position],
            "language_pair": record.get("language_pair", UNKNOWN_LANGUAGE_PAIR),
        }


def records_to_prefix_windows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [window for record in records for window in iter_prefix_windows(record)]


def make_split_loaders(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    **loader_kwargs: Any,
):
    """Build all splits while forcing validation/test to reuse train LIDs."""

    loader_kwargs = dict(loader_kwargs)
    loader_kwargs.pop("shuffle", None)
    train_loader = create_streaming_dataloader(train_records, shuffle=True, **loader_kwargs)
    lid_vocab = train_loader.dataset.lid_vocab
    validation_loader = create_streaming_dataloader(
        validation_records, lid_vocab=lid_vocab, shuffle=False, **loader_kwargs
    )
    test_loader = create_streaming_dataloader(
        test_records, lid_vocab=lid_vocab, shuffle=False, **loader_kwargs
    )
    return train_loader, validation_loader, test_loader


def multitask_cross_entropy(
    switch_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    switch_labels: torch.Tensor,
    duration_labels: torch.Tensor,
    *,
    switch_weight: float = 1.0,
    duration_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Mask switch ``-100`` and both duration sentinels (``-1``, ``-100``)."""

    if switch_logits.shape[:-1] != switch_labels.shape:
        raise ValueError("switch logits/labels shapes do not align")
    if duration_logits.shape[:-1] != duration_labels.shape:
        raise ValueError("duration logits/labels shapes do not align")
    switch_valid = switch_labels.ne(IGNORE_INDEX)
    duration_valid = duration_labels.ge(0)
    switch_loss = (
        F.cross_entropy(switch_logits[switch_valid], switch_labels[switch_valid])
        if switch_valid.any()
        else switch_logits.sum() * 0.0
    )
    duration_loss = (
        F.cross_entropy(duration_logits[duration_valid], duration_labels[duration_valid])
        if duration_valid.any()
        else duration_logits.sum() * 0.0
    )
    return {
        "loss": switch_weight * switch_loss + duration_weight * duration_loss,
        "switch_loss": switch_loss,
        "duration_loss": duration_loss,
        "switch_count": switch_valid.sum(),
        "duration_count": duration_valid.sum(),
    }
