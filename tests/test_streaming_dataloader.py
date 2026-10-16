import math

import pytest
import torch
from torch import nn
from transformers import XLMRobertaConfig, XLMRobertaModel

from src.switchlingua_labels import IGNORE_INDEX, NON_SWITCH_DURATION, encode_and_label
from src.streaming_dataloader import (
    SwitchLinguaStreamingDataset,
    build_causal_mask,
    causal_mask_for_backbone,
    collate_streaming_batch,
    create_streaming_dataloader,
    iter_prefix_windows,
    make_split_loaders,
    multitask_cross_entropy,
    to_additive_mask,
)
from src.naive_baseline import CurrentTokenBaseline, evaluate_by_pair


class FakeEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __call__(self, words, *, is_split_into_words, return_attention_mask, padding=False, max_length=None, truncation=False):
        input_ids = [1]
        word_ids = [None]
        for word_index, word in enumerate(words):
            pieces = [word] if len(word) <= 2 else [word[:2], word[2:]]
            for piece_index, _ in enumerate(pieces):
                input_ids.append(10 + word_index * 10 + piece_index)
                word_ids.append(word_index)
        input_ids.append(2)
        word_ids.append(None)
        return FakeEncoding(input_ids, [1] * len(input_ids), word_ids)


def make_record(words, languages, language_pair="Arabic-English"):
    record = encode_and_label(FakeTokenizer(), words, languages)
    record["language_pair"] = language_pair
    return record


@pytest.fixture
def records():
    return [
        make_record(["ana", "tabaat", "honestly"], ["ar", "ar", "en"]),
        make_record(["ok", "yes"], ["en", "en"], language_pair="Spanish-English"),
    ]


def test_dataset_converts_member_b_output_to_tensors(records):
    dataset = SwitchLinguaStreamingDataset(records)
    item = dataset[0]

    assert dataset.lid_vocab == {"ar": 0, "en": 1}
    assert item["input_ids"].tolist() == records[0]["input_ids"]
    assert item["switch_labels"].tolist() == records[0]["switch_labels"]
    assert item["duration_labels"].tolist() == records[0]["duration_labels"]
    assert item["current_lid"].tolist()[0] == IGNORE_INDEX
    assert item["current_lid"].tolist()[1:-1] == [0, 0, 0, 0, 1, 1]
    assert item["language_pair"] == "Arabic-English"


def test_dataset_rejects_misaligned_record(records):
    broken = dict(records[0])
    broken["switch_labels"] = broken["switch_labels"][:-1]
    with pytest.raises(ValueError):
        SwitchLinguaStreamingDataset([broken])


def test_truncation_keeps_labels_computed_on_full_utterance(records):
    item = SwitchLinguaStreamingDataset(records, max_length=4)[0]
    # Position 3 is the last kept token; its label still points at the true next sub-word.
    assert item["input_ids"].size(0) == 4
    assert item["switch_labels"].tolist() == records[0]["switch_labels"][:4]


def test_collate_pads_inputs_and_ignores_padded_labels(records):
    dataset = SwitchLinguaStreamingDataset(records)
    batch = collate_streaming_batch([dataset[0], dataset[1]], pad_token_id=0)

    long_len = len(records[0]["input_ids"])
    short_len = len(records[1]["input_ids"])
    assert batch["input_ids"].shape == (2, long_len)
    assert batch["attention_mask"][1].tolist() == [1] * short_len + [0] * (long_len - short_len)
    assert batch["input_ids"][1, short_len:].eq(0).all()
    for field in ("switch_labels", "duration_labels", "current_lid"):
        assert batch[field][1, short_len:].eq(IGNORE_INDEX).all()
    assert batch["causal_mask"].shape == (2, 1, long_len, long_len)
    assert batch["language_pair"] == ["Arabic-English", "Spanish-English"]


def test_duration_keeps_non_switch_marker_distinct_from_ignore(records):
    batch = collate_streaming_batch([SwitchLinguaStreamingDataset(records)[0]])
    duration = batch["duration_labels"][0].tolist()
    assert NON_SWITCH_DURATION in duration
    assert IGNORE_INDEX in duration


def test_causal_mask_blocks_future_and_padding():
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    mask = build_causal_mask(attention_mask)

    expected_full = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask[0, 0], expected_full)
    assert not mask[:, 0].triu(diagonal=1).any()
    assert not mask[1, 0, :, 2:].any()
    assert mask[1, 0].any(dim=-1).all()  # no fully masked query row


def test_additive_mask_is_zero_where_allowed():
    mask = build_causal_mask(torch.ones(1, 3, dtype=torch.long))
    additive = to_additive_mask(mask)
    assert additive[0, 0, 1, 0] == 0
    assert additive[0, 0, 0, 1] == torch.finfo(torch.float32).min


def test_dataloader_yields_batches(records):
    loader = create_streaming_dataloader(records, batch_size=2, pad_token_id=0)
    batch = next(iter(loader))
    assert batch["input_ids"].size(0) == 2
    assert batch["causal_mask"].dtype == torch.bool


def test_dataloader_requires_tokenizer_pad_id(records):
    with pytest.raises(ValueError, match="pad_token_id"):
        create_streaming_dataloader(records, batch_size=2)


def test_xlmr_cannot_see_future_tokens():
    torch.manual_seed(42)
    config = XLMRobertaConfig(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=1,
    )
    model = XLMRobertaModel(config).eval()
    original_ids = torch.tensor([[0, 10, 11, 12, 2]])
    changed_ids = torch.tensor([[0, 10, 11, 90, 2]])
    attention_mask = torch.ones_like(original_ids)
    causal_mask = build_causal_mask(attention_mask)
    backbone_mask = causal_mask_for_backbone(causal_mask)

    with torch.no_grad():
        original = model(
            input_ids=original_ids,
            attention_mask=backbone_mask,
        ).last_hidden_state
        changed = model(
            input_ids=changed_ids,
            attention_mask=backbone_mask,
        ).last_hidden_state

    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6)


def test_prefix_windows_contain_only_history(records):
    record = records[0]
    windows = list(iter_prefix_windows(record))

    valid_positions = [t for t, label in enumerate(record["switch_labels"]) if label != IGNORE_INDEX]
    assert [window["position"] for window in windows] == valid_positions
    for window in windows:
        t = window["position"]
        assert window["prefix_input_ids"] == record["input_ids"][: t + 1]
        assert len(window["prefix_language_ids"]) == t + 1
        assert window["switch_label"] == record["switch_labels"][t]
        assert window["duration_label"] == record["duration_labels"][t]


# --- Future-information leakage tests -------------------------------------


class TinyCausalSelfAttentionModel(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=16, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(64, hidden_size)
        self.layers = nn.ModuleList(nn.Linear(hidden_size, 3 * hidden_size) for _ in range(num_layers))
        self.switch_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, causal_mask):
        positions = torch.arange(input_ids.size(1))
        hidden = self.embedding(input_ids) + self.position(positions)[None]
        additive = to_additive_mask(causal_mask[:, 0], hidden.dtype)
        for layer in self.layers:
            query, key, value = layer(hidden).chunk(3, dim=-1)
            scores = query @ key.transpose(-1, -2) / math.sqrt(query.size(-1)) + additive
            hidden = hidden + torch.softmax(scores, dim=-1) @ value
        return self.switch_head(hidden)


def _replace_future(input_ids, attention_mask, t, generator):
    corrupted = input_ids.clone()
    seq_len = input_ids.size(1)
    if t + 1 < seq_len:
        random_ids = torch.randint(3, 128, (input_ids.size(0), seq_len - t - 1), generator=generator)
        future_real = attention_mask[:, t + 1 :].bool()
        corrupted[:, t + 1 :] = torch.where(future_real, random_ids, corrupted[:, t + 1 :])
    return corrupted


def test_no_future_leakage_when_future_tokens_are_replaced(records):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = TinyCausalSelfAttentionModel().eval()
    dataset = SwitchLinguaStreamingDataset(records)
    batch = collate_streaming_batch([dataset[i] for i in range(len(dataset))])

    with torch.no_grad():
        original = model(batch["input_ids"], batch["causal_mask"])
        for t in range(batch["input_ids"].size(1)):
            corrupted = _replace_future(batch["input_ids"], batch["attention_mask"], t, generator)
            output = model(corrupted, batch["causal_mask"])
            assert torch.allclose(output[:, : t + 1], original[:, : t + 1], atol=1e-6), f"leak at t={t}"


def test_leakage_check_detects_a_bidirectional_mask(records):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = TinyCausalSelfAttentionModel().eval()
    batch = collate_streaming_batch([SwitchLinguaStreamingDataset(records)[0]])
    bidirectional = batch["attention_mask"].bool()[:, None, None, :].expand_as(batch["causal_mask"])

    with torch.no_grad():
        original = model(batch["input_ids"], bidirectional)
        corrupted = _replace_future(batch["input_ids"], batch["attention_mask"], 0, generator)
        output = model(corrupted, bidirectional)
    assert not torch.allclose(output[:, :1], original[:, :1], atol=1e-6)


def test_collated_inputs_at_t_do_not_depend_on_future_words():
    base = make_record(["ana", "ok", "honestly"], ["ar", "en", "en"])
    changed = make_record(["ana", "ok", "zz"], ["ar", "en", "ar"])
    t = 3  # last sub-word of "ok"; the words after it differ
    base_item = SwitchLinguaStreamingDataset([base], lid_vocab={"ar": 0, "en": 1})[0]
    changed_item = SwitchLinguaStreamingDataset([changed], lid_vocab={"ar": 0, "en": 1})[0]

    for field in ("input_ids", "attention_mask", "current_lid"):
        assert torch.equal(base_item[field][: t + 1], changed_item[field][: t + 1])


def test_split_loaders_reuse_training_lid_vocab(records):
    train, validation, test = make_split_loaders(
        records,
        records,
        records,
        batch_size=2,
        pad_token_id=0,
    )
    assert validation.dataset.lid_vocab == train.dataset.lid_vocab
    assert test.dataset.lid_vocab == train.dataset.lid_vocab


def test_multitask_loss_masks_duration_minus_one_and_minus_100():
    switch_logits = torch.zeros(1, 4, 2, requires_grad=True)
    duration_logits = torch.zeros(1, 4, 3, requires_grad=True)
    result = multitask_cross_entropy(
        switch_logits,
        duration_logits,
        switch_labels=torch.tensor([[0, 1, -100, -100]]),
        duration_labels=torch.tensor([[-1, 2, -100, -100]]),
    )
    assert result["switch_count"].item() == 2
    assert result["duration_count"].item() == 1
    result["loss"].backward()


def test_current_token_baseline_ignores_invalid_labels():
    train = [{
        "input_ids": [10, 11, 0],
        "switch_labels": [1, 0, -100],
        "duration_labels": [2, -1, -100],
        "language_pair": "Arabic-English",
    }]
    model = CurrentTokenBaseline().fit(train)
    result = evaluate_by_pair(model, train)
    assert sum(model.global_counts.values()) == 2
    assert sum(model.global_durations.values()) == 1
    assert result["overall"]["support"] == 2
    assert result["overall"]["duration_support"] == 1
