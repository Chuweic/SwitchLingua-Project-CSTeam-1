"""Runnable Member C example: Member B records -> streaming batch -> prefix windows.

Uses Member B's DemoTokenizer; replace it with a Hugging Face fast tokenizer
and real word-level language IDs in the full pipeline.
"""

from examples.member_b_label_example import DemoTokenizer
from src.switchlingua_labels import encode_and_label
from src.streaming_dataloader import create_streaming_dataloader, iter_prefix_windows


def make_record(words, word_language_ids, language_pair):
    record = encode_and_label(DemoTokenizer(), words, word_language_ids)
    record["language_pair"] = language_pair
    return record


def main():
    records = [
        make_record(["أنا", "تابعت", "honestly"], ["ar", "ar", "en"], "Arabic-English"),
        make_record(["Creo", "que", "it's", "fine"], ["es", "es", "en", "en"], "Spanish-English"),
    ]

    loader = create_streaming_dataloader(records, batch_size=2)
    batch = next(iter(loader))
    print("lid_vocab:", loader.dataset.lid_vocab)
    for field in ("input_ids", "attention_mask", "current_lid", "switch_labels", "duration_labels"):
        print(f"{field}:\n{batch[field]}")
    print("language_pair:", batch["language_pair"])
    print("causal_mask[1] (1 = may attend; padding keys are 0):")
    print(batch["causal_mask"][1, 0].int())

    print("\nPrefix-only windows for the Arabic-English utterance:")
    for window in iter_prefix_windows(records[0]):
        print(
            f"t={window['position']}  prefix={window['prefix_input_ids']}  "
            f"lid={window['prefix_language_ids']}  "
            f"switch={window['switch_label']}  duration={window['duration_label']}"
        )


if __name__ == "__main__":
    main()
