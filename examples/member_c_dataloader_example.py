import argparse
from pathlib import Path

from examples.member_b_label_example import DemoTokenizer
from src.switchlingua_labels import encode_and_label, load_fast_tokenizer, records_from_parquet
from src.streaming_dataloader import create_streaming_dataloader, iter_prefix_windows


def make_record(words, word_language_ids, language_pair):
    record = encode_and_label(DemoTokenizer(), words, word_language_ids)
    record["language_pair"] = language_pair
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/processed/train.parquet")
    parser.add_argument("--language-pair", default="Arabic-English")
    parser.add_argument("--max-length", type=int, default=24)
    args = parser.parse_args()

    parquet = Path(args.parquet)
    if parquet.exists():
        tokenizer = load_fast_tokenizer()
        records = records_from_parquet(
            str(parquet),
            tokenizer,
            language_pairs=[args.language_pair],
            max_rows_per_pair=2,
            max_utterances_per_row=1,
        )
        pad_token_id = tokenizer.pad_token_id
        source = "real SwitchLingua parquet (automatic weak word LIDs)"
    else:
        records = [
            make_record(["أنا", "تابعت", "honestly"], ["ar", "ar", "en"], "Arabic-English"),
            make_record(["Creo", "que", "it's", "fine"], ["es", "es", "en", "en"], "Spanish-English"),
        ]
        pad_token_id = 0
        source = "demo fallback; put train.parquet under data/processed for real output"

    loader = create_streaming_dataloader(
        records, batch_size=2, pad_token_id=pad_token_id, max_length=args.max_length
    )
    batch = next(iter(loader))
    print("source:", source)
    print("lid_vocab:", loader.dataset.lid_vocab)
    if parquet.exists():
        for index, record in enumerate(records):
            print(f"record[{index}] text:", record["text"])
            print(f"record[{index}] word_lids:", record["word_language_ids"])
    for field in ("input_ids", "attention_mask", "current_lid", "switch_labels", "duration_labels"):
        print(f"{field}:\n{batch[field]}")
    print("language_pair:", batch["language_pair"])
    print("causal_mask[1] (1 = may attend; padding keys are 0):")
    print(batch["causal_mask"][1, 0].int())

    if parquet.exists():
        display_tokens = tokenizer.convert_ids_to_tokens(records[0]["input_ids"])
    else:
        display_tokens = [str(token_id) for token_id in records[0]["input_ids"]]

    print("\nToken-level predictive labels:")
    print(
        f"{'Position':<10}"
        f"{'Token':<20}"
        f"{'LID':<10}"
        f"{'Switch':<10}"
        f"{'Duration':<10}"
    )
    for position, (token, lid, switch, duration) in enumerate(
        zip(
            display_tokens,
            records[0]["subword_language_ids"],
            records[0]["switch_labels"],
            records[0]["duration_labels"],
        )
    ):
        print(
            f"{position:<10}"
            f"{token:<20}"
            f"{str(lid):<10}"
            f"{switch:<10}"
            f"{duration:<10}"
        )

    print("\nSelected prefix-only windows for the first utterance:")
    windows = list(iter_prefix_windows(records[0]))
    selected = windows[:5]
    first_switch = next((window for window in windows if window["switch_label"] == 1), None)
    if first_switch is not None and first_switch not in selected:
        selected.append(first_switch)
    for window in selected:
        print(
            f"t={window['position']}  prefix={window['prefix_input_ids']}  "
            f"lid={window['prefix_language_ids']}  "
            f"switch={window['switch_label']}  duration={window['duration_label']}"
        )


if __name__ == "__main__":
    main()
