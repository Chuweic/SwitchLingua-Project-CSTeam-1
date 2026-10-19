from pathlib import Path

import pandas as pd

from src.switchlingua_labels import PairRestrictedWordLID


def main():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_parquet(
        root / "data" / "processed" / "validation.parquet"
    )

    pair_data = data[
        data["language_pair"] == "German-French"
    ]

    if pair_data.empty:
        raise ValueError("No German-French rows in validation data")

    sentences = pair_data.explode("utterances").copy()
    sentences = sentences[
        sentences["utterances"].apply(
            lambda text: isinstance(text, str) and bool(text.strip())
        )
    ]
    sentences = sentences.drop_duplicates(subset=["utterances"])
    sample = sentences.sample(
        n=min(50, len(sentences)),
        random_state=42,
    )

    lid = PairRestrictedWordLID()
    rows = []

    for sample_id, row in enumerate(
        sample.itertuples(index=False), start=1
    ):
        words, labels = lid.annotate(
            row.utterances,
            row.first_language,
            row.second_language,
        )

        for word_index, (word, label) in enumerate(
            zip(words, labels)
        ):
            rows.append({
                "sample_id": sample_id,
                "row_id": row.row_id,
                "text": row.utterances,
                "word_index": word_index,
                "word": word,
                "predicted_lid": label,
                "manual_lid": "",
                "notes": "",
            })

    output = root / "results" / "german_french_manual_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Sampled sentences: {len(sample)}")
    print(f"Output file: {output}")


if __name__ == "__main__":
    main()