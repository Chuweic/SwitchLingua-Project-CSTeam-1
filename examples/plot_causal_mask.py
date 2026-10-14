"""Draw the causal attention mask produced by ``collate_streaming_batch``.

Saves ``docs/figures/causal_mask.png``. The mask is computed by the real
Member C code from a padded two-utterance batch, not drawn by hand.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from examples.member_b_label_example import DemoTokenizer
from src.switchlingua_labels import IGNORE_INDEX, encode_and_label
from src.streaming_dataloader import SwitchLinguaStreamingDataset, collate_streaming_batch


OUTPUT_PATH = "docs/figures/causal_mask.png"
HIGHLIGHT_ROW = 2  # "que": the position that predicts the switch to English

ALLOWED, FUTURE, PADDING = 0, 1, 2
COLORS = ["#2E6FD8", "#E4E7EC", "#9AA1AD"]


def build_batch():
    long_record = encode_and_label(DemoTokenizer(), ["أنا", "تابعت", "honestly"], ["ar", "ar", "en"])
    record = encode_and_label(DemoTokenizer(), ["Creo", "que", "it's", "fine"], ["es", "es", "en", "en"])
    dataset = SwitchLinguaStreamingDataset([long_record, record])
    return collate_streaming_batch([dataset[0], dataset[1]]), record


def main():
    batch, record = build_batch()
    row = 1
    mask = batch["causal_mask"][row, 0]
    attention_mask = batch["attention_mask"][row]
    seq_len = mask.size(0)

    tokens = ["[CLS]", "Creo", "que", "it's", "fine", "[SEP]", "[PAD]"]
    languages = ["—", "es", "es", "en", "en", "—", "—"]
    switch = batch["switch_labels"][row].tolist()

    categories = [[ALLOWED] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i, j]:
                categories[i][j] = ALLOWED
            elif not attention_mask[j]:
                categories[i][j] = PADDING
            else:
                categories[i][j] = FUTURE

    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=200)
    ax.imshow(categories, cmap=ListedColormap(COLORS), vmin=0, vmax=2)

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels([f"{tok}\n({lang})" for tok, lang in zip(tokens, languages)], fontsize=10)
    ax.set_yticklabels([f"t={i}  {tok}" for i, tok in enumerate(tokens)], fontsize=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Key position j (what the model can look at)", fontsize=11, labelpad=10)
    ax.set_ylabel("Query position t (where the prediction is made)", fontsize=11)

    ax.set_xticks([x - 0.5 for x in range(seq_len + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(seq_len + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(seq_len):
        for j in range(seq_len):
            symbol = "✓" if categories[i][j] == ALLOWED else "×"
            color = "white" if categories[i][j] != FUTURE else "#8A919C"
            ax.text(j, i, symbol, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.add_patch(Rectangle((-0.5, HIGHLIGHT_ROW - 0.5), seq_len, 1, fill=False, edgecolor="#E8590C", linewidth=3, zorder=5))

    for i, label in enumerate(switch):
        text = "ignored" if label == IGNORE_INDEX else f"switch = {label}"
        weight = "bold" if i == HIGHLIGHT_ROW else "normal"
        color = "#E8590C" if i == HIGHLIGHT_ROW else "#3A3F47"
        ax.text(seq_len - 0.3, i, text, ha="left", va="center", fontsize=10, color=color, fontweight=weight)
    ax.text(seq_len - 0.3, -0.85, "target at t\n(is t+1 a switch?)", ha="left", va="bottom", fontsize=9, color="#3A3F47")

    fig.text(
        0.5,
        -0.04,
        "Example: at t=2 the model sees only [CLS] Creo que, yet must predict that t+1 (it's) switches es → en.",
        ha="center",
        fontsize=10,
        color="#E8590C",
    )

    ax.legend(
        handles=[
            Patch(color=COLORS[ALLOWED], label="Allowed: j ≤ t and real token"),
            Patch(color=COLORS[FUTURE], label="Blocked: future token (j > t)"),
            Patch(color=COLORS[PADDING], label="Blocked: padding key"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle("Causal attention mask for streaming switch prediction", fontsize=13, fontweight="bold", y=1.04)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    print("Saved", OUTPUT_PATH)


if __name__ == "__main__":
    main()
