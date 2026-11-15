import json

from src.person4_analysis import build_model_summary, discover_runs


def _write_run(path, backbone, mode, f1=0.5, pair_f1=0.4):
    payload = {
        "overall": {
            "anticipatory_f1": f1,
            "precision": 0.5,
            "recall": 0.5,
            "duration_accuracy": 0.6,
        },
        "by_language_pair": {
            "Japanese-English": {
                "anticipatory_f1": pair_f1,
                "duration_accuracy": 0.6,
            }
        },
        "universality_std": 0.07,
        "run": {"backbone": backbone, "punctuation_mode": mode},
        "history": {"best_epoch": 3},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_runs_prefers_eight_epoch_neutral_files(tmp_path):
    _write_run(tmp_path / "train_xlmr_neutral.json", "xlm-roberta-base", "neutral", f1=0.51)
    _write_run(tmp_path / "converge_xlmr_neutral_8ep.json", "xlm-roberta-base", "neutral", f1=0.61)

    runs = discover_runs(tmp_path)

    assert runs["XLM-R:C"]["path"].name == "converge_xlmr_neutral_8ep.json"
    assert runs["XLM-R:C"]["status"] == "final"


def test_discover_runs_can_force_fair_three_epoch_comparison(tmp_path):
    _write_run(tmp_path / "train_xlmr_neutral.json", "xlm-roberta-base", "neutral", f1=0.51)
    _write_run(tmp_path / "converge_xlmr_neutral_8ep.json", "xlm-roberta-base", "neutral", f1=0.61)

    runs = discover_runs(tmp_path, prefer_convergence=False)

    assert runs["XLM-R:C"]["path"].name == "train_xlmr_neutral.json"
    assert runs["XLM-R:C"]["status"] == "provisional"


def test_discover_runs_can_select_only_final_neutral_models(tmp_path):
    _write_run(tmp_path / "converge_xlmr_neutral_8ep.json", "xlm-roberta-base", "neutral", f1=0.61)
    _write_run(tmp_path / "converge_mbert_neutral_8ep.json", "bert-base-multilingual-cased", "neutral", f1=0.60)

    runs = discover_runs(tmp_path, prefer_convergence=True, approaches={"C"})

    assert set(runs) == {"XLM-R:C", "mBERT:C"}
    assert all(item["status"] == "final" for item in runs.values())


def test_discover_runs_marks_three_epoch_fallback_provisional(tmp_path):
    _write_run(tmp_path / "train_xlmr_neutral.json", "xlm-roberta-base", "neutral")

    runs = discover_runs(tmp_path)

    assert runs["XLM-R:C"]["path"].name == "train_xlmr_neutral.json"
    assert runs["XLM-R:C"]["status"] == "provisional"


def test_build_model_summary_includes_naive_gain_and_status(tmp_path):
    run_path = tmp_path / "train_xlmr_neutral.json"
    _write_run(run_path, "xlm-roberta-base", "neutral", f1=0.61)
    naive_path = tmp_path / "naive_baseline_14pairs_neutral.json"
    naive_path.write_text(json.dumps({"overall": {"anticipatory_f1": 0.49}}), encoding="utf-8")
    runs = discover_runs(tmp_path)

    table = build_model_summary(runs, tmp_path)

    row = table.iloc[0]
    assert row["Anticipatory F1"] == 0.61
    assert row["Naive F1"] == 0.49
    assert row["F1 gain over naive"] == 0.12
    assert row["Result status"] == "provisional"
