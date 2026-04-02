import argparse
import json
import pandas as pd
from typing import List, Tuple

from sacrebleu.metrics import BLEU, CHRF
import jiwer


# ====================================================
# Load predictions + gold
# ====================================================
def load_pred_gold(
    csv_path: str,
    json_path: str,
    task: str,
    pred_col: str = "sentence",
) -> Tuple[List[str], List[str]]:
    """
    Loads predictions and references.
    Automatically SKIPS rows where reference (gold) is missing or empty.
    """
    # Load predictions
    df = pd.read_csv(csv_path)
    preds_all = df[pred_col].astype(str).tolist()

    # Load JSON gold
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clips = data["clips"]

    preds, refs = [], []

    for i, pred in enumerate(preds_all):
        if i >= len(clips):
            break

        clip = clips[i]

        # Choose reference by task
        if task.lower() == "asr":
            ref = clip.get("transcription", None)
        else:
            ref = clip.get("en_transcription", None)

        # Skip if missing
        if ref is None or str(ref).strip() == "":
            continue

        preds.append(pred)
        refs.append(str(ref))

    print(f"[INFO] Loaded {len(preds)} valid examples for task='{task}' "
          f"(skipped {len(preds_all) - len(preds)} rows with missing gold).")

    return preds, refs


# ====================================================
# ASR normalization
# ====================================================
def normalize_asr(texts: List[str]) -> List[str]:
    return [t.strip().lower() for t in texts]


# ====================================================
# ASR evaluation
# ====================================================
def evaluate_asr(preds: List[str], refs: List[str]):
    preds_norm = normalize_asr(preds)
    refs_norm = normalize_asr(refs)

    wer = jiwer.wer(refs_norm, preds_norm)
    cer = jiwer.cer(refs_norm, preds_norm)

    print("=== ASR Evaluation ===")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")

    return {"wer": wer, "cer": cer}


# ====================================================
# MT evaluation (S2T / T2T)
# ====================================================
def evaluate_mt(preds: List[str], refs: List[str]):
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF(char_order=6, word_order=2, beta=2, lowercase=False)

    bleu = bleu_metric.corpus_score(preds, [refs]).score
    chrf = chrf_metric.corpus_score(preds, [refs]).score
    combined = 0.6 * bleu + 0.4 * chrf

    print("=== MT Evaluation (S2T / T2T) ===")
    print(f"BLEU:   {bleu:.2f}")
    print(f"chrF++: {chrf:.2f}")
    print(f"0.6 * BLEU + 0.4 * chrF: {combined:.2f}")

    return {"bleu": bleu, "chrf": chrf, "combined": combined}


# ====================================================
# Main (argparse)
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR/S2T/T2T predictions.")

    parser.add_argument("--asr_csv", type=str, required=True,
                        help="Path to prediction CSV (created_sentence_*.csv)")
    parser.add_argument("--json_gold", type=str, required=True,
                        help="Path to gold JSON (with clips[])")
    parser.add_argument("--task", type=str, required=True,
                        choices=["asr", "s2t_translation", "t2t_translation"],
                        help="Evaluation task type")

    args = parser.parse_args()

    # Load pred + gold
    preds, refs = load_pred_gold(args.asr_csv, args.json_gold, task=args.task)

    # Evaluate
    if args.task.lower() == "asr":
        evaluate_asr(preds, refs)
    else:
        evaluate_mt(preds, refs)
