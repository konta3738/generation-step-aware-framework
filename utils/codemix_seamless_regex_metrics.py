#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# pip install regex transformers sentencepiece
import regex
from transformers import AutoTokenizer

MODEL_NAME = "facebook/seamless-m4t-v2-large"

RE_HIRA = regex.compile(r"\p{Script=Hiragana}")
RE_KATA = regex.compile(r"\p{Script=Katakana}")
RE_HAN  = regex.compile(r"\p{Script=Han}")
RE_LAT  = regex.compile(r"\p{Script=Latin}")

RE_HIRA_ALL = regex.compile(r"\p{Script=Hiragana}")
RE_KATA_ALL = regex.compile(r"\p{Script=Katakana}")
RE_HAN_ALL  = regex.compile(r"\p{Script=Han}")
RE_LAT_ALL  = regex.compile(r"\p{Script=Latin}")

SP_SPACE = "▁"


def norm_sp_token(t: str) -> str:
    return t.replace(SP_SPACE, "")


def tokenize_sentence(tokenizer: AutoTokenizer, sentence: str) -> List[str]:
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    return tokenizer.convert_ids_to_tokens(ids)


# ----------------------------
# Token-level buckets (Latin/CJK/Mix/Other)
# ----------------------------
def token_bucket_latin_cjk_mix_other(token_str: str) -> str:
    """
    Token-level bucket:
      - Mix: token contains BOTH Latin AND any CJK script (Hira/Kata/Han)
      - Latin: contains Latin and NO CJK
      - CJK: contains CJK and NO Latin
      - Other: neither Latin nor CJK (digits/punct/emoji/etc or empty)
    """
    t = norm_sp_token(token_str)
    if not t:
        return "Other"
    if t.startswith("<") and t.endswith(">"):
        return "Other"

    has_lat = RE_LAT.search(t) is not None
    has_cjk = (RE_HIRA.search(t) is not None) or (RE_KATA.search(t) is not None) or (RE_HAN.search(t) is not None)

    if has_lat and has_cjk:
        return "Mix"
    if has_lat:
        return "Latin"
    if has_cjk:
        return "CJK"
    return "Other"


def count_token_buckets(tokens: List[str]) -> Dict[str, int]:
    counts = {"Latin": 0, "CJK": 0, "Mix": 0, "Other": 0}
    for tok in tokens:
        counts[token_bucket_latin_cjk_mix_other(tok)] += 1
    return counts


# ----------------------------
# Character-level script counts (Hira/Kata/Han ratios separately)
# ----------------------------
def count_char_scripts(text: str) -> Dict[str, int]:
    text = "" if text is None else str(text)

    n_hira = len(RE_HIRA_ALL.findall(text))
    n_kata = len(RE_KATA_ALL.findall(text))
    n_han  = len(RE_HAN_ALL.findall(text))
    n_lat  = len(RE_LAT_ALL.findall(text))

    total = len(text)
    other = max(0, total - (n_hira + n_kata + n_han + n_lat))
    cjk = n_hira + n_kata + n_han

    return {
        "char_total": total,
        "char_hiragana": n_hira,
        "char_katakana": n_kata,
        "char_han": n_han,
        "char_latin": n_lat,
        "char_other": other,
        "char_cjk": cjk,
    }


# ----------------------------
# Metrics
# ----------------------------
def m_index_from_distribution(p: List[float]) -> float:
    n = len(p)
    if n <= 1:
        return 0.0
    return (1.0 - sum(pi * pi for pi in p)) / (n - 1)


def i_index_from_sequence(seq: List[str], ignore: Optional[set] = None) -> float:
    """
    switches/(N-1) after removing labels in ignore
    """
    if ignore:
        seq = [x for x in seq if x not in ignore]
    if len(seq) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return switches / (len(seq) - 1)


def per_row_metrics(tokenizer: AutoTokenizer, sentence: str) -> Dict[str, float]:
    sentence = "" if sentence is None else str(sentence)

    toks = tokenize_sentence(tokenizer, sentence)
    tok_counts = count_token_buckets(toks)
    tok_total = sum(tok_counts.values())

    # token rates (over all tokens)
    def safe_rate(x: int, denom: int) -> float:
        return (x / denom) if denom else 0.0

    out: Dict[str, float] = {}

    out["tok_total"] = float(tok_total)
    out["tok_latin"] = float(tok_counts["Latin"])
    out["tok_cjk"] = float(tok_counts["CJK"])
    out["tok_mix"] = float(tok_counts["Mix"])
    out["tok_other"] = float(tok_counts["Other"])

    out["rate_tok_latin"] = safe_rate(tok_counts["Latin"], tok_total)
    out["rate_tok_cjk"] = safe_rate(tok_counts["CJK"], tok_total)
    out["rate_tok_mix"] = safe_rate(tok_counts["Mix"], tok_total)
    out["rate_tok_other"] = safe_rate(tok_counts["Other"], tok_total)

    # Latin vs CJK proportions:
    # Option 1: ignore Mix+Other (pure comparison)
    denom_lc = tok_counts["Latin"] + tok_counts["CJK"]
    out["p_tok_latin_given_pure_lc"] = safe_rate(tok_counts["Latin"], denom_lc)
    out["p_tok_cjk_given_pure_lc"] = safe_rate(tok_counts["CJK"], denom_lc)

    # Option 2: include Mix by splitting half/half (optional, can remove if unwanted)
    # Here we provide it because it's often useful.
    denom_lc_soft = tok_counts["Latin"] + tok_counts["CJK"] + tok_counts["Mix"]
    out["p_tok_latin_with_mix_half"] = safe_rate(tok_counts["Latin"] + 0.5 * tok_counts["Mix"], denom_lc_soft)
    out["p_tok_cjk_with_mix_half"] = safe_rate(tok_counts["CJK"] + 0.5 * tok_counts["Mix"], denom_lc_soft)

    # M-index
    # 2-way M over pure Latin/CJK (ignoring Mix/Other)
    if denom_lc:
        pL = tok_counts["Latin"] / denom_lc
        pC = tok_counts["CJK"] / denom_lc
        out["M_index_2way_pure_LatinCJK"] = m_index_from_distribution([pL, pC])
    else:
        out["M_index_2way_pure_LatinCJK"] = 0.0

    # 4-way M over (Latin/CJK/Mix/Other)
    if tok_total:
        p4 = [
            tok_counts["Latin"] / tok_total,
            tok_counts["CJK"] / tok_total,
            tok_counts["Mix"] / tok_total,
            tok_counts["Other"] / tok_total,
        ]
        out["M_index_4way_tok"] = m_index_from_distribution(p4)
    else:
        out["M_index_4way_tok"] = 0.0

    # I-index
    # For switching, you need a sequence. We compute:
    # - I_pure: treat Mix as its own label, ignore Other
    seq = [token_bucket_latin_cjk_mix_other(t) for t in toks]
    out["I_index_3label_LatinCJKMix_ignoreOther"] = i_index_from_sequence(seq, ignore={"Other"})

    # - I_binary: map Mix -> Mix (ignored) and compute switches between pure Latin/CJK only
    seq_bin = [x for x in seq if x in ("Latin", "CJK")]
    out["I_index_2label_pure_LatinCJK"] = i_index_from_sequence(seq_bin, ignore=None)

    # character-level script breakdown
    ch = count_char_scripts(sentence)
    out.update({k: float(v) for k, v in ch.items()})

    char_total = ch["char_total"]
    out["rate_char_hiragana"] = safe_rate(ch["char_hiragana"], char_total)
    out["rate_char_katakana"] = safe_rate(ch["char_katakana"], char_total)
    out["rate_char_han"] = safe_rate(ch["char_han"], char_total)
    out["rate_char_cjk"] = safe_rate(ch["char_cjk"], char_total)
    out["rate_char_latin"] = safe_rate(ch["char_latin"], char_total)
    out["rate_char_other"] = safe_rate(ch["char_other"], char_total)

    return out


def main(csv_path: str, out_path: Optional[str] = None, sentence_col: str = "sentence") -> None:
    df = pd.read_csv(csv_path)
    if sentence_col not in df.columns:
        raise ValueError(f"Column '{sentence_col}' not found. Available: {list(df.columns)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    metrics = [per_row_metrics(tokenizer, s) for s in df[sentence_col].fillna("").astype(str).tolist()]
    mdf = pd.DataFrame(metrics)
    out_df = pd.concat([df, mdf], axis=1)

    # Corpus-level summaries (token-weighted)
    tok_latin = int(mdf["tok_latin"].sum())
    tok_cjk = int(mdf["tok_cjk"].sum())
    tok_mix = int(mdf["tok_mix"].sum())
    tok_other = int(mdf["tok_other"].sum())
    tok_total = tok_latin + tok_cjk + tok_mix + tok_other

    print("=== Corpus-level (token-weighted) ===")
    print(f"tok_total={tok_total}  Latin={tok_latin}  CJK={tok_cjk}  Mix={tok_mix}  Other={tok_other}")
    if tok_total:
        print(
            f"rate_tok_latin={tok_latin/tok_total:.4f}  "
            f"rate_tok_cjk={tok_cjk/tok_total:.4f}  "
            f"rate_tok_mix={tok_mix/tok_total:.4f}  "
            f"rate_tok_other={tok_other/tok_total:.4f}"
        )

    denom_lc = tok_latin + tok_cjk
    if denom_lc:
        pL = tok_latin / denom_lc
        pC = tok_cjk / denom_lc
        print(f"p_tok_latin_given_pure_lc={pL:.4f}  p_tok_cjk_given_pure_lc={pC:.4f}")
        print(f"M_index_2way_pure_LatinCJK={m_index_from_distribution([pL, pC]):.4f}")

    if tok_total:
        p4 = [tok_latin/tok_total, tok_cjk/tok_total, tok_mix/tok_total, tok_other/tok_total]
        print(f"M_index_4way_tok={m_index_from_distribution(p4):.4f}")

    # Character-level summary
    char_total = int(mdf["char_total"].sum())
    char_hira = int(mdf["char_hiragana"].sum())
    char_kata = int(mdf["char_katakana"].sum())
    char_han = int(mdf["char_han"].sum())
    char_lat = int(mdf["char_latin"].sum())
    char_other = int(mdf["char_other"].sum())
    print("\n=== Corpus-level (character-weighted, raw sentence) ===")
    print(f"char_total={char_total}  Hira={char_hira}  Kata={char_kata}  Han={char_han}  Latin={char_lat}  Other={char_other}")
    if char_total:
        print(
            f"rate_char_hiragana={char_hira/char_total:.4f}  "
            f"rate_char_katakana={char_kata/char_total:.4f}  "
            f"rate_char_han={char_han/char_total:.4f}  "
            f"rate_char_latin={char_lat/char_total:.4f}  "
            f"rate_char_other={char_other/char_total:.4f}"
        )

    if out_path:
        out_df.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--sentence_col", type=str, default="sentence")
    args = ap.parse_args()

    main(args.csv_path, out_path=args.out_path, sentence_col=args.sentence_col)