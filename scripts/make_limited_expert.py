# scripts/make_limited_expert.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, List

import pandas as pd

import re
from typing import Sequence, Tuple

_TEXT_DECODER_PREFIX = "text_decoder."

# Keep it simple + robust: identify subtypes by substring
def _is_text_decoder_attn(layer: str) -> bool:
    # includes self_attn and cross_attention (your actual names)
    # "cross_attention" appears in your printout, not "cross_attn"
    return layer.startswith(_TEXT_DECODER_PREFIX) and (
        ".self_attn." in layer or ".cross_attention." in layer
    )

def _is_text_decoder_ffn(layer: str) -> bool:
    return layer.startswith(_TEXT_DECODER_PREFIX) and ".ffn." in layer

def _write_top_bottom(
    df_pool: pd.DataFrame,
    *,
    n: int,
    out_dir: Path,
    stem: str,
) -> None:
    """
    Write:
      {stem}_top_{n}.csv
      {stem}_bottom_{n}.csv

    from df_pool sorted by 'ap'.
    """
    if df_pool.empty:
        print(f"[make_limited_expert:{stem}] Skipping: 0 rows matched.")
        return

    df_top = df_pool.sort_values("ap", ascending=False).head(n)
    df_bot = df_pool.sort_values("ap", ascending=True).head(n)

    top_file = out_dir / f"{stem}_top_{n}.csv"
    bot_file = out_dir / f"{stem}_bottom_{n}.csv"

    print(f"[make_limited_expert:{stem}] pool={len(df_pool)}, top={len(df_top)}, bottom={len(df_bot)}")
    print(f"[make_limited_expert:{stem}] Writing:\n  - {top_file}\n  - {bot_file}")

    df_top.to_csv(top_file, index=False)
    df_bot.to_csv(bot_file, index=False)

def _expertise_dir(root_dir: str, task: str, model_name: str, language: str) -> Path:
    """
    Base expertise directory (without responses_tok{k}).

    Expected layout:
      {root_dir}/Speech/{task}/{model_name}/sense/{language}/expertise/
        responses_tok{k}/expertise.csv
    """
    return (
        Path(root_dir)
        / "Speech"
        / task
        / model_name
        / "sense"
        / language
        / "expertise"
    )

def _sense_dir(root_dir: str, task: str, model_name: str) -> Path:
    """
    Base sense directory:
      {root_dir}/Speech/{task}/{model_name}/sense/
        <language>/
    """
    return Path(root_dir) / "Speech" / task / model_name / "sense"


def _iter_languages(sense_dir: Path) -> List[str]:
    """
    Return sorted list of language directory names under sense/.
    A language is considered valid if it has: <lang>/expertise/responses_tok*/expertise.csv
    """
    langs: List[str] = []
    if not sense_dir.is_dir():
        return langs
    for p in sorted(sense_dir.iterdir()):
        if not p.is_dir():
            continue
        exp_base = p / "expertise"
        if sorted(exp_base.glob("responses_tok*/expertise.csv")):
            langs.append(p.name)
    return langs

def _iter_tok_expertise_csvs(exp_base_dir: Path) -> List[Path]:
    """
    Return sorted list of:
      exp_base_dir/responses_tok*/expertise.csv
    """
    csvs = sorted(exp_base_dir.glob("responses_tok*/expertise.csv"))
    return csvs


def _make_limited_from_one_csv(src_csv: Path, threshold: int, out_dir: Optional[Path] = None) -> None:
    """
    Read one expertise.csv and write limited files into out_dir (default: src_csv.parent).
    """
    if not src_csv.is_file():
        raise FileNotFoundError(f"expertise.csv not found: {src_csv}")

    exp_dir = out_dir if out_dir is not None else src_csv.parent
    exp_dir.mkdir(parents=True, exist_ok=True)

    half = int(threshold // 2)

    print(f"[make_limited_expert] Reading: {src_csv}")
    df = pd.read_csv(src_csv)

    if "ap" not in df.columns:
        raise ValueError(f"'ap' column not found in {src_csv}. Columns: {list(df.columns)}")
    if "layer" not in df.columns:
        raise ValueError(f"'layer' column not found in {src_csv}. Columns: {list(df.columns)}")

    # Be robust to non-string entries in 'layer'
    df["layer"] = df["layer"].astype(str)

    modules = ["speech_encoder", "text_encoder", "text_decoder"]
    total_rows = len(df)
    print(f"[make_limited_expert] Total rows in expertise.csv: {total_rows}")

    for module in modules:
        df_mod = df[df["layer"].str.startswith(f"{module}.")].copy()

        if df_mod.empty:
            print(f"[make_limited_expert] Skipping {module}: 0 rows matched.")
            continue

        # Sort by AP and slice
        df_top = df_mod.sort_values("ap", ascending=False).head(half)
        df_bot = df_mod.sort_values("ap", ascending=True).head(half)
        df_both = pd.concat([df_top, df_bot], axis=0, ignore_index=True)
        df_top2 = df_mod.sort_values("ap", ascending=False).head(threshold)
        df_bot2 = df_mod.sort_values("ap", ascending=True).head(threshold)

        top_file = exp_dir / f"{module}_expertise_limited_{half}_top.csv"
        bottom_file = exp_dir / f"{module}_expertise_limited_{half}_bottom.csv"
        both_file = exp_dir / f"{module}_expertise_limited_{threshold}_both.csv"
        top_file2 = exp_dir / f"{module}_expertise_limited_{threshold}_top.csv"
        bottom_file2 = exp_dir / f"{module}_expertise_limited_{threshold}_bottom.csv"

        print(
            f"[make_limited_expert:{module}] rows={len(df_mod)}, "
            f"top={len(df_top)}, bottom={len(df_bot)}, both={len(df_both)}"
        )
        print(
            f"[make_limited_expert:{module}] Writing:\n"
            f"  - {top_file}\n"
            f"  - {bottom_file}\n"
            f"  - {both_file}"
        )

        df_top.to_csv(top_file, index=False)
        df_bot.to_csv(bottom_file, index=False)
        df_both.to_csv(both_file, index=False)
        df_top2.to_csv(top_file2, index=False)
        df_bot2.to_csv(bottom_file2, index=False)

        if module == "text_decoder":
            # Filter FROM FULL expertise.csv rows for text_decoder, then pick top/bottom N
            df_attn = df[df["layer"].apply(_is_text_decoder_attn)].copy()
            df_ffn  = df[df["layer"].apply(_is_text_decoder_ffn)].copy()

            for n in (1000, 2000):
                _write_top_bottom(df_attn, n=n, out_dir=exp_dir, stem="text_decoder_attn_expertise")
                _write_top_bottom(df_ffn,  n=n, out_dir=exp_dir, stem="text_decoder_ffn_expertise")

def make_limited_expert_all_toks(
    model_name: str,
    language: str,
    threshold: int,
    *,
    root_dir: str,
    task: str,
    out_subdir: Optional[str] = None,
) -> None:
    """
    For every:
      {root_dir}/Speech/{task}/{model_name}/sense/{language}/expertise/responses_tok*/expertise.csv

    create limited-expertise CSVs.

    By default, outputs are written alongside each expertise.csv:
      .../responses_tok{k}/(speech_encoder_expertise_limited_..., ...)

    If out_subdir is provided (e.g., "limited"), outputs go to:
      .../responses_tok{k}/{out_subdir}/(files...)
    """
    base_dir = _expertise_dir(root_dir=root_dir, task=task, model_name=model_name, language=language)
    csvs = _iter_tok_expertise_csvs(base_dir)

    if not csvs:
        raise FileNotFoundError(f"No expertise.csv found under: {base_dir}/responses_tok*/expertise.csv")

    print(f"[make_limited_expert_all_toks] Found {len(csvs)} tok dirs under: {base_dir}")

    for src_csv in csvs:
        tok_dir = src_csv.parent  # .../responses_tok{k}
        out_dir = (tok_dir / out_subdir) if out_subdir else tok_dir
        print(f"[make_limited_expert_all_toks] Processing: {src_csv} -> {out_dir}")
        _make_limited_from_one_csv(src_csv=src_csv, threshold=threshold, out_dir=out_dir)

def make_limited_expert_all_languages(
    model_name: str,
    threshold: int,
    *,
    root_dir: str,
    task: str,
    out_subdir: Optional[str] = None,
) -> None:
    """
    Process ALL languages under:
      {root_dir}/Speech/{task}/{model_name}/sense/<language>/expertise/responses_tok*/expertise.csv
    """
    sense_dir = _sense_dir(root_dir=root_dir, task=task, model_name=model_name)
    langs = _iter_languages(sense_dir)

    if not langs:
        raise FileNotFoundError(
            f"No languages with expertise.csv found under: {sense_dir}/<language>/expertise/responses_tok*/expertise.csv"
        )

    print(f"[make_limited_expert_all_languages] Found {len(langs)} languages under: {sense_dir}")

    n_ok = 0
    n_fail = 0
    for lang in langs:
        try:
            make_limited_expert_all_toks(
                model_name=model_name,
                language=lang,
                threshold=threshold,
                root_dir=root_dir,
                task=task,
                out_subdir=out_subdir,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[make_limited_expert_all_languages][WARN] Failed language={lang}: {e}")

    print(f"[make_limited_expert_all_languages] Done. ok={n_ok}, failed={n_fail}")