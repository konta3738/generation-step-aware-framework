#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create text_decoder_expertise.csv from expertise.csv, then compute
Spearman correlation heatmaps across speech-side vs text-side languages.

Directory assumptions:
  .../Speech/{dir1}/seamless-m4t-v2-large/sense/{lang}_speech_VC/expertise/responses_tok{k}/expertise.csv

Usage example:
python make_spearman_heatmaps.py \
  --root-s2t "./set_appropriate_path_2/Speech/s2t_translation/seamless-m4t-v2-large/sense" \
  --root-t2t "./set_appropriate_path_2/Speech/t2t_translation/seamless-m4t-v2-large/sense" \
  --out-dir "./_spearman_heatmaps" \
  --keys layer unit
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


TOK_DIR_RGX = re.compile(r"^responses_tok(\d+)$")


def safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")
        return None


def list_token_dirs(expertise_dir: Path) -> Dict[str, Path]:
    """
    Return mapping: tok_dir_name -> Path for directories like responses_tok2.
    """
    out: Dict[str, Path] = {}
    if not expertise_dir.exists():
        return out
    for p in sorted([x for x in expertise_dir.iterdir() if x.is_dir()]):
        if TOK_DIR_RGX.match(p.name):
            out[p.name] = p
    return out


def discover_langs_speech_vc(root: Path, whitelist: Optional[List[str]] = None) -> List[str]:
    """
    Discover lang keys from directories like "{lang}_speech_VC".
    """
    if not root.exists():
        return []
    langs: List[str] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        name = d.name
        if name.endswith("_speech_VC"):
            lang = name.replace("_speech_VC", "")
            if whitelist is None or lang in whitelist:
                langs.append(lang)
    return langs


def make_text_decoder_expertise_csv(expertise_csv: Path, out_csv: Path) -> bool:
    """
    Read expertise.csv, keep only rows where layer starts with 'text_decoder',
    and save them as text_decoder_expertise.csv.
    """
    df = safe_read_csv(expertise_csv)
    if df is None:
        return False

    if "layer" not in df.columns:
        print(f"[WARN] Missing 'layer' column in {expertise_csv}")
        return False

    text_df = df[df["layer"].astype(str).str.startswith("text_decoder")].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    text_df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv} ({len(text_df):,} rows)")
    return True


def compute_spearman_from_files(
    file1: Path,
    file2: Path,
    value_col: str = "ap",
    key_cols: Optional[List[str]] = None,
) -> float:
    """
    Same logic as your spearman_ap.py:
    - if key_cols is provided, align rows by those columns
    - otherwise, use row order
    Returns correlation only.
    """
    if key_cols is None:
        df1 = pd.read_csv(file1, usecols=[value_col])
        df2 = pd.read_csv(file2, usecols=[value_col])

        if len(df1) != len(df2):
            raise ValueError(
                f"Row count mismatch: {file1} has {len(df1)} rows, "
                f"but {file2} has {len(df2)} rows. Use key-based alignment."
            )

        x = df1[value_col].to_numpy()
        y = df2[value_col].to_numpy()

    else:
        cols1 = key_cols + [value_col]
        cols2 = key_cols + [value_col]

        df1 = pd.read_csv(file1, usecols=cols1)
        df2 = pd.read_csv(file2, usecols=cols2)

        df1 = df1.rename(columns={value_col: f"{value_col}_1"})
        df2 = df2.rename(columns={value_col: f"{value_col}_2"})

        merged = df1.merge(df2, on=key_cols, how="inner")

        if merged.empty:
            raise ValueError(f"No matching rows found after merge: {file1} vs {file2}")

        x = merged[f"{value_col}_1"].to_numpy()
        y = merged[f"{value_col}_2"].to_numpy()

    corr, _ = spearmanr(x, y)
    return float(corr)


def collect_tok_names(root_s2t: Path, root_t2t: Path, s2t_langs: List[str], t2t_langs: List[str]) -> List[str]:
    tok_names = set()

    for lang in s2t_langs:
        tok_names |= set(list_token_dirs(root_s2t / f"{lang}_speech_VC" / "expertise").keys())

    for lang in t2t_langs:
        tok_names |= set(list_token_dirs(root_t2t / f"{lang}_speech_VC" / "expertise").keys())

    return sorted(tok_names)


def prepare_text_decoder_csvs_for_tok(root: Path, langs: List[str], tok: str) -> Dict[str, Path]:
    """
    For a given root and tok dir, create text_decoder_expertise.csv for each language.
    Returns mapping: lang -> text_decoder_expertise.csv path
    """
    out: Dict[str, Path] = {}

    for lang in langs:
        tok_dir = root / f"{lang}_speech_VC" / "expertise" / tok
        expertise_csv = tok_dir / "expertise.csv"
        textdec_csv = tok_dir / "text_decoder_expertise.csv"

        if not expertise_csv.exists():
            print(f"[WARN] Missing expertise.csv: {expertise_csv}")
            continue

        ok = make_text_decoder_expertise_csv(expertise_csv, textdec_csv)
        if ok:
            out[lang] = textdec_csv

    return out


def plot_heatmap(
    mat: np.ndarray,
    speech_labels: List[str],
    text_labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    #plt.figure(figsize=(10, 8))
    plt.figure(figsize=(18, 16))
    ax = plt.gca()

    im = ax.imshow(mat, interpolation="nearest", cmap="coolwarm", vmin=-1.0, vmax=1.0)

    #ax.set_title(title, fontsize=18)
    ax.set_xlabel("Speech", fontsize=60)
    ax.set_ylabel("Text", fontsize=60)

    ax.set_xticks(range(len(speech_labels)))
    ax.set_xticklabels(speech_labels, rotation=45, ha="right", fontsize=80)
    ax.set_yticks(range(len(text_labels)))
    ax.set_yticklabels(text_labels, fontsize=80)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=40)
    #cbar.set_label("Spearman correlation", fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            label = "nan" if np.isnan(val) else f"{val:.2f}"
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=65, color="black", fontweight="bold"
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def make_spearman_heatmaps(
    *,
    root_s2t: Path,
    root_t2t: Path,
    out_dir: Path,
    lang_whitelist: Optional[List[str]] = None,
    value_col: str = "ap",
    key_cols: Optional[List[str]] = None,
) -> None:
    s2t_langs = discover_langs_speech_vc(root_s2t, whitelist=lang_whitelist)
    t2t_langs = discover_langs_speech_vc(root_t2t, whitelist=lang_whitelist)

    if not s2t_langs:
        raise SystemExit(f"[ERROR] No languages found under root_s2t: {root_s2t}")
    if not t2t_langs:
        raise SystemExit(f"[ERROR] No languages found under root_t2t: {root_t2t}")

    tok_names = collect_tok_names(root_s2t, root_t2t, s2t_langs, t2t_langs)
    if not tok_names:
        raise SystemExit("[ERROR] No responses_tok{k} directories found.")

    out_dir.mkdir(parents=True, exist_ok=True)

    for tok in tok_names:
        print(f"\n[INFO] Processing {tok}")

        s2t_csvs = prepare_text_decoder_csvs_for_tok(root_s2t, s2t_langs, tok)
        t2t_csvs = prepare_text_decoder_csvs_for_tok(root_t2t, t2t_langs, tok)

        #speech_labels = [f"speech_{lang}" for lang in s2t_langs]
        #text_labels = [f"text_{lang}" for lang in t2t_langs]
        speech_labels = [lang for lang in s2t_langs]
        text_labels = [lang for lang in t2t_langs]

        mat = np.full((len(text_labels), len(speech_labels)), np.nan, dtype=float)

        for i, text_lang in enumerate(t2t_langs):
            for j, speech_lang in enumerate(s2t_langs):
                file_text = t2t_csvs.get(text_lang)
                file_speech = s2t_csvs.get(speech_lang)

                if file_text is None or file_speech is None:
                    continue

                try:
                    corr = compute_spearman_from_files(
                        file1=file_text,
                        file2=file_speech,
                        value_col=value_col,
                        key_cols=key_cols,
                    )
                    mat[i, j] = corr
                    print(
                        f"[OK] {tok} | text_{text_lang} vs speech_{speech_lang} "
                        f"-> rho={corr:.6f}"
                    )
                except Exception as e:
                    print(
                        f"[WARN] Failed for {tok} | text_{text_lang} vs speech_{speech_lang}: {e}"
                    )

        out_path = out_dir / f"spearman__text_decoder__speechX_textY__{tok}.png"
        title = f"Speech vs Text (text_decoder AP Spearman) — {tok}"
        plot_heatmap(mat, speech_labels, text_labels, title, out_path)

    print(f"\n[DONE] All heatmaps saved to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create text_decoder_expertise.csv from expertise.csv and compute "
            "speech-vs-text Spearman heatmaps."
        )
    )
    parser.add_argument(
        "--root-s2t",
        type=str,
        required=True,
        help="Path to s2t sense root, e.g. .../s2t_translation/.../sense",
    )
    parser.add_argument(
        "--root-t2t",
        type=str,
        required=True,
        help="Path to t2t sense root, e.g. .../t2t_translation/.../sense",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./_spearman_heatmaps",
        help="Directory to save heatmaps",
    )
    parser.add_argument(
        "--lang",
        nargs="*",
        default=None,
        help="Optional language whitelist, e.g. --lang de en fr ja zh",
    )
    parser.add_argument(
        "--value-col",
        default="ap",
        help="Column to correlate (default: ap)",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=["layer", "unit"],
        help="Key columns for alignment, e.g. --keys uuid or --keys layer unit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    key_cols = args.keys if args.keys else None

    make_spearman_heatmaps(
        root_s2t=Path(args.root_s2t),
        root_t2t=Path(args.root_t2t),
        out_dir=Path(args.out_dir),
        lang_whitelist=args.lang,
        value_col=args.value_col,
        key_cols=key_cols,
    )


if __name__ == "__main__":
    main()