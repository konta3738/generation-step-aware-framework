#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create text_decoder_expertise.csv from expertise.csv, then compute
layer-wise Spearman correlation heatmaps across speech-side vs text-side languages.

Directory assumptions:
  .../Speech/{dir1}/seamless-m4t-v2-large/sense/{lang}_speech_VC/expertise/responses_tok{k}/expertise.csv

Outputs:
  1) text_decoder_expertise.csv in each responses_tok{k} dir
  2) one heatmap per decoder layer:
       spearman__text_decoder__layerXX__speechX_textY__responses_tok{k}.png
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
LAYER_NUM_RGX = re.compile(r"text_decoder.*?layers\.(\d+)", re.IGNORECASE)


def safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")
        return None


def list_token_dirs(expertise_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not expertise_dir.exists():
        return out
    for p in sorted([x for x in expertise_dir.iterdir() if x.is_dir()]):
        if TOK_DIR_RGX.match(p.name):
            out[p.name] = p
    return out


def discover_langs_speech_vc(root: Path, whitelist: Optional[List[str]] = None) -> List[str]:
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


def collect_tok_names(root_s2t: Path, root_t2t: Path, s2t_langs: List[str], t2t_langs: List[str]) -> List[str]:
    tok_names = set()

    for lang in s2t_langs:
        tok_names |= set(list_token_dirs(root_s2t / f"{lang}_speech_VC" / "expertise").keys())

    for lang in t2t_langs:
        tok_names |= set(list_token_dirs(root_t2t / f"{lang}_speech_VC" / "expertise").keys())

    return sorted(tok_names)


def parse_text_decoder_layer(layer_value: str) -> Optional[int]:
    if not isinstance(layer_value, str):
        return None
    m = LAYER_NUM_RGX.search(layer_value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def make_text_decoder_expertise_csv(expertise_csv: Path, out_csv: Path) -> bool:
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


def prepare_text_decoder_csvs_for_tok(root: Path, langs: List[str], tok: str) -> Dict[str, Path]:
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


def compute_layerwise_spearman_from_files(
    file1: Path,
    file2: Path,
    value_col: str = "ap",
    key_cols: Optional[List[str]] = None,
    num_layers: int = 24,
) -> List[float]:
    """
    Returns a list of Spearman correlations, one per layer.
    Alignment follows the same idea as your spearman_ap.py, but done separately per layer.
    """
    usecols = ["layer", value_col]
    if key_cols is not None:
        for c in key_cols:
            if c not in usecols:
                usecols.append(c)

    df1 = pd.read_csv(file1, usecols=usecols)
    df2 = pd.read_csv(file2, usecols=usecols)

    if "layer" not in df1.columns or "layer" not in df2.columns:
        raise ValueError("Both files must contain a 'layer' column.")

    df1 = df1.copy()
    df2 = df2.copy()

    df1["decoder_layer_idx"] = df1["layer"].astype(str).map(parse_text_decoder_layer)
    df2["decoder_layer_idx"] = df2["layer"].astype(str).map(parse_text_decoder_layer)

    corrs: List[float] = []

    for layer_idx in range(num_layers):
        sub1 = df1[df1["decoder_layer_idx"] == layer_idx].copy()
        sub2 = df2[df2["decoder_layer_idx"] == layer_idx].copy()

        if sub1.empty or sub2.empty:
            corrs.append(np.nan)
            continue

        if key_cols is None:
            if len(sub1) != len(sub2):
                corrs.append(np.nan)
                continue

            x = sub1[value_col].to_numpy()
            y = sub2[value_col].to_numpy()
        else:
            cols1 = list(key_cols) + [value_col]
            cols2 = list(key_cols) + [value_col]

            sub1 = sub1[cols1].rename(columns={value_col: f"{value_col}_1"})
            sub2 = sub2[cols2].rename(columns={value_col: f"{value_col}_2"})

            merged = sub1.merge(sub2, on=key_cols, how="inner")

            if merged.empty:
                corrs.append(np.nan)
                continue

            x = merged[f"{value_col}_1"].to_numpy()
            y = merged[f"{value_col}_2"].to_numpy()

        if len(x) < 2 or len(y) < 2:
            corrs.append(np.nan)
            continue

        corr, _ = spearmanr(x, y)
        corrs.append(float(corr) if pd.notna(corr) else np.nan)

    return corrs


def plot_single_layer_heatmap(
    mat: np.ndarray,
    speech_labels: List[str],
    text_labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    im = ax.imshow(mat, interpolation="nearest", cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Speech-side language", fontsize=14)
    ax.set_ylabel("Text-side language", fontsize=14)

    ax.set_xticks(range(len(speech_labels)))
    ax.set_xticklabels(speech_labels, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(text_labels)))
    ax.set_yticklabels(text_labels, fontsize=13)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Spearman correlation", fontsize=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            label = "nan" if np.isnan(val) else f"{val:.2f}"
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=13, color="black", fontweight="bold"
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] {out_path}")


def make_layerwise_spearman_heatmaps(
    *,
    root_s2t: Path,
    root_t2t: Path,
    out_dir: Path,
    lang_whitelist: Optional[List[str]] = None,
    value_col: str = "ap",
    key_cols: Optional[List[str]] = None,
    num_layers: int = 24,
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

        speech_labels = [f"speech_{lang}" for lang in s2t_langs]
        text_labels = [f"text_{lang}" for lang in t2t_langs]

        # mats[layer_idx][text_i, speech_j]
        mats = [np.full((len(text_labels), len(speech_labels)), np.nan, dtype=float) for _ in range(num_layers)]

        for i, text_lang in enumerate(t2t_langs):
            for j, speech_lang in enumerate(s2t_langs):
                file_text = t2t_csvs.get(text_lang)
                file_speech = s2t_csvs.get(speech_lang)

                if file_text is None or file_speech is None:
                    continue

                try:
                    layer_corrs = compute_layerwise_spearman_from_files(
                        file1=file_text,
                        file2=file_speech,
                        value_col=value_col,
                        key_cols=key_cols,
                        num_layers=num_layers,
                    )

                    for layer_idx, corr in enumerate(layer_corrs):
                        mats[layer_idx][i, j] = corr

                    corr_str = ", ".join(
                        [f"L{idx}:{c:.3f}" if not np.isnan(c) else f"L{idx}:nan" for idx, c in enumerate(layer_corrs)]
                    )
                    print(f"[OK] {tok} | text_{text_lang} vs speech_{speech_lang} -> {corr_str}")

                except Exception as e:
                    print(f"[WARN] Failed for {tok} | text_{text_lang} vs speech_{speech_lang}: {e}")

        # save one heatmap per layer
        for layer_idx in range(num_layers):
            out_path = out_dir / tok / f"spearman__text_decoder__layer{layer_idx:02d}__speechX_textY__{tok}.png"
            title = f"Speech vs Text (text_decoder layer {layer_idx} AP Spearman) — {tok}"
            plot_single_layer_heatmap(
                mat=mats[layer_idx],
                speech_labels=speech_labels,
                text_labels=text_labels,
                title=title,
                out_path=out_path,
            )

    print(f"\n[DONE] All layer-wise heatmaps saved to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create text_decoder_expertise.csv from expertise.csv and compute "
            "layer-wise speech-vs-text Spearman heatmaps."
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
        default="./_spearman_heatmaps_layerwise",
        help="Directory to save layer-wise heatmaps",
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
    parser.add_argument(
        "--num-layers",
        type=int,
        default=24,
        help="Number of decoder layers (default: 24)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    key_cols = args.keys if args.keys else None

    make_layerwise_spearman_heatmaps(
        root_s2t=Path(args.root_s2t),
        root_t2t=Path(args.root_t2t),
        out_dir=Path(args.out_dir),
        lang_whitelist=args.lang,
        value_col=args.value_col,
        key_cols=key_cols,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()