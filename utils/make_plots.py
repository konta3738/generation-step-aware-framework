#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified plotting script (code1 + code2) with token-dir sweep.

Directory assumptions (expertise side):
  .../Speech/{dir1}/seamless-m4t-v2-large/sense/{dir2}/expertise/responses_tok{k}/<csvs>

Figure outputs (stacked plots, code2):
  ./figure/{dir1}/{dir2}/responses_tok{k}/<pngs>
  ./figure/{dir1}/ALL_LANGUAGES/responses_tok{k}/<pngs>

Overlap heatmaps (code1):
  Saved under --overlap-out (unchanged behavior).

Notes:
- dir2 is kept EXACT (= subdirectory name under sense_root) to avoid collisions such as:
  modality_s2t vs modality_t2t, lang_modality_neuron vs lang_de, etc.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# Shared helpers
# ============================================================

TOK_DIR_RGX = re.compile(r"^responses_tok(\d+)$")


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


def safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}")
        return None


# ============================================================
# ------------------------- CODE 1 ---------------------------
# Overlap heatmaps: Speech vs Text (text_decoder)
# Now: expertise/responses_tok{k}/{file}.csv
# ============================================================

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


def make_overlap_heatmaps(
    *,
    root_s2t: Path,
    root_t2t: Path,
    layer_max: int,
    out_dir: Path,
    k: int,
    lang_whitelist: Optional[List[str]] = None,
) -> None:
    """
    For each tok-dir found under expertise/, generate overlap heatmaps for:
      Speech langs (X) vs Text langs (Y),
    using text_decoder expertise CSVs.
    """
    kind_to_filename = {
        "bottom": f"text_decoder_expertise_limited_{k}_bottom.csv",
        "top":    f"text_decoder_expertise_limited_{k}_top.csv",
        "both":   f"text_decoder_expertise_limited_{2*k}_both.csv",
    }
    kind_display = {
        "bottom": f"bottom {k}",
        "top":    f"top {k}",
        "both":   f"both {2*k}",
    }

    layer_rgx = re.compile(r"^(speech_encoder|text_encoder|text_decoder).*?layers\.(\d+)", re.IGNORECASE)
    trail_unit_rgx = re.compile(r":(\d+)$")

    def parse_module_and_layer(layer_str: str):
        if not isinstance(layer_str, str):
            return None, None
        if layer_str.startswith("text_decoder.layer_norm"):
            return "text_decoder", layer_max - 1
        m = layer_rgx.search(layer_str)
        if not m:
            return None, None
        mod = m.group(1)
        try:
            idx = int(m.group(2))
        except Exception:
            return None, None
        return mod, idx

    def layer_unit_key(row) -> str:
        layer_str = str(row.get("layer", ""))
        if "unit" in row and pd.notna(row["unit"]):
            try:
                unit_str = str(int(row["unit"]))
            except Exception:
                unit_str = str(row["unit"])
        else:
            m = trail_unit_rgx.search(layer_str)
            unit_str = m.group(1) if m else "0"
        return f"{layer_str}##{unit_str}"

    def collect_sets_from_csv(csv_path: Path) -> List[set]:
        df = safe_read_csv(csv_path)
        per_layer_all = [set() for _ in range(layer_max)]
        if df is None or "layer" not in df.columns:
            return per_layer_all

        for _, row in df.iterrows():
            mod, idx = parse_module_and_layer(row["layer"])
            if mod != "text_decoder" or not isinstance(idx, int) or not (0 <= idx < layer_max):
                continue
            uid = layer_unit_key(row)
            per_layer_all[idx].add(uid)
        return per_layer_all

    def union_over_layers(list_of_sets_per_layer):
        out = set()
        for s in list_of_sets_per_layer:
            out |= s
        return out

    # Discover languages
    s2t_langs = discover_langs_speech_vc(root_s2t, whitelist=lang_whitelist)
    t2t_langs = discover_langs_speech_vc(root_t2t, whitelist=lang_whitelist)

    # 内部管理用ラベル（speech/text を区別）
    speech_labels = [f"speech_{l}" for l in s2t_langs]
    text_labels   = [f"text_{l}" for l in t2t_langs]

    # 表示用ラベル（言語名だけ）
    speech_ticklabels = s2t_langs
    text_ticklabels = t2t_langs

    # Collect tok-dirs (union across s2t and t2t)
    tok_names = set()
    for lang in s2t_langs:
        tok_names |= set(list_token_dirs(root_s2t / f"{lang}_speech_VC" / "expertise").keys())
    for lang in t2t_langs:
        tok_names |= set(list_token_dirs(root_t2t / f"{lang}_speech_VC" / "expertise").keys())
    tok_names = sorted(tok_names)

    if not tok_names:
        print("[WARN] No responses_tok{k} directories found for overlap heatmaps.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for tok in tok_names:
        for kind, filename in kind_to_filename.items():
            per_label_per_layer_all: Dict[str, List[set]] = {}

            # ---- S2T: speech_{lang} ----
            for lang in s2t_langs:
                tok_dir = root_s2t / f"{lang}_speech_VC" / "expertise" / tok
                csv_path = tok_dir / filename
                label = f"speech_{lang}"
                per_label_per_layer_all[label] = collect_sets_from_csv(csv_path)

            # ---- T2T: text_{lang} ----
            for lang in t2t_langs:
                tok_dir = root_t2t / f"{lang}_speech_VC" / "expertise" / tok
                csv_path = tok_dir / filename
                label = f"text_{lang}"
                per_label_per_layer_all[label] = collect_sets_from_csv(csv_path)

            sets_total_speech = [union_over_layers(per_label_per_layer_all[lbl]) for lbl in speech_labels]
            sets_total_text   = [union_over_layers(per_label_per_layer_all[lbl]) for lbl in text_labels]

            mat = np.zeros((len(text_labels), len(speech_labels)), dtype=int)
            for i, text_set in enumerate(sets_total_text):
                for j, speech_set in enumerate(sets_total_speech):
                    mat[i, j] = len(text_set.intersection(speech_set))

            # 行列を見やすく表示
            mat_df = pd.DataFrame(
                mat,
                index=text_ticklabels,      # 行: text 側
                columns=speech_ticklabels   # 列: speech 側
            )

            print("\n" + "=" * 80)
            print(f"[MATRIX] k={k} tok={tok}, kind={kind}")
            print(mat_df.to_string())
            print("=" * 80)

            display_kind = kind_display[kind]
            title = f"Speech vs Text (text_decoder) — {display_kind} — {tok}"
            out_path = out_dir / f"overlap__textdec__{kind}__speechX_textY__TOTAL__{tok}.png"

            # ===== ここから描画設定をかなり大きく =====
            plt.figure(figsize=(18, 16))
            ax = plt.gca()
            im = ax.imshow(mat, interpolation="nearest", cmap="viridis", aspect="auto")

            # タイトルも必要なら表示
            # ax.set_title(title, fontsize=34, fontweight="bold", pad=24)

            # 軸タイトル
            ax.set_xlabel("Speech", fontsize=60, labelpad=3)
            ax.set_ylabel("Text", fontsize=60, labelpad=3)

            # tick位置
            ax.set_xticks(range(len(speech_labels)))
            ax.set_yticks(range(len(text_labels)))

            # 表示ラベルは lang のみ
            ax.set_xticklabels(
                speech_ticklabels,
                rotation=0,
                ha="center",
                fontsize=80
            )
            ax.set_yticklabels(
                text_ticklabels,
                fontsize=80
            )

            # tick周り
            ax.tick_params(axis="x", labelsize=80, pad=14)
            ax.tick_params(axis="y", labelsize=80, pad=14)

            # 枠線
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
                spine.set_color("black")

            # カラーバー
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=28)

            # heatmap内の数字
            vmax = mat.max() if mat.size > 0 else 0
            threshold = vmax * 0.5

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    color = "white" if val > threshold else "black"
                    ax.text(
                        j, i, str(val),
                        ha="center",
                        va="center",
                        fontsize=85,
                        fontweight="bold",
                        color="white"
                    )

            plt.tight_layout(pad=1.5)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SAVE] {out_path}")
            generated.append(out_path)

    print(f"\n[DONE] Overlap heatmaps: generated {len(generated)} PNGs in {out_dir}")
# ============================================================
# ------------------------- CODE 2 ---------------------------
# Stacked plots: component + submodules, per dir2 + ALL_LANGUAGES
# Now: expertise/responses_tok{k}/{file}.csv
# Output: figure/{dir1}/{dir2}/responses_tok{k}/...
# ============================================================

AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
LEGEND_TITLE_FONTSIZE = 15

TARGET_FILES = {
    "speech_encoder": [
        "speech_encoder_expertise_limited_1000_top.csv",
        "speech_encoder_expertise_limited_1000_bottom.csv",
    ],
    "text_encoder": [
        "text_encoder_expertise_limited_1000_top.csv",
        "text_encoder_expertise_limited_1000_bottom.csv",
    ],
    "text_decoder": [
        "text_decoder_expertise_limited_1000_top.csv",
        "text_decoder_expertise_limited_1000_bottom.csv",
    ],
}

LANG_COLOR_MAP = {
    "de": "#1f77b4",
    "en": "#ff7f0e",
    "es": "#2ca02c",
    "fr": "#d62728",
    "ja": "#9467bd",
    "zh": "#8c564b",
}

BASE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

SPEECH_ENCODER_PRESET = {
    "ffn1_layer_norm":              "#1f77b4",
    "ffn1.intermediate_dense":      "#ff7f0e",
    "ffn1.output_dense":            "#2ca02c",
    "ffn2_layer_norm":              "#d62728",
    "ffn2.intermediate_dense":      "#9467bd",
    "ffn2.output_dense":            "#8c564b",
    "self_attn_layer_norm":         "#e377c2",
    "self_attn.linear_q":           "#7f7f7f",
    "self_attn.linear_k":           "#bcbd22",
    "self_attn.linear_v":           "#17becf",
    "self_attn.linear_out":         "#aec7e8",
    "conv_module.layer_norm":           "#ffbb78",
    "conv_module.pointwise_conv1":      "#98df8a",
    "conv_module.glu":                  "#ff9896",
    "conv_module.depthwise_conv":       "#c5b0d5",
    "conv_module.depthwise_layer_norm": "#c49c94",
    "conv_module.pointwise_conv2":      "#f7b6d2",
}

TEXT_DECODER_PRESET = {
    "self_attn_layer_norm": "#1f77b4",
    "self_attn.q_proj":     "#ff7f0e",
    "self_attn.k_proj":     "#2ca02c",
    "self_attn.v_proj":     "#d62728",
    "self_attn.out_proj":   "#9467bd",
    "cross_attention_layer_norm": "#8c564b",
    "cross_attention.q_proj":     "#e377c2",
    "cross_attention.k_proj":     "#7f7f7f",
    "cross_attention.v_proj":     "#bcbd22",
    "cross_attention.out_proj":   "#17becf",
    "ffn_layer_norm":       "#ff9896",
    "ffn.fc1":              "#c5b0d5",
    "ffn.fc2":              "#ffbb78",
}

TEXT_ENCODER_PRESET = {
    "self_attn_layer_norm": "#1f77b4",
    "self_attn.q_proj":     "#ff7f0e",
    "self_attn.k_proj":     "#2ca02c",
    "self_attn.v_proj":     "#d62728",
    "self_attn.out_proj":   "#9467bd",
    "ffn_layer_norm":       "#ff9896",
    "ffn.fc1":              "#c5b0d5",
    "ffn.fc2":              "#ffbb78",
}

SUBMODULE_COLORS: Dict[str, str] = {}
SUBMODULE_COLORS.update(SPEECH_ENCODER_PRESET)
SUBMODULE_COLORS.update(TEXT_DECODER_PRESET)
SUBMODULE_COLORS.update(TEXT_ENCODER_PRESET)


def get_submodule_color(_module: str, sub_name: str) -> str:
    if sub_name in SUBMODULE_COLORS:
        return SUBMODULE_COLORS[sub_name]
    used = set(SUBMODULE_COLORS.values())
    for c in BASE_COLORS:
        if c not in used:
            SUBMODULE_COLORS[sub_name] = c
            return c
    c = BASE_COLORS[len(used) % len(BASE_COLORS)]
    SUBMODULE_COLORS[sub_name] = c
    return c


KNOWN_TASK_PREFIXES = {"asr", "s2t", "t2t", "s2s", "en2x"}


def get_lang_key_from_dirname(dirname: str) -> str:
    """
    Heuristic for ALL_LANGUAGES coloring keys.
    Keeps old behavior but doesn't affect output dirs anymore.
    """
    parts = dirname.split("_")
    if len(parts) >= 2 and parts[0].lower() in KNOWN_TASK_PREFIXES:
        return parts[1]
    return parts[0]


def detect_subset_type(stem: str) -> str:
    for suf in ("_top", "_bottom"):
        if stem.endswith(suf):
            return suf[1:]
    return "unknown"


def subset_label_from_type(subset_type: str) -> str:
    if subset_type == "top":
        return "top_1000"
    if subset_type == "bottom":
        return "bottom_1000"
    return subset_type


def plot_stacked_from_dict(layer_to_series_dict, out_path: Path, color_map=None):
    if not layer_to_series_dict:
        return

    names = sorted(layer_to_series_dict.keys())
    layer_max = max(len(v) for v in layer_to_series_dict.values())

    padded = {}
    for name, vec in layer_to_series_dict.items():
        if len(vec) < layer_max:
            vec = vec + [0] * (layer_max - len(vec))
        padded[name] = vec

    x = range(layer_max)
    bottom = [0] * layer_max

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for name in names:
        vec = padded[name]
        color = None
        if color_map is not None and name in color_map and color_map[name] is not None:
            color = color_map[name]
        ax.bar(x, vec, bottom=bottom, label=name, color=color, edgecolor="black", linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vec)]

    ax.set_xlabel("Layer index", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Neuron count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=18)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_stacked_subcomponents(module: str, sub_counts: Dict[str, List[int]], out_path: Path, layer_max: int):
    if not sub_counts:
        return
    totals = {k: sum(v) for k, v in sub_counts.items()}
    sorted_subs = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    keep = [name for name, _ in sorted_subs]

    stacked_dict = {}
    color_map = {}
    for name in keep:
        vec = sub_counts[name]
        if len(vec) < layer_max:
            vec = vec + [0] * (layer_max - len(vec))
        stacked_dict[name] = vec
        color_map[name] = get_submodule_color(module, name)

    plot_stacked_from_dict(stacked_dict, out_path, color_map=color_map)


def run_stacked_for_sense_root(sense_root: Path, figure_root: Path, layer_max: int) -> List[Path]:
    """
    Enumerate:
      sense_root/{dir2}/expertise/responses_tok{k}/<csvs>
    Output:
      figure_root/{dir1}/{dir2}/responses_tok{k}/<pngs>
      figure_root/{dir1}/ALL_LANGUAGES/responses_tok{k}/<pngs>
    """
    sense_root = Path(sense_root)
    if not sense_root.exists():
        raise FileNotFoundError(f"Sense root not found: {sense_root}")

    # sense_root = .../Speech/{dir1}/seamless-m4t-v2-large/sense
    dir1 = sense_root.parent.parent.name  # {dir1}
    generated: List[Path] = []

    layer_rgx = re.compile(r"^(speech_encoder|text_encoder|text_decoder).*?layers\.(\d+)", re.IGNORECASE)

    def parse_module_and_layer(layer_str: str):
        if not isinstance(layer_str, str):
            return None, None
        m = layer_rgx.search(layer_str)
        if not m:
            return None, None
        mod = m.group(1)
        try:
            idx = int(m.group(2))
        except Exception:
            return None, None
        if not (0 <= idx < layer_max):
            return None, None
        return mod, idx

    def classify_component(layer_str: str) -> str:
        if not isinstance(layer_str, str):
            return "other"
        if "distance_embedding" in layer_str:
            return "ignore"
        s = layer_str.lower()
        if ("ffn" in s) or ("mlp" in s) or ("feed_forward" in s) or ("feedforward" in s):
            return "ffn"
        if ("attn" in s) or ("attention" in s) or ("mhsa" in s):
            return "attn"
        if ("conv" in s) or ("convolution" in s) or ("depthwise" in s):
            return "conv"
        return "other"

    def extract_subcomponent_name(layer_str: str):
        if not isinstance(layer_str, str):
            return None
        m = re.search(r"layers\.(\d+)\.(.+?)(?::\d+)?$", layer_str)
        if not m:
            return None
        return m.group(2)

    def count_per_layer_by_component(df: pd.DataFrame, module: str):
        has_conv = (module == "speech_encoder")
        comp_names = ["attn", "ffn"] + (["conv"] if has_conv else []) + ["other"]
        comp_counts = {c: [0] * layer_max for c in comp_names}
        if df is None or "layer" not in df.columns:
            return comp_counts
        for s in df["layer"]:
            mod, idx = parse_module_and_layer(s)
            if (mod == module) and isinstance(idx, int):
                c = classify_component(s)
                if c == "ignore":
                    continue
                if c == "conv" and not has_conv:
                    c = "other"
                if c not in comp_counts:
                    c = "other"
                comp_counts[c][idx] += 1
        return comp_counts

    def count_per_layer_by_subcomponent(df: pd.DataFrame, module: str, category: str):
        sub_counts: Dict[str, List[int]] = {}
        if df is None or "layer" not in df.columns:
            return sub_counts
        has_conv = (module == "speech_encoder")
        for s in df["layer"]:
            mod, idx = parse_module_and_layer(s)
            if (mod != module) or not isinstance(idx, int):
                continue
            c = classify_component(s)
            if c == "ignore":
                continue
            if c == "conv" and not has_conv:
                c = "other"
            if c != category:
                continue
            sub_name = extract_subcomponent_name(s)
            if sub_name is None:
                continue
            if sub_name not in sub_counts:
                sub_counts[sub_name] = [0] * layer_max
            sub_counts[sub_name][idx] += 1
        return sub_counts

    def out_dir_for_dir2(dir2: str, tok: str) -> Path:
        # figure/{dir1}/{dir2}/responses_tok{k}/
        d = figure_root / dir1 / dir2 / tok
        d.mkdir(parents=True, exist_ok=True)
        return d

    # gather tok names across all dir2
    dir2_dirs = sorted([p for p in sense_root.iterdir() if p.is_dir()])
    tok_names = set()
    for dir2_dir in dir2_dirs:
        tok_names |= set(list_token_dirs(dir2_dir / "expertise").keys())
    tok_names = sorted(tok_names)
    if not tok_names:
        print(f"[WARN] No responses_tok{k} dirs under: {sense_root}")
        return []

    TARGET_COMPONENTS = {
        "speech_encoder": {"attn", "conv", "ffn"},
        "text_encoder": {"attn", "ffn"},
        "text_decoder": {"attn", "ffn"},
    }

    for tok in tok_names:
        # aggregate for ALL_LANGUAGES (keyed by lang_key for colors only)
        module_comp_lang_counts = {"speech_encoder": {}, "text_encoder": {}, "text_decoder": {}}

        for dir2_dir in dir2_dirs:
            dir2 = dir2_dir.name  # keep exact
            tok_dir = dir2_dir / "expertise" / tok
            if not tok_dir.exists():
                continue

            lang_key = get_lang_key_from_dirname(dir2)

            for module, filenames in TARGET_FILES.items():
                for fname in filenames:
                    csv_path = tok_dir / fname
                    if not csv_path.exists():
                        continue
                    df = safe_read_csv(csv_path)
                    if df is None:
                        continue

                    subset_type = detect_subset_type(csv_path.stem)  # top/bottom
                    if subset_type not in {"top", "bottom"}:
                        continue

                    comp_counts = count_per_layer_by_component(df, module)

                    for comp_name, comp_vec in comp_counts.items():
                        if comp_name not in {"attn", "ffn", "conv"}:
                            continue
                        if module in {"text_encoder", "text_decoder"} and comp_name == "conv":
                            continue

                        # accumulate for ALL_LANGUAGES
                        module_comp_lang_counts.setdefault(module, {})
                        module_comp_lang_counts[module].setdefault(comp_name, {})
                        module_comp_lang_counts[module][comp_name].setdefault(subset_type, {})
                        module_comp_lang_counts[module][comp_name][subset_type].setdefault(lang_key, [0] * layer_max)

                        acc = module_comp_lang_counts[module][comp_name][subset_type][lang_key]
                        for i in range(layer_max):
                            acc[i] += comp_vec[i] if i < len(comp_vec) else 0

                        # detailed per-dir2 (submodules)
                        allowed_details = {"attn", "ffn", "conv"} if module == "speech_encoder" else {"attn", "ffn"}
                        if comp_name not in allowed_details:
                            continue

                        sub_counts = count_per_layer_by_subcomponent(df, module, comp_name)
                        if not sub_counts:
                            continue

                        subset_label = subset_label_from_type(subset_type)
                        out_dir = out_dir_for_dir2(dir2, tok)
                        detailed_png = out_dir / f"{module}_{subset_label}_{comp_name}_submodules.png"
                        plot_stacked_subcomponents(module, sub_counts, detailed_png, layer_max)
                        generated.append(detailed_png)
                        print(f"[SAVE] {detailed_png}")

        # ALL_LANGUAGES plots
        all_dir = out_dir_for_dir2("ALL_LANGUAGES", tok)

        for module, comp_dict in module_comp_lang_counts.items():
            for comp_name, region_dict in comp_dict.items():
                if comp_name not in TARGET_COMPONENTS.get(module, set()):
                    continue
                for region in ("top", "bottom"):
                    if region not in region_dict:
                        continue
                    lang_dict = region_dict[region]
                    if not lang_dict:
                        continue

                    out_path = all_dir / f"ALL_LANGUAGES__{module}__{region}__{comp_name}__layer_counts.png"
                    color_map = {lk: LANG_COLOR_MAP.get(lk, None) for lk in lang_dict.keys()}
                    plot_stacked_from_dict(lang_dict, out_path, color_map=color_map)
                    generated.append(out_path)
                    print(f"[SAVE] {out_path}")

    print(f"\n[DONE] {sense_root} → Generated {len(generated)} PNGs (per responses_tok*, no CSVs).")
    return generated


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified plotting script (code1+code2) with expertise/responses_tok{k} sweep."
    )

    ap.add_argument("--layer-max", type=int, default=24, help="Max layers (default: 24 => 0..23)")
    ap.add_argument("--do-overlap", action="store_true", help="Run overlap heatmaps (code1).")
    ap.add_argument("--do-stacked", action="store_true",
                    help="Run stacked plots (code2). If neither flag is set, run both.")
    ap.add_argument("--lang", nargs="*", default=None, help="Optional language whitelist, e.g. --lang de en fr")

    # code1 roots
    ap.add_argument("--root-s2t", type=str, default=None, help="ROOT_S2T (points to .../sense)")
    ap.add_argument("--root-t2t", type=str, default=None, help="ROOT_T2T (points to .../sense)")
    ap.add_argument("--overlap-out", type=str, default="./_overlap_heatmaps_textdecoder_speech_vs_text",
                    help="Output dir for overlap heatmaps")

    # code2 roots
    ap.add_argument("--sense-roots", nargs="*", default=None,
                    help="One or more SENSE_ROOT paths (each points to .../sense)")
    ap.add_argument("--figure-root", type=str, default="./figure",
                    help="Root dir for stacked plots (default: ./figure)")
    ap.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Subset size for top/bottom files. both is assumed to be 2*k (default: 1000)"
    )

    return ap.parse_args()


def main():
    args = parse_args()

    layer_max = int(args.layer_max)
    lang_whitelist = args.lang if args.lang else None

    run_overlap = args.do_overlap
    run_stacked = args.do_stacked
    if not run_overlap and not run_stacked:
        run_overlap = True
        run_stacked = True

    if run_overlap:
        if args.root_s2t is None or args.root_t2t is None:
            raise SystemExit("[ERROR] --do-overlap requires --root-s2t and --root-t2t")
        make_overlap_heatmaps(
            root_s2t=Path(args.root_s2t),
            root_t2t=Path(args.root_t2t),
            layer_max=layer_max,
            out_dir=Path(args.overlap_out),
            k=args.k,
            lang_whitelist=lang_whitelist,
        )

    if run_stacked:
        if not args.sense_roots:
            raise SystemExit("[ERROR] --do-stacked requires --sense-roots (one or more)")
        fig_root = Path(args.figure_root)
        all_gen: List[Path] = []
        for sr in args.sense_roots:
            all_gen.extend(run_stacked_for_sense_root(Path(sr), fig_root, layer_max))
        print(f"[DONE] Stacked plots total generated: {len(all_gen)}")


if __name__ == "__main__":
    main()