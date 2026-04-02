#!/usr/bin/env python3
"""
!python merge_responses.py \
  --src-root /content/multimodal-lang-neuron/set_appropriate_path_2 \
  --dst-root /content/multimodal-lang-neuron/expertise_calculation
"""
import os
import re
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def list_pkl_with_index(folder: Path):
    """Return list of (path, idx) for files matching ^\d+\.pkl$ in given folder."""
    out = []
    for f in folder.glob("*.pkl"):
        m = re.fullmatch(r"(\d+)\.pkl", f.name)
        if m:
            out.append((f, int(m.group(1))))
    return sorted(out, key=lambda x: x[1])

def next_index(folder: Path, pad=5):
    """Get next index for 00000.pkl style files."""
    items = list_pkl_with_index(folder)
    if not items:
        return 0
    return max(idx for _, idx in items) + 1

def main(src_root: Path, dst_root: Path, pad: int = 5):
    # Copy the entire tree first
    if dst_root.exists():
        print(f"[Info] Destination already exists: {dst_root}")
    else:
        print(f"[Copy] {src_root} -> {dst_root}")
        shutil.copytree(src_root, dst_root)

    sense_subpath = Path("Speech/asr/seamless-m4t-v2-large/sense")
    src_sense = src_root / sense_subpath
    dst_sense = dst_root / sense_subpath

    assert src_sense.exists(), f"Source sense dir not found: {src_sense}"
    dst_sense.mkdir(parents=True, exist_ok=True)

    lang_dirs = sorted([p for p in src_sense.iterdir() if p.is_dir()])

    for tgt_dir in tqdm(lang_dirs, desc="Per-language merge"):
        tgt_name = tgt_dir.name
        src_resp = src_sense / tgt_name / "responses"
        dst_resp = dst_sense / tgt_name / "responses"

        if not src_resp.exists():
            print(f"[Warn] No responses in source: {src_resp}")
            continue

        dst_resp.mkdir(parents=True, exist_ok=True)

        start_idx = next_index(dst_resp, pad)
        cur_idx = start_idx

        other_langs = [p for p in lang_dirs if p.name != tgt_name]

        print(f"\n[Target] {tgt_name}")
        print(f"  - Existing files in dest: {len(list(dst_resp.glob('*.pkl')))}")
        print(f"  - Start index for new copies: {cur_idx}")

        for other_dir in other_langs:
            other_resp = src_sense / other_dir.name / "responses"
            if not other_resp.exists():
                continue

            other_pkls = list_pkl_with_index(other_resp)
            if not other_pkls:
                continue

            print(f"    + Adding {len(other_pkls)} from {other_dir.name}")
            for src_file, _ in other_pkls:
                new_name = f"{cur_idx:0{pad}d}.pkl"
                dst_file = dst_resp / new_name
                shutil.copy2(src_file, dst_file)
                cur_idx += 1

        added = cur_idx - start_idx
        total = len(list(dst_resp.glob('*.pkl')))
        print(f"  -> Added {added} files. Now total: {total}")

    print(f"\n[Done] Mixed responses for each language in: {dst_sense}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy a set_appropriate_path_2 tree and merge response PKLs from all languages into each language directory."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Path to the original set_appropriate_path_2 directory.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Path to the new expertise_calculation directory.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=5,
        help="Zero-padding width for output filenames (default: 5 -> 00000.pkl).",
    )
    args = parser.parse_args()

    main(args.src_root, args.dst_root, args.pad)
