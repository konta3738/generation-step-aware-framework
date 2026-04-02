#!/usr/bin/env python
"""
Compute simple statistics (mean, variance, etc.) for each response_name
from cached response PKLs, using read_responses_from_cached.

It expects directories of the form:

  ./set_appropriate_path_2/Speech/{task}/seamless-m4t-v2-large/sense/{LANG}_speech_VC/responses/

where:
  task ∈ {asr, s2t_translation, t2t_translation}
  LANG ∈ {de, es, fr, ja, zh}

For each (task, LANG), we:
  - read all *.pkl in the responses/ dir (no sibling span)
  - ignore labels
  - compute statistics for each response_name (layer key)
  - save results to a CSV
  - additionally, create plots of mean and variance per module
    (speech_encoder, text_encoder, text_decoder), per task.

Plots:
  For each task ∈ {asr, s2t_translation, t2t_translation}
  and each module ∈ {speech_encoder, text_encoder, text_decoder},
  we make 2 plots (mean and variance), excluding response_names that
  contain "adapter". That is 3 (tasks) × 3 (modules) × 2 (metrics) = 18 PNGs.
"""

import argparse
import pathlib
import typing as t

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from selfcond.responses import read_responses_from_cached


def compute_stats_for_array(arr: np.ndarray) -> dict:
    """
    Given an array of shape (C, N), compute global stats over all entries.
    """
    flat = arr.astype(np.float32).ravel()
    if flat.size == 0:
        return {
            "n_elements": 0,
            "mean": np.nan,
            "var": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "mean_abs": np.nan,
            "mean_sq": np.nan,
        }

    mean = float(np.mean(flat))
    var = float(np.var(flat))
    std = float(np.sqrt(var))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    q25, q50, q75 = np.percentile(flat, [25, 50, 75]).astype(np.float32)
    mean_abs = float(np.mean(np.abs(flat)))
    mean_sq = float(np.mean(flat ** 2))  # energy

    return {
        "n_elements": int(flat.size),
        "mean": mean,
        "var": var,
        "std": std,
        "min": min_val,
        "max": max_val,
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "mean_abs": mean_abs,
        "mean_sq": mean_sq,
    }


def make_plots(
    df: pd.DataFrame,
    tasks: t.Sequence[str],
    plots_dir: pathlib.Path,
) -> None:
    """
    Create plots for mean and variance, per task and per module.

    - modules: speech_encoder, text_encoder, text_decoder
    - metrics: mean, var
    - excludes response_names that contain "adapter"
    - grouping: for each task+module, we first average the metric across langs
      per response_name, then plot one point per response_name.
    """
    modules = ["speech_encoder", "text_encoder", "text_decoder"]
    metrics = ["mean", "var"]

    plots_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for module in modules:
            # filter rows for this task & module, excluding adapter
            mask = (
                (df["task"] == task)
                & df["response_name"].str.contains(module)
                & ~df["response_name"].str.contains("adapter")
            )
            df_sub = df[mask]
            if df_sub.empty:
                print(f"[INFO] No data for task={task}, module={module}; skipping plots.")
                continue

            # Group by response_name and average across langs, to avoid
            # plotting duplicate layers multiple times.
            df_grouped = (
                df_sub.groupby("response_name", as_index=False)[metrics].mean()
                .sort_values(by="response_name")
            )

            x = np.arange(len(df_grouped))
            for metric in metrics:
                y = df_grouped[metric].values

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x, y, marker="o", linestyle="-")
                ax.set_title(f"{task} - {module} - {metric}")
                ax.set_xlabel("response_name (sorted index)")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)

                # Optional: keep tick labels light, since there can be many layers
                ax.set_xticks([])
                fig.tight_layout()

                safe_module = module.replace(".", "_")
                fname = f"{task}_{safe_module}_{metric}.png"
                out_path = plots_dir / fname
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                print(f"[INFO] Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute simple statistics over cached responses (no labels) "
                    "and generate summary plots."
    )
    parser.add_argument(
        "--base-dir",
        type=pathlib.Path,
        default=pathlib.Path("./set_appropriate_path_2/Speech"),
        help="Base directory for Speech responses "
             "(default: ./set_appropriate_path_2/Speech)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="seamless-m4t-v2-large",
        help="Model name used in the path (default: seamless-m4t-v2-large)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="asr,s2t_translation,t2t_translation",
        help="Comma-separated list of tasks to process "
             "(default: asr,s2t_translation,t2t_translation)",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="de,es,fr,ja,zh",
        help="Comma-separated list of langs to process "
             "(default: de,es,fr,ja,zh)",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=pathlib.Path("./response_stats.csv"),
        help="Output CSV file (default: ./response_stats.csv)",
    )
    parser.add_argument(
        "--plots-dir",
        type=pathlib.Path,
        default=pathlib.Path("./response_stats_plots"),
        help="Directory to save PNG plots (default: ./response_stats_plots)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information while processing",
    )

    args = parser.parse_args()

    base_dir: pathlib.Path = args.base_dir
    model_name: str = args.model_name
    tasks: t.List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    langs: t.List[str] = [l.strip() for l in args.langs.split(",") if l.strip()]
    out_csv: pathlib.Path = args.out_csv
    plots_dir: pathlib.Path = args.plots_dir
    verbose: bool = args.verbose

    records: t.List[dict] = []

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Tasks: {tasks}")
    print(f"[INFO] Langs: {langs}")
    print(f"[INFO] Output CSV: {out_csv}")
    print(f"[INFO] Plots dir: {plots_dir}")

    for task in tasks:
        for lang in langs:
            cached_dir = (
                base_dir
                / task
                / model_name
                / "sense"
                / f"{lang}_speech_VC"
                / "responses"
            )

            concept_name = f"{task}_{lang}"
            if not cached_dir.exists():
                print(f"[WARN] responses dir does not exist, skipping: {cached_dir}")
                continue

            print(f"[INFO] Reading {concept_name} from {cached_dir}")

            try:
                responses, labels, response_names = read_responses_from_cached(
                    cached_dir=cached_dir,
                    concept=concept_name,
                    verbose=verbose,
                    span_group=False,  # IMPORTANT: only this LANG dir, no siblings
                )
            except RuntimeError as e:
                print(f"[WARN] Could not read responses for {concept_name}: {e}")
                continue

            if not responses:
                print(f"[WARN] No responses found after filtering for {concept_name}")
                continue

            for resp_name, arr in responses.items():
                # arr is (C, N): C = time/feature length, N = kept samples
                C, N = arr.shape
                stats = compute_stats_for_array(arr)

                rec = {
                    "task": task,
                    "lang": lang,
                    "response_name": resp_name,
                    "C": int(C),
                    "N": int(N),
                }
                rec.update(stats)
                records.append(rec)

    if not records:
        print("[WARN] No data collected; not writing CSV or plots.")
        return

    df = pd.DataFrame.from_records(records)
    df.sort_values(by=["task", "lang", "response_name"], inplace=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved stats to {out_csv}")

    # Create plots (mean/var per task/module)
    make_plots(df, tasks=tasks, plots_dir=plots_dir)


if __name__ == "__main__":
    main()
