#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pathlib

import numpy as np
import pandas as pd

from selfcond.data import concept_list_to_df
from selfcond.expertise import ExpertiseResult
#from selfcond.responses import read_responses_from_cached
from selfcond.responses import read_responses_from_cached, read_responses_for_modality, read_responses_for_mmlang

from selfcond.models import get_layer_regex
from selfcond.visualization import (
    plot_scatter_pandas,
    plot_metric_per_layer,
    plot_in_dark_mode,
)

import re

# importing from compute_responses.py is better (might change later)
LANG2ID = {"en": 0, "ja": 1, "zh": 2, "es": 3, "fr": 4, "de": 5}
MODALITY_RE = re.compile(r"^(s2t|t2t|en2x_s2t|en2x_t2t)_modality$")
LANG_MODALITY_RE = re.compile(
    r"^(s2t|t2t|en2x_s2t|en2x_t2t)_([a-z]{2,3})_lang_modality$"
)
#MMLANG_RE = re.compile(r"^(de|es|ja|zh|fr)_mmlang_neuron$")
MMLANG_RE = re.compile(r"^(?:en2x_)?(de|es|ja|zh|fr)_mmlang_neuron$")

def _list_variants_from_one_task(
    *,
    speech_root: pathlib.Path,      # .../Speech
    task: str,                      # e.g. s2t_translation or en2x_s2t_translation
    model_short_name: str,
    concept_group: str,             # usually "sense"
) -> list[str]:
    """
    Return sorted unique response variants (directories) like:
      responses_tok0, responses_tok1, ...
    by scanning ONE representative task directory:
      speech_root/task/model_short_name/concept_group/*_speech_VC/responses/<variant>/
    """
    task_group_dir = speech_root / task / model_short_name / concept_group
    if not task_group_dir.exists():
        return []

    variants: set[str] = set()

    for lang_dir in sorted(task_group_dir.glob("*_speech_VC")):
        resp_root = lang_dir / "responses"
        if not resp_root.exists():
            continue
        for vdir in resp_root.iterdir():
            if vdir.is_dir() and vdir.name.startswith("responses_tok"):
                variants.add(vdir.name)

    return sorted(variants)


def _infer_probe_task_for_special_concepts(concept: str) -> str:
    """
    Decide which ONE task to probe variants from, based only on en2x vs non-en2x.
    """
    # "en2x" 判定を緩く：concept 名に en2x が入ってたら en2x 側、とする
    is_en2x = ("en2x" in concept)
    return "en2x_s2t_translation" if is_en2x else "s2t_translation"

def _infer_target_lang_id_from_concept(concept: str):
    """
    Try to infer the target language ID from the concept string.
    Matches patterns like 'en_speech_VC', 'de_speech_VC_new', etc.
    Returns an integer ID or None if not found.
    """
    # Look for language at start or after a slash, followed by _ or end
    m = re.search(r'(?:(?<=^)|(?<=/))(en|ja|zh|es|fr|de)(?=(_|$))', concept)
    if not m:
        return None
    lang = m.group(1)
    return LANG2ID.get(lang, None)

def _infer_exp_task_slug(concept: str) -> str | None:
    """
    Return the task_slug to use for saving expertise results.
    - normal concepts -> None (save under concept_dir)
    - modality/mmlang concepts -> dedicated task_slug (optionally prefixed with en2x_)
    """
    is_modality = bool(MODALITY_RE.match(concept) or LANG_MODALITY_RE.match(concept))
    is_mmlang   = bool(MMLANG_RE.match(concept))

    if not (is_modality or is_mmlang):
        return None

    # en2x 判定：mmlang は "en2x_de_mmlang_neuron" みたいに prefix が付く
    # modality は "en2x_s2t_modality" / "en2x_t2t_modality" / "en2x_s2t_de_lang_modality" 等
    is_en2x = concept.startswith("en2x_") or ("en2x_s2t" in concept) or ("en2x_t2t" in concept)

    if is_mmlang:
        base = "mmlang_neuron"
    else:
        base = "lang_modality_neuron" if "lang_modality" in concept else "modality_neuron"

    return f"en2x_{base}" if is_en2x else base

def analyze_expertise_for_concept(
    concept_dir: pathlib.Path,
    concept_group: str,
    concept: str,
    *,
    root_dir: pathlib.Path,
    model_short_name: str,
    responses_variant: str,
):
    """
    Analyze expertise for a concept.

    - For standard concepts (e.g. 'en_speech_VC'):
        use read_responses_from_cached and LANG2ID binarization.
    - For modality concepts:
        '{prefix}_modality' or '{prefix}_{lang}_lang_modality',
        prefix ∈ {s2t, t2t, en2x_s2t, en2x_t2t}:
        use read_responses_for_modality, and restrict to text_decoder.* keys.
    """
    #cached_responses_dir = concept_dir / "responses" / responses_variant
    #concept_exp_dir = concept_dir / "expertise" / responses_variant

    # Is this a modality concept?
    is_modality = bool(MODALITY_RE.match(concept) or LANG_MODALITY_RE.match(concept))
    is_mmlang = bool(MMLANG_RE.match(concept))

    exp_task_slug = _infer_exp_task_slug(concept)

    # Read cache path is always based on the given concept_dir
    cached_responses_dir = concept_dir / "responses" / responses_variant

    # Save path: default = concept_dir/... but override task_slug only for modality/mmlang
    if exp_task_slug is None:
        concept_exp_dir = concept_dir / "expertise" / responses_variant
    else:
        # root_dir is ".../${base_path}${datapath}/${task_slug}"
        exp_root_dir = root_dir.with_name(exp_task_slug)  # replace last path component
        exp_concept_dir = exp_root_dir / model_short_name / concept_group / concept
        concept_exp_dir = exp_concept_dir / "expertise" / responses_variant

    if ExpertiseResult.exists_in_disk(concept_exp_dir):
        print("Results found, skipping building")
        return

    if is_modality:
        # root_dir is a task dir (e.g. .../Speech/s2t_translation);
        # we want the Speech root to see both tasks in the pair.
        speech_root = root_dir.parent  # .../Speech
        print(f"[modality] reading from paired tasks under {speech_root}")

        try:
            responses, labels_for_expertise, response_names = read_responses_for_modality(
                speech_root=speech_root,
                model_name=model_short_name,
                concept_group=concept_group,
                concept=concept,
                responses_variant=responses_variant,
                verbose=True,
            )
        except RuntimeError as e:
            print(f"No modality responses found for concept {concept}: {e}")
            return

        # Restrict to text_decoder only for modality concepts
        responses = {k: v for k, v in responses.items() if "text_decoder." in k}
        if not responses:
            print(f"[modality] No text_decoder.* keys found for concept {concept}; skipping.")
            return

    elif is_mmlang:
        # NEW: mmlang_neuron feature
        #
        # root_dir is .../${base_path}${datapath}/mmlang_neuron
        # so speech_root is .../${base_path}${datapath} (which already includes /Speech),
        # and pkl files live under:
        #   ${base_path}${datapath}/s2t_translation/seamless-m4t-v2-large/sense/{LANG}_speech_VC/responses/
        #   ${base_path}${datapath}/t2t_translation/seamless-m4t-v2-large/sense/{LANG}_speech_VC/responses/
        m = MMLANG_RE.match(concept)
        target_lang = m.group(1)  # de, es, ja, zh, fr
        speech_root = root_dir.parent  # .../${base_path}${datapath}

        print(
            f"[mmlang_neuron] concept={concept} target_lang={target_lang} "
            f"reading from s2t/t2t under {speech_root}"
        )

        try:
            responses, labels_for_expertise, response_names = read_responses_for_mmlang(
                speech_root=speech_root,
                model_name=model_short_name,
                concept_group=concept_group,
                concept=concept,
                target_lang=target_lang,
                responses_variant=responses_variant,
                verbose=True,
            )
        except RuntimeError as e:
            print(f"[mmlang_neuron] No responses found for concept {concept}: {e}")
            return

        if not responses:
            print(f"[mmlang_neuron] Found no usable keys for concept {concept}; skipping.")
            return

    else:
        # Original behavior: read from cached_responses_dir only
        try:
            responses, labels_int, response_names = read_responses_from_cached(
                cached_responses_dir, concept
            )
        except RuntimeError:
            print(f"No responses found for concept {concept}")
            return

        assert (
            labels_int is not None
        ), "Cannot compute expertise, did not find any labels in cached responses."

        if not responses:
            print(f"Found response files but could not load them for concept {concept}")
            return

        # convert multiclass label to binary based on LANG2ID
        target_id = _infer_target_lang_id_from_concept(concept)
        if target_id is not None:
            labels_int = np.asarray(labels_int)
            labels_bin = (labels_int == target_id).astype(int)
            pos, neg = int(labels_bin.sum()), int((labels_bin == 0).sum())
            print(f"[binary labels] concept='{concept}' target_id={target_id} -> pos={pos}, neg={neg}")
            labels_for_expertise = labels_bin
        else:
            print(f"[binary labels] concept='{concept}': could not infer target language; keeping original labels")
            labels_for_expertise = labels_int

    concept_exp_dir.mkdir(exist_ok=True, parents=True)

    expertise_result = ExpertiseResult()
    print("expertise_result.build")
    expertise_result.build(
        responses=responses,
        labels=labels_for_expertise,
        concept=concept,
        concept_group=concept_group,
        forcing=True,
    )
    print("expertise_result.save")
    expertise_result.save(concept_exp_dir)


def build_result_figures(
    expertise_result: ExpertiseResult,
    results_dir: str,
    layer_types_regex=None,
    show_figures: bool = False,
):
    """
        Build expertise figures for a specific concept. It expects a `results_dir` with the following tree:

    ```
    results_dir
        concept_group
            concept
                expertise
                    expertise.csv (will be loaded to build results)
                    expertise_info.json (will be loaded to build results)
    ```

    Args:
        expertise_result: ExpertiseResult object with duly loaded results
        results_dir: Where to save the output assets
        show_figures: Show figures or just save?
    """
    print("Building plots")
    df = expertise_result.export_as_pandas()
    print(f"DataFrame sizeis {len(df)}")
    info_json = expertise_result.export_extra_info_json()

    concept = df["concept"].iloc[0]
    concept_group = df["group"].iloc[0]

    # Print top AP
    print(df.sort_values(by="ap", ascending=False).iloc[:10])

    # Show correlation of corr and on_value
    plot_scatter_pandas(
        df,
        "ap",
        "on_p50",
        out_dir=results_dir,
        y_lim=[0, 30],
        alpha=0.1,
        title=f"AP vs. ON Value (concept {concept_group}/{concept})",
        also_show=show_figures,
    )

    neurons_at_ap_df = pd.DataFrame(
        index=list(info_json["neurons_at_ap"].keys()),
        data={"neuron count": list(info_json["neurons_at_ap"].values())},
    )
    neurons_at_ap_df.index.name = "ap"
    print(f'\nmaxAP = {np.max(df["ap"])}')

    for k in [10, 100]:
        plot_metric_per_layer(
            df,
            metric="ap",
            out_dir=results_dir,
            top_k=k,
            layer_types_regex=layer_types_regex,
            also_show=show_figures,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_expertise.py",
        description=(
            "This script computes expertise results given a set of model "
            "responses collected using `compute_responses.py`.\nThe expertise "
            "results are saved as a DataFrame with `units` rows and various "
            "informative columns, and a json file with extra information."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        help=(
            "Root directory with responses. Should contain responses_"
            "`dir/model/concept_group/concept/responses`"
        ),
        required=True,
    )
    parser.add_argument("--model-name", type=str, help="The model name", required=True)
    parser.add_argument("--concepts", type=str, help="concepts to analyze")
    parser.add_argument("--k", type=int, help="Top K neurons to plot", default=10)
    parser.add_argument("--show", action="store_true", help="Show images or just save")
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Force skip for concepts with existing expertise results.",
    )
    parser.add_argument("--black", action="store_true", help="Figures in black mode")
    args = parser.parse_args()

    plot_in_dark_mode(args.black)

    root_dir = args.root_dir

    # Load concepts from file or list
    if not args.concepts:
        assert (root_dir / "concept_list.csv").exists()
        concepts_requested = root_dir / "concept_list.csv"
    else:
        concepts_requested = args.concepts.split(",")
        #if "," in args.concepts:
        #    concepts_requested = args.concepts.split(",")
        #else:
        #    concepts_requested = pathlib.Path(args.concepts)

    print(concepts_requested)
    concept_df = concept_list_to_df(concepts_requested)

    model_short_name = args.model_name.rstrip("/").split("/")[-1]


    for row_index, row in concept_df.iterrows():
        concept_dir = root_dir / model_short_name / row["group"] / row["concept"]
        responses_root = concept_dir / "responses"
        print("debug_responses_root:", responses_root)
        concept_name = row["concept"]
        is_modality = bool(MODALITY_RE.match(concept_name) or LANG_MODALITY_RE.match(concept_name))
        is_mmlang   = bool(MMLANG_RE.match(concept_name))

        if is_modality or is_mmlang:
            # modality/mmlang は concept_dir/responses ではなく、Speech/<task>/... から拾う
            speech_root = root_dir.parent  # .../Speech
            probe_task = _infer_probe_task_for_special_concepts(concept_name)

            variants = _list_variants_from_one_task(
                speech_root=speech_root,
                task=probe_task,
                model_short_name=model_short_name,
                concept_group=row["group"],
            )

            if not variants:
                # 代表タスクが空だったら、念のためもう片方もフォールバックで見る（任意）
                fallback_task = "en2x_s2t_translation" if probe_task == "s2t_translation" else "s2t_translation"
                variants = _list_variants_from_one_task(
                    speech_root=speech_root,
                    task=fallback_task,
                    model_short_name=model_short_name,
                    concept_group=row["group"],
                )

            if not variants:
                print(f"[skip] No response variants found under Speech/{probe_task} (special concept={concept_name})")
                continue
        else:
            # 従来どおり concept_dir/responses から拾う
            if responses_root.exists():
                variants = sorted([
                    p.name for p in responses_root.iterdir()
                    if p.is_dir() and p.name.startswith("responses_tok")
                ])
            else:
                variants = []

            if not variants:
                print(f"[skip] No response variants found under {responses_root}")
                continue

        for variant in variants:
            print(f"analyze_expertise_for_concept variant={variant}")
            analyze_expertise_for_concept(
                concept_dir=concept_dir,
                concept=row["concept"],
                concept_group=row["group"],
                root_dir=root_dir,
                model_short_name=model_short_name,
                responses_variant=variant,
            )

            exp_task_slug = _infer_exp_task_slug(concept_name)
            
            if exp_task_slug is None:
                expertise_dir = concept_dir / "expertise" / variant
            else:
                exp_root_dir = root_dir.with_name(exp_task_slug)
                expertise_dir = (
                    exp_root_dir
                    / model_short_name
                    / row["group"]
                    / concept_name
                    / "expertise"
                    / variant
                )

            
            if not ExpertiseResult.exists_in_disk(expertise_dir):
                print(f"[skip] No expertise results in {expertise_dir}")
                continue

            expertise_result = ExpertiseResult()
            expertise_result.load(expertise_dir)
            layer_types_regex = get_layer_regex(model_name=args.model_name)
            build_result_figures(
                expertise_result=expertise_result,
                results_dir=expertise_dir,
                layer_types_regex=layer_types_regex,
                show_figures=args.show,
            )
