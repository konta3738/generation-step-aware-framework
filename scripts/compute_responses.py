#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import pathlib
from pathlib import Path

from selfcond.data import (
    concept_list_to_df,
    PytorchTransformersTokenizer,
    ConceptDataset,
    AudioCfg,
)
from selfcond.responses import cache_responses
from selfcond.models import collect_responses_info, PytorchTransformersModel
from transformers import SeamlessM4TProcessor

logging.basicConfig(level=logging.WARNING)


def compute_and_save_responses(
    model_name: str,
    model_cache_dir: pathlib.Path,
    data_path: pathlib.Path,
    concept_group: str,
    concept: str,
    tokenizer: PytorchTransformersTokenizer,
    batch_size: int,
    response_save_path: pathlib.Path,
    num_per_concept: int,
    seq_len: int,
    device: str,
    task: str,
    *,
    processor=None,                 # NEW: thread through
    src_lang: str | None = None,    # NEW
    tgt_lang: str | None = None,    # NEW
    sampling_rate: int = 16000,     # NEW
    verbose: bool = False,
) -> None:
    model_short_name = model_name.rstrip("/").split("/")[-1]
    local_data_file = data_path / concept_group / f"{concept}.json"
    if not local_data_file.exists():
        print(f"Skipping {local_data_file}, file not found.")
        return

    out_dir = response_save_path / model_short_name / concept_group / concept / "responses"
    if out_dir.exists():
        print(out_dir)
        print(f"Skipping, already computed responses {local_data_file}")
        return

    random_seed = 1234

    LANG2ID = {"en":0, "ja":1, "zh":2, "es":3, "fr":4, "de":5}

    if task == "t2t_translation":
        dataset = ConceptDataset(
            json_file=local_data_file,
            tokenizer=tokenizer,
            seq_len=seq_len,
            num_per_concept=num_per_concept,
            random_seed=random_seed,
            modality="text",
            label_mode="multiclass",
            lang2id=LANG2ID,
            # NEW:
            task=task,
            src_lang_code3=src_lang,
            tgt_lang_code3=tgt_lang,
        )
    elif task in ("ASR", "s2t_translation"):
        dataset = ConceptDataset(
            json_file=local_data_file,
            tokenizer=tokenizer,          # unused in speech mode; kept for signature
            seq_len=seq_len,
            num_per_concept=num_per_concept,
            random_seed=random_seed,
            modality="speech",
            use_path_vc=True,
            speech_T=300,
            audio_cfg=AudioCfg(target_sr=sampling_rate, n_mels=160),
            label_mode="multiclass",
            lang2id=LANG2ID,
            # NEW:
            task=task,
            src_lang_code3=src_lang,
            tgt_lang_code3=tgt_lang,
        )
    else:
        raise ValueError("task must be one of: t2t_translation, ASR, s2t_translation")

    # --- determine output directory (for en2X translation) ---
    if src_lang == "eng" and task in ("s2t_translation", "t2t_translation") and tgt_lang:
        # Special case: English source → target
        subdir_name = f"eng2{tgt_lang}{dataset.concept}"
        save_path = response_save_path / model_short_name / dataset.concept_group / subdir_name
    else:
        # Default (original)
        save_path = response_save_path / model_short_name / dataset.concept_group / dataset.concept

    # Skip check as before
    #if (save_path / "responses").exists():
    #print(f"Skipping {dataset.concept_group}/{save_path.name}")
    #return
    
    #added for token6,11 because the original one wil just skip
    resp_root = save_path / "responses"
    if (resp_root / "responses_tok1").exists() and (resp_root / "responses_tok6").exists() and (resp_root / "responses_tok11").exists():
        print("Skipping ...")
        return

    if verbose:
        print(dataset, flush=True)

    save_path.mkdir(parents=True, exist_ok=True)

    # Load model
    tm_model = PytorchTransformersModel(
        model_name, seq_len=dataset._seq_len, cache_dir=model_cache_dir, device=device
    )

    # Layer selection
    responses_info_interm = collect_responses_info(model_name=model_name, model=tm_model)

    # If ASR and no tgt_lang given, default tgt_lang to src_lang
    eff_tgt = src_lang if (task == "ASR" and tgt_lang is None) else tgt_lang

    # Cache responses (processor enables language- & modality-aware inputs)
    cache_responses(
        model=tm_model,
        dataset=dataset,
        batch_size=batch_size,
        response_infos=responses_info_interm,
        save_path=save_path / "responses",
        task=task,
        processor=processor,
        src_lang=src_lang,
        tgt_lang=eff_tgt,
        sampling_rate=sampling_rate,
        decoder_steps=[2, 7, 12],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compute_responses.py")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--model-cache", type=pathlib.Path, default=None)
    parser.add_argument("--tok-cache", type=pathlib.Path, default=None)
    parser.add_argument("--data-path", type=pathlib.Path, required=True)
    parser.add_argument("--concepts", type=str, default="")
    parser.add_argument("--responses-path", type=pathlib.Path, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--num-per-concept", type=int, default=10000)
    parser.add_argument("--inf-batch-size", type=int, default=1, choices=[1])
    parser.add_argument("--device", type=str)
    parser.add_argument("--task", type=str,
                        choices=["t2t_translation", "ASR", "s2t_translation"],
                        default="t2t_translation")
    parser.add_argument("--src-lang", type=str, default=None)
    parser.add_argument("--tgt-lang", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    args = parser.parse_args()

    model_cache = args.model_cache if args.model_cache else None
    tok_cache = args.tok_cache if args.tok_cache else args.model_cache

    data_path = args.data_path
    responses_path = args.responses_path
    responses_path.mkdir(exist_ok=True, parents=True)

    if not args.concepts:
        assert (data_path / "concept_list.csv").exists()
        concepts_requested = data_path / "concept_list.csv"
    else:
        concepts_requested = args.concepts.split(",")

    concept_df = concept_list_to_df(concepts_requested)

    # Tokenizer (kept for non-Seamless models / text preprocessing)
    tokenizer = PytorchTransformersTokenizer(args.model_name_or_path, tok_cache)

    # Build processor only for SeamlessM4T models
    processor = None
    if "seamless" in args.model_name_or_path.lower() and "m4t" in args.model_name_or_path.lower():
        processor = SeamlessM4TProcessor.from_pretrained(args.model_name_or_path, cache_dir=tok_cache)

    print("TEST")
    print(concept_df)
    for _, row in concept_df.iterrows():
        concept, concept_group = row["concept"], row["group"]
        if concept in ["positive", "negative"] and concept_group == "keyword":
            continue

        print(f"Running inference to read responses on concept {concept_group}/{concept}")
        compute_and_save_responses(
            model_name=args.model_name_or_path,
            model_cache_dir=model_cache,
            data_path=data_path,
            concept_group=concept_group,
            concept=concept,
            seq_len=args.seq_len,
            num_per_concept=args.num_per_concept,
            batch_size=args.inf_batch_size,
            response_save_path=responses_path,
            tokenizer=tokenizer,
            verbose=True,
            device=args.device,
            task=args.task,
            processor=processor,                 # pass through
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            sampling_rate=args.sampling_rate,
        )
