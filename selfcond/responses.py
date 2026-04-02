
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import pathlib
import pickle
import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from selfcond.models import (
    TorchModel,
    ResponseInfo,
    processors_per_model,
    MODEL_INPUT_FIELDS,
    LABELS_FIELD,
)

from typing import List, Dict, Any
#import torch
import torchaudio

import re
import collections

import itertools

#to read pkl files from other tasks without copying unnecessarily
# ------------------------------
# Helper: collect *.pkl files from current concept and siblings
# ------------------------------
def _collect_response_pkls(cached_dir: pathlib.Path, *, span_group: bool = True) -> list[pathlib.Path]:
    """
    Return a sorted, de-duplicated list of *.pkl response files that includes:
      - cached_dir/*.pkl
      - (optionally) all sibling concept responses under cached_dir/../../*/responses/*.pkl
    """
    files = []
    print("debug_collect_response_pkls:",cached_dir)
    # 1) current concept's responses
    if cached_dir.exists():
        files.extend(cached_dir.glob("*.pkl"))

    if span_group:
        # 2) siblings under the same concept_group
        # cached_dir = .../<concept>/responses/responses_tok{k}
        tok_dir_name = cached_dir.name
        responses_dir = cached_dir.parent
        concept_dir = responses_dir.parent
        group_dir = concept_dir.parent  # .../<concept_group>
        if group_dir.exists():
            # Look for */responses/*.pkl one level below group_dir
            for resp_dir in group_dir.glob(f"*/responses/{tok_dir_name}"):
                # If it's the same as cached_dir, it will be de-duplicated later
                files.extend(resp_dir.glob("*.pkl"))

    # De-duplicate by resolved path and sort for stability
    files = sorted({p.resolve() for p in files})
    return files

# ------------------------------
# Save utility
# ------------------------------
def save_batch(batch: t.Dict[str, np.ndarray], batch_index: int, save_path: pathlib.Path) -> None:
    with (save_path / f"{batch_index:05d}.pkl").open("wb") as fp:
        pickle.dump(batch, fp)

# labels is a list but .detach is causing a problem
# --- NEW: a safe collate that turns lists/arrays into tensors and leaves strings alone

# --- NEW: safe conversion of labels to numpy
def _labels_to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    # last resort: single int/float
    return np.asarray([x])

# ------------------------------
# Small utility: normalize lang codes (turn '' or None -> None)
# ------------------------------
def _norm_lang(x: t.Optional[str]) -> t.Optional[str]:
    if x is None: 
        return None
    x = str(x).strip()
    return x or None  # '' -> None

# ------------------------------
# Collate: be permissive (keep strings/paths as object arrays or lists)
# ------------------------------
def _concatenate_data(examples: t.List[t.Dict[str, t.Any]]) -> t.Dict[str, t.Any]:
    out: t.Dict[str, t.Any] = {}
    keys = examples[0].keys()
    for k in keys:
        vals = [ex[k] for ex in examples]
        first = vals[0]
        # strings
        if isinstance(first, str):
            out[k] = np.array(vals, dtype=object)
        # list[str]
        elif isinstance(first, list) and (len(first) == 0 or isinstance(first[0], str)):
            out[k] = np.array(vals, dtype=object)
        # numeric arrays/lists -> tensor
        elif isinstance(first, (np.ndarray, list)) and (
            (isinstance(first, np.ndarray) and np.issubdtype(first.dtype, np.number))
            or (isinstance(first, list) and first and isinstance(first[0], (int, float)))
        ):
            out[k] = torch.tensor(vals)
        else:
            # anything else (dicts, mixed) -> keep list
            out[k] = vals
    return out


import pathlib
import torch
import torchaudio

def _to_pathlike(x):
    # Convert numpy.str_, PathLike, etc. to plain str/pathlib.Path
    if isinstance(x, (str, pathlib.Path)):
        return pathlib.Path(x)
    try:
        # numpy scalars -> str
        return pathlib.Path(str(x))
    except Exception:
        return None

def _load_waveform_any(ex: Dict[str, Any], target_sr: int) -> torch.Tensor:
    """
    Load mono waveform at target_sr from many possible example shapes:
      - ex["audio"] as:
          * torch.Tensor [T] or [C,T]
          * dict with {"array": np.ndarray, "sampling_rate": int} (HF datasets)
          * string/path to a file
      - or any of these keys as a path: "audio_path","wav_path","file","file_path",
        "filepath","filename","path_audio","source","src","wav","wave","path_vc","path"
    """
    # 0) direct tensor in ex["audio"]
    if "audio" in ex and isinstance(ex["audio"], torch.Tensor):
        wav = ex["audio"]
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav  # assume given in the right rate; resample only when we know the original

    # 1) HF-style dict in ex["audio"]
    if "audio" in ex and isinstance(ex["audio"], dict):
        a = ex["audio"]
        if "array" in a:
            import numpy as np
            arr = a["array"]
            wav = torch.from_numpy(arr.astype(np.float32))
            if wav.ndim == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # [1,T]
            orig_sr = int(a.get("sampling_rate", target_sr))
            if orig_sr != target_sr:
                wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
            return wav

    # 2) a bunch of common path-like keys
    candidate_path_keys = [
        "audio", "audio_path", "path_audio", "wav_path",
        "file", "file_path", "filepath", "filename",
        "source", "src", "wav", "wave",
        "path_vc", "path",
    ]
    for k in candidate_path_keys:
        if k in ex and ex[k] is not None and not isinstance(ex[k], dict):
            p = _to_pathlike(ex[k])
            if p is not None and p.exists():
                wav, sr = torchaudio.load(str(p))
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)
                return wav

    # 3) last resort: clear error with context
    available = {k: type(v).__name__ for k, v in ex.items()}
    raise ValueError(
        f"No audio found in example. Available keys/types: {available}. "
        f"Expected a tensor/dict under 'audio' or a path under one of {candidate_path_keys}."
    )



# replace your _as_examples_list with this
def _py_scalar(x):
    # unwrap numpy scalars / 0-d arrays
    try:
        import numpy as _np
        if isinstance(x, (_np.generic,)):
            return x.item()
        if isinstance(x, _np.ndarray) and x.ndim == 0:
            return x.item()
    except Exception:
        pass
    return x

def _as_examples_list(b):
    if isinstance(b, list):
        return b
    if isinstance(b, dict):
        first_key = next(iter(b))
        n = len(b[first_key])
        out = []
        for i in range(n):
            ex = {}
            for k, v in b.items():
                if isinstance(v, (list, tuple)):
                    ex[k] = _py_scalar(v[i])
                elif hasattr(v, "__getitem__") and not isinstance(v, (str, bytes)):
                    ex[k] = _py_scalar(v[i])
                else:
                    ex[k] = v
            out.append(ex)
        return out
    raise TypeError(f"Unexpected batch type: {type(b)}")

def _build_inputs_with_processor(
    batch: List[Dict[str, Any]],
    task: str,
    processor,
    sampling_rate: int,
    src_lang: str | None,
    tgt_lang: str | None,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:

    if task == "t2t_translation":
        # 1) If the dataset is already tokenized, just use it.
        if isinstance(batch, dict) and "input_ids" in batch:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch.get("attention_mask", None)
                                   .to(device) if batch.get("attention_mask") is not None else None,
            }
            # remove Nones so model.forward(**inputs) doesn’t choke
            return {k: v for k, v in inputs.items() if v is not None}

        # 2) Otherwise, build from raw texts.
        examples = _as_examples_list(batch)
        texts = [
            (ex.get("text")
             or ex.get("src_text")
             or ex.get("sentence")
             or ex.get("input")  # a couple of common fallbacks
             or "")
            for ex in examples
        ]
        if not any(texts):
            raise ValueError("No texts found in batch for t2t_translation.")
        feats = processor(
            text=texts,
            src_lang=_norm_lang(src_lang),
            tgt_lang=_norm_lang(tgt_lang),
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in feats.items()}

    # --- existing audio branches unchanged ---
    examples = _as_examples_list(batch)
    if task in ("ASR", "s2t_translation"):
        wavs = [_load_waveform_any(ex, sampling_rate) for ex in examples]
        wavs_1d = [w.squeeze(0) for w in wavs]
        feats = processor(
            audios=wavs_1d,
            sampling_rate=sampling_rate,
            src_lang=_norm_lang(src_lang) if src_lang else None,
            tgt_lang=_norm_lang(tgt_lang) if (task == "s2t_translation") else None,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in feats.items()}

    raise ValueError(f"Unsupported task: {task}")

#generate() to get token 6 and 11 as well
'''
def _get_decoder_prefixes_from_generate(
    model: TorchModel,
    inputs: Dict[str, torch.Tensor],
    steps: list[int],
    tgt_lang: str | None = None,
) -> Dict[int, torch.Tensor]:
    """
    Return decoder_input_ids prefixes for each step in `steps`.
    step=1 -> [BOS] only (no need to generate, but we unify interface)
    step>1 -> generate enough tokens, then take prefix [:, :step]
    """
    assert steps and min(steps) >= 1
    module = model.module  # SeamlessM4Tv2Model
    device = next(module.parameters()).device

    steps_sorted = sorted(set(steps))
    max_step = max(steps_sorted)

    # --- ensure we have decoder start token only case ---
    # For encoder-decoder, generate() returns sequences starting with decoder_start_token_id.
    # We need at least max_step tokens in the returned sequences.
    with torch.no_grad():
        # Important: max_new_tokens counts *generated tokens after start token*.
        # So to get sequence length = max_step, need max_new_tokens = max_step - 1.
        gen_out = module.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            tgt_lang = tgt_lang if tgt_lang else None,
            max_new_tokens=max_step - 1,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
        )

    seq = gen_out.sequences  # (B, L) typically L == max_step
    if seq.size(1) < max_step:
        # very defensive: if generate stopped early for some reason, pad by re-generating longer
        with torch.no_grad():
            gen_out = module.generate(
                **{k: v.to(device) for k, v in inputs.items()},
                max_new_tokens=(max_step - 1) + (max_step - seq.size(1)),
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
            )
        seq = gen_out.sequences

    prefixes = {s: seq[:, :s].contiguous() for s in steps_sorted}
    return prefixes
'''
def _extract_sequences(gen_out):
    # transformersのGenerateOutput系
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    # dictで返るケース
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    # tuple/listで返るケース（先頭がsequencesであることが多い）
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    # そもそもTensorで返るケース
    if torch.is_tensor(gen_out):
        return gen_out
    raise TypeError(f"Unknown generate() output type: {type(gen_out)}")


def _get_decoder_prefixes_from_generate(
    model: TorchModel,
    inputs: Dict[str, torch.Tensor],
    steps: list[int],
    tgt_lang: str | None = None,
) -> Dict[int, torch.Tensor]:
    assert steps and min(steps) >= 1
    module = model.module
    device = next(module.parameters()).device

    steps_sorted = sorted(set(steps))
    max_step = max(steps_sorted)

    gen_kwargs = dict(
        **{k: v.to(device) for k, v in inputs.items()},
        max_new_tokens=max_step - 1,
        num_beams=1,
        do_sample=False,
        return_dict_in_generate=True,
        generate_speech=False,
    )

    # Seamlessは tgt_lang を generate に渡せる環境もある（ダメなら落として再試行）
    if tgt_lang is not None:
        gen_kwargs["tgt_lang"] = tgt_lang

    with torch.no_grad():
        try:
            gen_out = module.generate(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("tgt_lang", None)
            gen_out = module.generate(**gen_kwargs)

    seq = _extract_sequences(gen_out)

    if seq.size(1) < max_step:
        with torch.no_grad():
            extra = (max_step - 1) + (max_step - seq.size(1))
            gen_kwargs["max_new_tokens"] = extra
            try:
                gen_out = module.generate(**gen_kwargs)
            except TypeError:
                gen_kwargs.pop("tgt_lang", None)
                gen_out = module.generate(**gen_kwargs)
        seq = _extract_sequences(gen_out)

    return {s: seq[:, :s].contiguous() for s in steps_sorted}


def cache_responses(
    model: TorchModel,
    dataset: Dataset,
    response_infos: t.List[ResponseInfo],
    batch_size: int,
    save_path: pathlib.Path,
    *,
    task: str = "t2t_translation",
    processor=None,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    sampling_rate: int = 16000,
    decoder_steps: list[int] | None = None, #token1,6,11
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    process_fn_list = processors_per_model(model)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_concatenate_data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if decoder_steps is None:
        decoder_steps = [2, 7, 12] #<\s>と__eng__をスキップ
    decoder_steps = sorted(set(decoder_steps))

    #make 3 different save directories for token 1,6,11
    save_path.mkdir(parents=True, exist_ok=True)
    step_to_dir = {}
    for s in decoder_steps:
        d = save_path / f"responses_tok{s}"
        d.mkdir(parents=True, exist_ok=True)
        step_to_dir[s] = d

    for i, batch in tqdm(enumerate(data_loader), desc="Caching inference"):
        # --- one-time debug on first batch
        if i == 0:
            try:
                examples_dbg = _as_examples_list(batch)
                print("[DEBUG] First example keys/types:",
                      {k: type(v).__name__ for k, v in examples_dbg[0].items()})
                # Helpful: show any path-like keys and whether files actually exist
                path_keys = ["audio","audio_path","path_audio","wav_path","file","file_path","filepath","filename",
                             "source","src","wav","wave","path_vc","path"]
                has = {k: examples_dbg[0].get(k) for k in path_keys if k in examples_dbg[0]}
                print("[DEBUG] Candidate audio-path fields (first example):", has)
                for k, v in has.items():
                    try:
                        p = _to_pathlike(v)
                        print(f"[DEBUG] {k} -> {p} exists={p.exists() if p else None}")
                    except Exception as e:
                        print(f"[DEBUG] {k} -> {v} (exists? error: {e})")
            except Exception as e:
                print("[DEBUG] Unable to inspect first example:", e)

        # Build inputs
        try:
            inputs = _build_inputs_with_processor(
                batch=batch,
                task=task,
                processor=processor,
                sampling_rate=sampling_rate,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                device=dev,
            )
        except Exception as e:
            # Extra context on first failure
            if i == 0:
                print("[ERROR] While building inputs for first batch:", repr(e))
            raise
        
        #token1,6,11のために追加されたやつだがBOS BOSにならないかとかチェックする必要あり。
        # --- Seamless: build decoder_input_ids prefixes for each step ---
        # step=1 は BOS だけでOKなので、generate から取って統一しても良いし、
        # run_inferenceの自動BOS生成に任せても良い。ここでは統一して prefix を渡す。
        prefixes = _get_decoder_prefixes_from_generate(model, inputs, decoder_steps, tgt_lang)

        '''
        #あとで消すdebug
        print("[DEBUG] input keys:", inputs.keys())
        for k,v in inputs.items():
            if torch.is_tensor(v):
                print(" ", k, tuple(v.shape), v.dtype)
        seq = prefixes[max(decoder_steps)]  # 例えば step=11 の prefix
        pad_id = getattr(model.module.config, "pad_token_id", 0)
        print("[DEBUG] prefix len:", seq.size(1))
        print("[DEBUG] pad ratio:", (seq == pad_id).float().mean().item())

        cfg = model.module.config
        print("[DEBUG] dec_start:", getattr(cfg, "decoder_start_token_id", None),
            "eos:", getattr(cfg, "eos_token_id", None),
            "pad:", getattr(cfg, "pad_token_id", None))
        #ここまでdebugであとで消す
        '''
        for s in decoder_steps:
            step_inputs = dict(inputs)
            step_inputs["decoder_input_ids"] = prefixes[s]
            step_inputs["decoder_attention_mask"] = (prefixes[s] != 0).long()

            response_batch = model.run_inference(
                inputs=step_inputs,
                outputs={ri.name for ri in response_infos}
            )

            for fn in process_fn_list:
                response_batch = fn(response_batch)

            if LABELS_FIELD in batch:
                response_batch[LABELS_FIELD] = _labels_to_numpy(batch[LABELS_FIELD])

            save_batch(response_batch, i, step_to_dir[s])

        '''
        # Run model
        response_batch = model.run_inference(inputs=inputs, outputs={ri.name for ri in response_infos})

        # Post-process
        for fn in process_fn_list:
            response_batch = fn(response_batch)

        # Labels (optional)
        if LABELS_FIELD in batch:
            response_batch[LABELS_FIELD] = _labels_to_numpy(batch[LABELS_FIELD])

        save_batch(response_batch, i, save_path)
        '''
# ------------------------------
# Reader
# ------------------------------
#when the T is too long, it is recorded in the wrong way bc of our implementation (transpose part), thus exclude those ones with long T before concatanation.
def read_responses_from_cached(
    cached_dir: pathlib.Path, concept: str, verbose: bool = False, *, span_group: bool = True
) -> t.Tuple[t.Dict[str, np.ndarray], t.Optional[np.ndarray], t.Set[str]]:
    """
    Read cached response PKLs and return:
      - data: dict[key] -> array shaped (C, N) where C is time/feature length, N is kept samples
      - labels: (N,) or None
      - response_names: set of kept keys
    Policy:
      • Skip known variable-length keys by substring.
      • Compute majority length per key. For keys with majority share >= 0.90, keep only PKL files
        whose length matches the majority; files violating any such key are fully excluded (labels too).
      • Keys with weak majority (< 0.90) are dropped entirely.
    """
    data: t.Dict[str, np.ndarray] = {}
    labels: t.List[float] = []
    response_names: t.Set[str] = set()
    labels_name = LABELS_FIELD

    # Collect candidate files
    all_files = _collect_response_pkls(cached_dir, span_group=span_group)
    if not all_files:
        raise RuntimeError("No responses found")

    # Keys to skip (robust substrings)
    skip_substrings = {"self_attn.distance_embedding", "feature_projection"}

    # Helper: orient any 1D / column vector to (1, T); return (arr_1xT, T) or (None, None)
    def _to_row_1xT(a: np.ndarray) -> t.Tuple[t.Optional[np.ndarray], t.Optional[int]]:
        if a.ndim == 1:
            a = a[None, :]
            return a, a.shape[1]
        if a.ndim == 2:
            if a.shape[0] == 1:
                return a, a.shape[1]
            if a.shape[1] == 1:
                a = a.T
                return a, a.shape[1]
        return None, None

    # ---------- PASS 1: scan lengths per key, per file ----------
    import collections
    per_file_key_len: t.List[t.Dict[str, int]] = []
    per_file_payloads: t.List[t.Dict[str, np.ndarray]] = []
    per_file_labels: t.List[t.Optional[float]] = []
    len_stats: t.Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    candidate_keys: t.Optional[t.Set[str]] = None

    for file_name in tqdm(all_files, total=len(all_files), desc=f"Scanning {concept}"):
        with file_name.open("rb") as fp:
            rb = pickle.load(fp)

        # decide keys on first file (excluding labels + skip patterns)
        if candidate_keys is None:
            candidate_keys = {
                k for k in rb.keys()
                if k != labels_name and not any(s in k for s in skip_substrings)
            }
            if verbose:
                print(f"[init] candidate_keys={len(candidate_keys)}")

        # collect per-file arrays and lengths for candidate keys
        payloads: t.Dict[str, np.ndarray] = {}
        lengths: t.Dict[str, int] = {}
        for k in candidate_keys:
            if k not in rb:
                continue
            arr = np.asarray(rb[k])
            row, T = _to_row_1xT(arr)
            if row is None or T is None:
                continue
            payloads[k] = row
            lengths[k] = T
            len_stats[k][T] += 1

        per_file_payloads.append(payloads)
        # label (may be missing)
        per_file_labels.append(
            (np.asarray(rb[labels_name]).ravel()[0] if labels_name in rb else None)
        )
        per_file_key_len.append(lengths)

    if candidate_keys is None:
        raise RuntimeError("No usable keys found in cached responses.")

    # Majority length per key and share
    maj_len: t.Dict[str, int] = {}
    maj_share: t.Dict[str, float] = {}
    for k, ctr in len_stats.items():
        if not ctr:
            continue
        t_star, n_star = ctr.most_common(1)[0]
        total = sum(ctr.values())
        maj_len[k] = t_star
        maj_share[k] = (n_star / total) if total else 0.0

    # Keys to actually use: require strong majority (>= 0.90)
    strong_keys = {k for k in candidate_keys if maj_share.get(k, 0.0) >= 0.90}
    weak_keys = set(candidate_keys) - strong_keys
    if verbose:
        print(f"[majority] strong_keys={len(strong_keys)} weak_keys(dropped)={len(weak_keys)}")

    # ---------- Build keep mask: drop entire file if any strong key mismatches ----------
    keep_mask: t.List[bool] = []
    for i, lengths in enumerate(per_file_key_len):
        ok = True
        for k in strong_keys:
            mT = maj_len[k]
            # require file to have this key with matching majority length
            if k not in lengths or lengths[k] != mT:
                ok = False
                break
        keep_mask.append(ok)

    kept_indices = [i for i, ok in enumerate(keep_mask) if ok]
    if verbose:
        print(f"[filter] kept_files={len(kept_indices)}/{len(all_files)}")

    if not kept_indices:
        raise RuntimeError("After majority filtering, no files remain. Loosen the threshold or inspect cache.")

    # ---------- PASS 2: materialize matrices and aligned labels ----------
    # Assemble data per key from kept files only
    for k in strong_keys:
        target_T = maj_len[k]
        rows: t.List[np.ndarray] = []
        for i in kept_indices:
            row = per_file_payloads[i].get(k)
            if row is None or row.shape[1] != target_T:
                # Should not happen due to keep_mask; guard anyway
                continue
            rows.append(row)
        if not rows:
            continue
        stacked = np.concatenate(rows, axis=0).T  # shape (target_T, N_kept)
        data[k] = stacked
        if verbose:
            print(f"[data] {k}: {stacked.shape}")

    # Build labels aligned to kept files (drop files we excluded)
    kept_labels = []
    for i in kept_indices:
        lb = per_file_labels[i]
        if lb is not None:
            kept_labels.append(lb)
    labels_arr = np.asarray(kept_labels) if kept_labels else None

    response_names = set(data.keys())

    # Final sanity: ensure all N match labels length (when labels exist)
    if labels_arr is not None and data:
        Ns = {arr.shape[1] for arr in data.values()}
        assert len(Ns) == 1, f"N mismatch across keys: {Ns}"
        N = Ns.pop()
        assert N == labels_arr.shape[0], f"labels N={labels_arr.shape[0]} vs responses N={N}"

    return data, labels_arr, response_names

def read_responses_for_modality(
    speech_root: pathlib.Path,
    model_name: str,
    concept_group: str,
    concept: str,
    responses_variant: str | None = None,
    verbose: bool = False,
) -> t.Tuple[t.Dict[str, np.ndarray], np.ndarray, t.Set[str]]:
    """
    Read *.pkl from a pair of tasks (s2t vs t2t, or en2x_s2t vs en2x_t2t)
    and build binary labels according to the 'modality' concept.

    Supported concepts:
      - '{prefix}_modality'
      - '{prefix}_{lang}_lang_modality'

    where prefix ∈ {s2t, t2t, en2x_s2t, en2x_t2t} and lang is a 2–3 letter code
    (e.g. de, es, cmn, spa, fra, deu, jpn).

    IMPORTANT: majority-length filtering is applied *per task*,
    then the kept examples from each task are concatenated.
    """
    # allow 2–3 letter lang codes
    modality_re = re.compile(r"^(s2t|t2t|en2x_s2t|en2x_t2t)_modality$")
    lang_modality_re = re.compile(
        r"^(s2t|t2t|en2x_s2t|en2x_t2t)_([a-z]{2,3})_lang_modality$"
    )

    m_mod  = modality_re.match(concept)
    m_lang = lang_modality_re.match(concept)

    if m_mod:
        prefix = m_mod.group(1)
        lang_filter = None
    elif m_lang:
        prefix = m_lang.group(1)
        lang_filter = m_lang.group(2)   # e.g. 'de', 'cmn', 'spa'
    else:
        raise ValueError(f"Unsupported modality concept name: {concept}")

    prefix_to_tasks = {
        "s2t":       ["s2t_translation", "t2t_translation"],
        "t2t":       ["s2t_translation", "t2t_translation"],
        "en2x_s2t":  ["en2x_s2t_translation", "en2x_t2t_translation"],
        "en2x_t2t":  ["en2x_s2t_translation", "en2x_t2t_translation"],
    }
    tasks = prefix_to_tasks[prefix]

    prefix_to_positive_task = {
        "s2t":       "s2t_translation",
        "t2t":       "t2t_translation",
        "en2x_s2t":  "en2x_s2t_translation",
        "en2x_t2t":  "en2x_t2t_translation",
    }
    positive_task = prefix_to_positive_task[prefix]

    # ---------------------------
    # Collect files per task
    # ---------------------------
    # task -> list[(pkl_path, label)]
    task_to_files: dict[str, list[tuple[pathlib.Path, int]]] = {}

    for task in tasks:
        task_group_dir = speech_root / task / model_name / concept_group
        if not task_group_dir.exists():
            if verbose:
                print(f"[modality] missing task dir: {task_group_dir}")
            continue

        per_task_files: list[tuple[pathlib.Path, int]] = []
        for lang_dir in sorted(task_group_dir.glob("*_speech_VC")):
            base = lang_dir.name.split("_")[0]   # e.g. 'de_speech_VC' → 'de', 'eng2spaen_speech_VC' → 'eng2spaen'

            # --- NEW: parse en2x dirnames into target 3-letter code ---
            if task.startswith("en2x_"):
                # expect something like 'eng2spaen', 'eng2cmnen', 'eng2fraen', ...
                m = re.match(r"eng2([a-z]+)en$", base)
                if m:
                    lang_name = m.group(1)      # 'spa', 'cmn', 'fra', 'deu', 'jpn'
                else:
                    # fallback if pattern does not match
                    lang_name = base
            else:
                # s2t/t2t: dirs are like 'de_speech_VC', so base is already 'de'
                lang_name = base

            resp_dir = lang_dir / "responses" / responses_variant
            if not resp_dir.exists():
                continue

            for pkl_path in sorted(resp_dir.glob("*.pkl")):
                if lang_filter is None:
                    # prefix_modality: only task decides label
                    label = 1 if task == positive_task else 0
                else:
                    # prefix_lang_lang_modality: task AND lang must match
                    label = 1 if (task == positive_task and lang_name == lang_filter) else 0
                per_task_files.append((pkl_path, int(label)))

        if per_task_files:
            task_to_files[task] = per_task_files

    if not task_to_files:
        raise RuntimeError(f"[modality] No PKL files found under {speech_root} for concept={concept}")

    # ---------------------------
    # Helper: majority-length filter for ONE task
    # ---------------------------
    def _filter_one_task(
        file_infos: list[tuple[pathlib.Path, int]],
        task_name: str,
    ) -> tuple[dict[str, list[np.ndarray]], list[int], set[str]]:
        """
        Returns:
          rows_by_key: key -> list[row[1,T]]
          labels_task: list[int] for kept examples
          keys_task:   set of keys that survived
        """
        labels_name = LABELS_FIELD
        skip_substrings = {"self_attn.distance_embedding", "feature_projection"}

        per_file_key_len: list[dict[str, int]] = []
        per_file_payloads: list[dict[str, np.ndarray]] = []
        len_stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        candidate_keys: t.Optional[set[str]] = None
        labels_raw: list[int] = []

        def _to_row_1xT(a: np.ndarray) -> tuple[t.Optional[np.ndarray], t.Optional[int]]:
            if a.ndim == 1:
                a = a[None, :]
                return a, a.shape[1]
            if a.ndim == 2:
                if a.shape[0] == 1:
                    return a, a.shape[1]
                if a.shape[1] == 1:
                    a = a.T
                    return a, a.shape[1]
            return None, None

        for pkl_path, lab in tqdm(
            file_infos, total=len(file_infos), desc=f"Scanning {concept} ({task_name})"
        ):
            with pkl_path.open("rb") as fp:
                rb = pickle.load(fp)

            if candidate_keys is None:
                candidate_keys = {
                    k for k in rb.keys()
                    if k != labels_name and not any(s in k for s in skip_substrings)
                }
                if verbose:
                    print(f"[modality init {task_name}] candidate_keys={len(candidate_keys)}")

            payloads: dict[str, np.ndarray] = {}
            lengths: dict[str, int] = {}
            for k in candidate_keys:
                if k not in rb:
                    continue
                arr = np.asarray(rb[k])
                row, T = _to_row_1xT(arr)
                if row is None or T is None:
                    continue
                payloads[k] = row
                lengths[k] = T
                len_stats[k][T] += 1

            per_file_payloads.append(payloads)
            per_file_key_len.append(lengths)
            labels_raw.append(int(lab))

        if candidate_keys is None:
            return {}, [], set()

        # Majority per key (within this task)
        maj_len: dict[str, int] = {}
        maj_share: dict[str, float] = {}
        for k, ctr in len_stats.items():
            if not ctr:
                continue
            t_star, n_star = ctr.most_common(1)[0]
            total = sum(ctr.values())
            maj_len[k] = t_star
            maj_share[k] = (n_star / total) if total else 0.0

        strong_keys = {k for k in candidate_keys if maj_share.get(k, 0.0) >= 0.90}
        if verbose:
            print(f"[modality majority {task_name}] strong_keys={len(strong_keys)}")

        # Drop files that don't match majority length for any strong key
        keep_mask: list[bool] = []
        for i, lengths in enumerate(per_file_key_len):
            ok = True
            for k in strong_keys:
                mT = maj_len[k]
                if k not in lengths or lengths[k] != mT:
                    ok = False
                    break
            keep_mask.append(ok)

        kept_indices = [i for i, ok in enumerate(keep_mask) if ok]
        if verbose:
            print(f"[modality filter {task_name}] kept_files={len(kept_indices)}/{len(file_infos)}")

        if not kept_indices:
            return {}, [], set()

        rows_by_key: dict[str, list[np.ndarray]] = {k: [] for k in strong_keys}
        labels_task: list[int] = []

        for idx in kept_indices:
            labels_task.append(labels_raw[idx])
            payloads = per_file_payloads[idx]
            for k in strong_keys:
                row = payloads.get(k)
                if row is None or row.shape[1] != maj_len[k]:
                    continue
                rows_by_key[k].append(row)

        keys_task = {k for k, rows in rows_by_key.items() if rows}
        rows_by_key = {k: rows for k, rows in rows_by_key.items() if rows}

        return rows_by_key, labels_task, keys_task

    # ---------------------------
    # Run per-task filtering and then merge
    # ---------------------------
    all_rows_by_key: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    all_labels: list[int] = []
    all_keys_sets: list[set[str]] = []

    for task_name, file_infos in task_to_files.items():
        rows_by_key, labels_task, keys_task = _filter_one_task(file_infos, task_name)
        if not rows_by_key:
            if verbose:
                print(f"[modality] task {task_name} had no usable files after filtering.")
            continue
        all_labels.extend(labels_task)
        all_keys_sets.append(keys_task)
        for k, rows in rows_by_key.items():
            all_rows_by_key[k].extend(rows)

    if not all_rows_by_key or not all_labels:
        raise RuntimeError(f"[modality] After per-task filtering, no data remains for concept={concept}")

    # Keep only keys that appear in every task that contributed
    if all_keys_sets:
        common_keys = set.intersection(*all_keys_sets)
    else:
        common_keys = set(all_rows_by_key.keys())

    data: dict[str, np.ndarray] = {}
    response_names: set[str] = set()

    for k in common_keys:
        rows = all_rows_by_key.get(k, [])
        if not rows:
            continue
        stacked = np.concatenate(rows, axis=0).T  # (T, N_total)
        data[k] = stacked
        response_names.add(k)
        if verbose:
            print(f"[modality data] {k}: {stacked.shape}")

    labels_arr = np.asarray(all_labels, dtype=int)

    if data:
        Ns = {arr.shape[1] for arr in data.values()}
        assert len(Ns) == 1, f"N mismatch across modality keys: {Ns}"
        N = Ns.pop()
        assert N == labels_arr.shape[0], f"labels N={labels_arr.shape[0]} vs responses N={N}"

    return data, labels_arr, response_names

def read_responses_for_mmlang(
    speech_root: pathlib.Path,
    model_name: str,
    concept_group: str,
    concept: str,
    target_lang: str,
    responses_variant: str | None = None,
    verbose: bool = False,
    languages: t.Optional[list[str]] = None,
) -> t.Tuple[t.Dict[str, np.ndarray], np.ndarray, t.Set[str]]:
    """
    Read *.pkl from:
        s2t_translation/seamless-m4t-v2-large/sense/{LANG}_speech_VC/responses/
        t2t_translation/seamless-m4t-v2-large/sense/{LANG}_speech_VC/responses/
    for LANG in {de, es, ja, zh, fr}, and build binary labels:

        label = 1  if LANG == target_lang
              = 0  otherwise

    Majority-length filtering is applied per task, then examples from s2t and t2t
    are concatenated.

    Args:
        speech_root: ${base_path}${datapath}, which already includes /Speech
        model_name:  e.g. 'seamless-m4t-v2-large'
        target_lang: 'de', 'es', 'ja', 'zh', or 'fr'
        verbose:     print debug info
        languages:   optional override for the list of langs to include
    """
    import collections

    if languages is None:
        languages = ["de", "es", "ja", "zh", "fr"]

    X2EN_TO_EN2X = {"de": "deu", "es": "spa", "fr": "fra", "ja": "jpn", "zh": "cmn"}

    is_en2x = concept.startswith("en2x_")
    tasks = ["en2x_s2t_translation", "en2x_t2t_translation"] if is_en2x else ["s2t_translation", "t2t_translation"]
    print("processing as en2x for mmlang" if is_en2x else "processing as X2en for mmlang")

    task_to_files: dict[str, list[tuple[pathlib.Path, int]]] = {}

    en2x_re = re.compile(r"^eng2([a-z]+)en$")  # 'eng2spaen' -> 'spa'

    # en2x 側で比較するターゲットコード（de->deu 等）
    target_code_en2x = X2EN_TO_EN2X.get(target_lang, target_lang)

    for task in tasks:
        per_task_files: list[tuple[pathlib.Path, int]]] = []

        sense_dir = speech_root / task / model_name / "sense"
        if not sense_dir.exists():
            if verbose:
                print(f"[mmlang_neuron] missing sense dir: {sense_dir}")
            continue

    if is_en2x:
        # en2x: sense_dir/*_speech_VC を列挙して、eng2XXXen から XXX を取る
        for lang_dir in sorted(sense_dir.glob("*_speech_VC")):
            base = lang_dir.name.split("_")[0]  # 'eng2spaen' 等
            m = en2x_re.match(base)
            if not m:
                continue
            lang_code = m.group(1)  # spa/cmn/fra/deu/jpn/...
            # languages が指定されていたらフィルタ（任意）
            if languages is not None:
                # languages は X2en では ["de","es",..] 想定なので、ここでは変換して比較
                allowed = {X2EN_TO_EN2X.get(x, x) for x in languages}
                if lang_code not in allowed:
                    continue

            resp_dir = lang_dir / "responses" / responses_variant
            if not resp_dir.exists():
                continue

            label = 1 if lang_code == target_code_en2x else 0
            for pkl_path in sorted(resp_dir.glob("*.pkl")):
                per_task_files.append((pkl_path, int(label)))

    else:
        # 既存の s2t/t2t はそのまま（languages は 2文字コード前提）
        for lang in languages:
            lang_dir = speech_root / task / model_name / "sense" / f"{lang}_speech_VC"
            resp_dir = lang_dir / "responses" / responses_variant
            if not resp_dir.exists():
                if verbose:
                    print(f"[mmlang_neuron] missing responses dir: {resp_dir}")
                continue

            for pkl_path in sorted(resp_dir.glob("*.pkl")):
                label = 1 if lang == target_lang else 0
                per_task_files.append((pkl_path, label))

    if per_task_files:
        task_to_files[task] = per_task_files
    
    if not task_to_files:
        raise RuntimeError(
            f"[mmlang_neuron] No PKL files found under {speech_root} "
            f"for target_lang={target_lang}"
        )

    # ---------------------------
    # Helper: majority-length filter for ONE task
    # ---------------------------
    def _filter_one_task(
        file_infos: list[tuple[pathlib.Path, int]],
        task_name: str,
    ) -> tuple[dict[str, list[np.ndarray]], list[int], set[str]]:
        """
        Returns:
          rows_by_key: key -> list[row[1,T]]
          labels_task: list[int] for kept examples
          keys_task:   set of keys that survived
        """
        labels_name = LABELS_FIELD
        skip_substrings = {"self_attn.distance_embedding", "feature_projection"}

        per_file_key_len: list[dict[str, int]] = []
        per_file_payloads: list[dict[str, np.ndarray]] = []
        len_stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        candidate_keys: t.Optional[set[str]] = None
        labels_raw: list[int] = []

        def _to_row_1xT(a: np.ndarray) -> tuple[t.Optional[np.ndarray], t.Optional[int]]:
            if a.ndim == 1:
                a = a[None, :]
                return a, a.shape[1]
            if a.ndim == 2:
                if a.shape[0] == 1:
                    return a, a.shape[1]
                if a.shape[1] == 1:
                    a = a.T
                    return a, a.shape[1]
            return None, None

        for pkl_path, lab in tqdm(
            file_infos, total=len(file_infos), desc=f"Scanning mmlang ({task_name})"
        ):
            with pkl_path.open("rb") as fp:
                rb = pickle.load(fp)

            if candidate_keys is None:
                candidate_keys = {
                    k for k in rb.keys()
                    if k != labels_name and not any(s in k for s in skip_substrings)
                }
                if verbose:
                    print(f"[mmlang init {task_name}] candidate_keys={len(candidate_keys)}")

            payloads: dict[str, np.ndarray] = {}
            lengths: dict[str, int] = {}
            for k in candidate_keys:
                if k not in rb:
                    continue
                arr = np.asarray(rb[k])
                row, T = _to_row_1xT(arr)
                if row is None or T is None:
                    continue
                payloads[k] = row
                lengths[k] = T
                len_stats[k][T] += 1

            per_file_payloads.append(payloads)
            per_file_key_len.append(lengths)
            labels_raw.append(int(lab))

        if candidate_keys is None:
            return {}, [], set()

        # Majority per key (within this task)
        maj_len: dict[str, int] = {}
        maj_share: dict[str, float] = {}
        for k, ctr in len_stats.items():
            if not ctr:
                continue
            t_star, n_star = ctr.most_common(1)[0]
            total = sum(ctr.values())
            maj_len[k] = t_star
            maj_share[k] = (n_star / total) if total else 0.0

        strong_keys = {k for k in candidate_keys if maj_share.get(k, 0.0) >= 0.90}
        if verbose:
            print(f"[mmlang majority {task_name}] strong_keys={len(strong_keys)}")

        # Drop files that don't match majority length for any strong key
        keep_mask: list[bool] = []
        for i, lengths in enumerate(per_file_key_len):
            ok = True
            for k in strong_keys:
                mT = maj_len[k]
                if k not in lengths or lengths[k] != mT:
                    ok = False
                    break
            keep_mask.append(ok)

        kept_indices = [i for i, ok in enumerate(keep_mask) if ok]
        if verbose:
            print(f"[mmlang filter {task_name}] kept_files={len(kept_indices)}/{len(file_infos)}")

        if not kept_indices:
            return {}, [], set()

        rows_by_key: dict[str, list[np.ndarray]] = {k: [] for k in strong_keys}
        labels_task: list[int] = []

        for idx in kept_indices:
            labels_task.append(labels_raw[idx])
            payloads = per_file_payloads[idx]
            for k in strong_keys:
                row = payloads.get(k)
                if row is None or row.shape[1] != maj_len[k]:
                    continue
                rows_by_key[k].append(row)

        keys_task = {k for k, rows in rows_by_key.items() if rows}
        rows_by_key = {k: rows for k, rows in rows_by_key.items() if rows}

        return rows_by_key, labels_task, keys_task

    # ---------------------------
    # Run per-task filtering and then merge
    # ---------------------------
    all_rows_by_key: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    all_labels: list[int] = []
    all_keys_sets: list[set[str]] = []

    for task_name, file_infos in task_to_files.items():
        rows_by_key, labels_task, keys_task = _filter_one_task(file_infos, task_name)
        if not rows_by_key:
            if verbose:
                print(f"[mmlang] task {task_name} had no usable files after filtering.")
            continue
        all_labels.extend(labels_task)
        all_keys_sets.append(keys_task)
        for k, rows in rows_by_key.items():
            all_rows_by_key[k].extend(rows)

    if not all_rows_by_key or not all_labels:
        raise RuntimeError(
            f"[mmlang] After per-task filtering, no data remains for target_lang={target_lang}"
        )

    # Keep only keys that appear in every contributing task
    if all_keys_sets:
        common_keys = set.intersection(*all_keys_sets)
    else:
        common_keys = set(all_rows_by_key.keys())

    data: dict[str, np.ndarray] = {}
    response_names: set[str] = set()

    for k in common_keys:
        rows = all_rows_by_key.get(k, [])
        if not rows:
            continue
        stacked = np.concatenate(rows, axis=0).T  # (T, N_total)
        data[k] = stacked
        response_names.add(k)
        if verbose:
            print(f"[mmlang data] {k}: {stacked.shape}")

    labels_arr = np.asarray(all_labels, dtype=int)

    if data:
        Ns = {arr.shape[1] for arr in data.values()}
        assert len(Ns) == 1, f"N mismatch across mmlang keys: {Ns}"
        N = Ns.pop()
        assert N == labels_arr.shape[0], f"labels N={labels_arr.shape[0]} vs responses N={N}"

    return data, labels_arr, response_names
