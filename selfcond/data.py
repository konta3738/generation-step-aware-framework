#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#


import typing as t
import json
import pathlib
from typing import List, Iterable, Dict, Union, Tuple, Optional

import numpy as np
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import os
from dataclasses import dataclass

try:
    import torchaudio
    import torch
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False
    torchaudio = None
    torch = None

#for en2X translation task (to add labels)
CODE3_TO_CODE2 = {
    "eng": "en",
    "spa": "es",
    "fra": "fr",
    "cmn": "zh",
    "cmn_Hant": "zh",   # collapse to zh to match LANG2ID
    "deu": "de",
    "jpn": "ja",
}
# ---------- Audio config (optional for legacy feature extraction) ----------
@dataclass
class AudioCfg:
    target_sr: int = 16000
    n_mels: int = 160
    win_length_ms: float = 25
    hop_length_ms: float = 10
    fmin: float = 0.0
    fmax: float = None
    log_eps: float = 1e-10

def _load_wav_mono(path: str, target_sr: int) -> "torch.Tensor":
    if not _HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio is required for speech modality but is not available.")
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0)

def _wav_to_logmel(wav: "torch.Tensor", cfg: AudioCfg) -> "torch.Tensor":
    n_fft = int(round(cfg.target_sr * cfg.win_length_ms / 1000.0))
    hop_length = int(round(cfg.target_sr * cfg.hop_length_ms / 1000.0))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        center=True,
        power=2.0,
        normalized=False,
    )(wav)
    mel = torch.clamp(mel, min=cfg.log_eps)
    logmel = torch.log(mel).transpose(0, 1).contiguous()  # [frames, n_mels]
    return logmel

def _pad_or_trim_time(x: "torch.Tensor", T: int) -> "torch.Tensor":
    t, f = x.shape
    if t == T:
        return x
    if t > T:
        return x[:T, :]
    out = torch.zeros((T, f), dtype=x.dtype)
    out[:t, :] = x
    return out

# ---------- concept list ----------
def concept_list_to_df(concepts_list_or_file: Union[pathlib.Path, Iterable[str]]) -> pd.DataFrame:
    assert isinstance(concepts_list_or_file, (pathlib.Path, list))
    if isinstance(concepts_list_or_file, pathlib.Path):
        try:
            concept_df = pd.read_csv(concepts_list_or_file)
        except Exception as exc:
            raise RuntimeError(f"Error reading concepts file. {exc}")
    else:
        try:
            c_groups, c_names = [], []
            for c in concepts_list_or_file:
                concept_group, name = c.split("/")
                c_groups.append(concept_group)
                c_names.append(name)
            concept_df = pd.DataFrame(data={"group": c_groups, "concept": c_names})
        except Exception as exc:
            raise RuntimeError(f"Error parsing concept list. {exc}")
    return concept_df

# ---------- tokenizer wrapper ----------
class PytorchTransformersTokenizer:
    def __init__(self, model_name: str, cache_dir: pathlib.Path):
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=cache_dir)
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def pad_indexed_tokens(self, indexed_tokens: List[int], min_num_tokens: int) -> List[int]:
        assert min_num_tokens and min_num_tokens > 0
        pad_token_id: int = self._tokenizer.pad_token_id
        num_effective_tokens = len(indexed_tokens)
        pad_tokens: int = max(min_num_tokens - num_effective_tokens, 0)
        return indexed_tokens + [pad_token_id] * pad_tokens

    def pre_process_sequence(self, text: str, min_num_tokens: int = None) -> Dict[str, List]:
        indexed_tokens: List[int] = self._tokenizer.encode(text)
        num_effective_tokens = len(indexed_tokens)
        if min_num_tokens is not None:
            indexed_tokens = self.pad_indexed_tokens(indexed_tokens, min_num_tokens)
        attention_mask: List[int] = [1] * num_effective_tokens + [0] * (len(indexed_tokens) - num_effective_tokens)
        return {"input_ids": indexed_tokens, "attention_mask": attention_mask}

    def preprocess_dataset(self, sentence_list: List[str], min_num_tokens: int = None) -> Dict[str, List]:
        named_data: t.Dict[str, t.List] = defaultdict(list)
        for seq in tqdm(sentence_list, desc="Preprocessing", total=len(sentence_list)):
            if isinstance(seq, str):
                named_data_seq = self.pre_process_sequence(text=seq, min_num_tokens=min_num_tokens)
                for k, v in named_data_seq.items():
                    named_data[k].append(v)
        return named_data

    @property
    def model_name(self) -> str:
        return self._model_name

# ---------- base dataset ----------
class DatasetForSeqModels(Dataset):
    def __init__(
        self,
        path: pathlib.Path,
        seq_len: int,
        tokenizer: PytorchTransformersTokenizer,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> None:
        super().__init__()
        self._data: Dict[str, List] = {}
        self._model_input_fields: List[str] = []
        self._seq_len = seq_len
        self._num_per_concept = num_per_concept
        self._tokenizer = tokenizer

        unprocessed_data, self._labels = self._load_data(
            path=path, seq_len=seq_len, random_seed=random_seed, num_per_concept=num_per_concept
        )

        preprocessed_named_data = self._tokenizer.preprocess_dataset(
            sentence_list=unprocessed_data, min_num_tokens=self.seq_len
        )

        self._model_input_fields = list(preprocessed_named_data.keys())
        self._data["data"] = unprocessed_data  # list[str]
        self._data["labels"] = self._labels
        self._data.update(preprocessed_named_data)

        self._remove_too_long_data()
        self._verify_data_integrity()

    def __str__(self) -> str:
        msg = f"Dataset fields:\n"
        msg += f'\tData {len(self._data["data"])}'
        msg += f'\tTokens {np.array(self._data["input_ids"]).shape}\n'
        v = self.data["labels"]
        msg += f"\t\t {np.sum(v)}/{len(v) - np.sum(v)} pos/neg examples.\n"
        return msg

    def _load_data(
        self,
        path: pathlib.Path,
        seq_len: int = 20,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    def _verify_data_integrity(self) -> None:
        assert isinstance(self._data["input_ids"][0], list)
        assert isinstance(self._data["input_ids"][0][0], int)

    def _remove_too_long_data(self) -> None:
        remove_idx = []
        for idx, tokens in enumerate(self._data["input_ids"]):
            if len(tokens) > self.seq_len:
                remove_idx.append(idx)
        remove_idx = sorted(remove_idx, reverse=True)
        for key in list(self._data.keys()):
            for i in remove_idx:
                del self._data[key][i]

    def get_input_fields(self) -> List[Union[str, Iterable[str]]]:
        return list(self._model_input_fields)

    @property
    def data(self) -> Dict[str, List]:
        return self._data

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def num_per_concept(self) -> Optional[int]:
        return self._num_per_concept

    def __len__(self):
        return len(self._data[list(self._data.keys())[0]])

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.data.keys()}

# ---------- concept dataset (text + speech) ----------
class ConceptDataset(Dataset):
    """
    Supports:
      Old speech JSON: {'clips': {'positive': [...], 'negative': [...]} }
      New speech JSON: {'clips': [ {..., 'language': 'ja', ...}, ... ] }
      Old text JSON:   {'sentences': {'positive': [...], 'negative': [...]} }

    Exposes:
      - speech mode: data (list[dict]), labels (list[int]), audio_path (list[str])
      - text   mode: data (list[str]),  labels (list[int]), text (list[str])

    Labeling modes:
      - binary_by_language: requires target_lang='xx' → 1 if ex.language==xx else 0
      - multiclass: requires lang2id dict mapping lang -> int (e.g., {'en':0,'ja':1,...})
      - legacy_posneg: positive=1, negative=0 (default when old-format clips dict is used)
    """
    def __init__(
        self,
        json_file: pathlib.Path,
        tokenizer: PytorchTransformersTokenizer,
        seq_len: int = 100,
        num_per_concept: int = None,
        random_seed: int = None,
        *,
        modality: str = "text",           # "text" | "speech"
        use_path_vc: bool = True,
        speech_T: int = 300,
        audio_cfg: AudioCfg = AudioCfg(),
        make_features: bool = False,
        # NEW: labeling controls
        label_mode: str = "legacy_posneg",   # "legacy_posneg" | "binary_by_language" | "multiclass"
        target_lang: str | None = None,      # required if label_mode="binary_by_language"
        lang2id: Dict[str, int] | None = None,  # required if label_mode="multiclass"
        # 3 below are for en2x translation tasks (to change labels)
        task: str | None = None,
        src_lang_code3: str | None = None,
        tgt_lang_code3: str | None = None,
    ) -> None:
        super().__init__()
        assert str(json_file).endswith(".json")
        self._modality = modality
        self._seq_len = seq_len
        self._num_per_concept = num_per_concept

        with json_file.open("r", encoding="utf-8") as fp:
            js = json.load(fp)

        # Decide if we should override labels using tgt_lang instead of clip language
        override_label_code2: str | None = None
        if (
            label_mode == "multiclass"
            and task in {"s2t_translation", "t2t_translation"}
            and (src_lang_code3 or "").lower() == "eng"
            and tgt_lang_code3
        ):
            code2 = CODE3_TO_CODE2.get(tgt_lang_code3, None)
            if code2 is not None:
                override_label_code2 = code2

        self._concept = js.get("concept")
        self._concept_group = js.get("group")

        has_sentences = "sentences" in js
        has_clips = "clips" in js

        rng = np.random.RandomState(random_seed)
        def _subsample(lst):
            if num_per_concept is None or len(lst) <= num_per_concept:
                return lst
            idx = rng.choice(len(lst), num_per_concept, replace=False)
            return [lst[i] for i in idx]

        # ---------- TEXT ----------
        if modality == "text":
            if has_sentences:
                # old text shape
                pos = list(js["sentences"].get("positive", []))
                neg = list(js["sentences"].get("negative", []))
                pos = _subsample(pos)
                neg = _subsample(neg)
                sentences = pos + neg
                labels = [1] * len(pos) + [0] * len(neg)
            elif has_clips:
                # derive from speech clips’ transcriptions
                clips = js["clips"]
                # normalize to list of dicts (old shape vs new shape)
                if isinstance(clips, dict):  # old format -> flatten and keep label names
                    pos = _subsample(list(clips.get("positive", [])))
                    neg = _subsample(list(clips.get("negative", [])))
                    sentences = [c.get("transcription", "") for c in (pos + neg)]
                    labels = [1] * len(pos) + [0] * len(neg)
                else:
                    # new format: list of items with language + transcription
                    items = _subsample(list(clips))
                    sentences = [c.get("transcription", "") for c in items]
                    # label by chosen mode
                    if label_mode == "binary_by_language":
                        if not target_lang:
                            raise ValueError("target_lang is required when label_mode='binary_by_language'.")
                        labels = [1 if (c.get("language") == target_lang) else 0 for c in items]
                    elif label_mode == "multiclass":
                        if not lang2id:
                            raise ValueError("lang2id is required when label_mode='multiclass'.")
                        if override_label_code2 is not None:
                            # Use the same target-language label for all items
                            tgt_id = int(lang2id[override_label_code2])
                            labels = [tgt_id for _ in items]
                        else:
                            labels = [int(lang2id[c.get("language")]) for c in items]
                        '''
                        elif label_mode == "multiclass":
                        if not lang2id:
                            raise ValueError("lang2id is required when label_mode='multiclass'.")
                        labels = [int(lang2id[c.get("language")]) for c in items]
                        '''
                    else:
                        # legacy_posneg isn't meaningful here; fall back to binary_by_language guard
                        raise ValueError("For new 'clips' list in text mode, specify label_mode='binary_by_language' or 'multiclass'.")
            else:
                raise RuntimeError("JSON must contain either 'sentences' or 'clips'.")

            self._data = {
                "data": sentences,
                "labels": labels,
                "text": sentences,
            }
            pre = tokenizer.preprocess_dataset(sentences, min_num_tokens=seq_len)
            self._data.update(pre)

        # ---------- SPEECH ----------
        elif modality == "speech":
            if not has_clips:
                raise RuntimeError("Speech modality requires a 'clips' JSON.")
            raw_meta: List[Dict[str, t.Any]] = []
            labels: List[int] = []

            clips = js["clips"]

            if isinstance(clips, dict):
                # ----- old format: {'positive': [...], 'negative': [...]}
                if label_mode == "legacy_posneg":
                    label_map = {"positive": 1, "negative": 0}
                elif label_mode in ("binary_by_language", "multiclass"):
                    # We can still use old format, but label by language if present
                    label_map = None
                else:
                    raise ValueError(f"Unsupported label_mode: {label_mode}")

                for label_name, clip_list in clips.items():
                    subset = _subsample(list(clip_list))
                    for c in subset:
                        wav_path = c.get("path_VC") if (use_path_vc and "path_VC" in c) else c.get("path")
                        if not wav_path:
                            continue
                        raw_meta.append({
                            "path": wav_path,
                            "transcription": c.get("transcription", "")
                        })
                        if label_mode == "legacy_posneg" and label_map is not None:
                            labels.append(int(label_map.get(label_name, 0)))
                        elif label_mode == "binary_by_language":
                            if not target_lang:
                                raise ValueError("target_lang is required when label_mode='binary_by_language'.")
                            labels.append(1 if (c.get("language") == target_lang) else 0)
                        elif label_mode == "multiclass":
                            if not lang2id:
                                raise ValueError("lang2id is required when label_mode='multiclass'.")
                            labels.append(int(lang2id[c.get("language")]))
 
            else:
                # ----- new format: flat list of clip dicts, each has 'language'
                items = _subsample(list(clips))
                for c in items:
                    wav_path = c.get("path_VC") if (use_path_vc and "path_VC" in c) else c.get("path")
                    if not wav_path:
                        continue
                    raw_meta.append({
                        "path": wav_path,
                        "transcription": c.get("transcription", "")
                    })
                    if label_mode == "binary_by_language":
                        if not target_lang:
                            raise ValueError("target_lang is required when label_mode='binary_by_language'.")
                        labels.append(1 if (c.get("language") == target_lang) else 0)
                    elif label_mode == "multiclass":
                        if not lang2id:
                            raise ValueError("lang2id is required when label_mode='multiclass'.")
                        if override_label_code2 is not None:
                            tgt_id = int(lang2id[override_label_code2])
                            labels.append(tgt_id)
                        else:
                            labels.append(int(lang2id[c.get("language")]))
                        '''
                        elif label_mode == "multiclass":
                        if not lang2id:
                            raise ValueError("lang2id is required when label_mode='multiclass'.")
                        labels.append(int(lang2id[c.get("language")]))
                        '''
                    elif label_mode == "legacy_posneg":
                        raise ValueError("legacy_posneg requires old {'positive','negative'} clips shape.")

            self._data = {
                "data": raw_meta,
                "labels": labels,
                "audio_path": [m["path"] for m in raw_meta],
            }

            if make_features:
                if not _HAS_TORCHAUDIO:
                    raise RuntimeError("torchaudio required to make_features.")
                feats = []
                for m in raw_meta:
                    wav = _load_wav_mono(m["path"], audio_cfg.target_sr)
                    logmel = _wav_to_logmel(wav, audio_cfg)
                    logmel = _pad_or_trim_time(logmel, speech_T)
                    feats.append(logmel.numpy().astype(np.float32))
                self._data["input_features"] = feats

        else:
            raise ValueError("modality must be 'text' or 'speech'.")

    # Standard Dataset API unchanged
    def __len__(self) -> int:
        return len(self._data["labels"])

    def __getitem__(self, idx: int) -> Dict[str, t.Any]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    @property
    def concept(self) -> str:
        return self._concept

    @property
    def concept_group(self) -> str:
        return self._concept_group
