#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Original file from:
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conditional text generation with the auto-regressive models of the HuggingFace Transformers repository.
"""

import typing as t
import argparse
import logging
import pathlib
import pickle

import pandas as pd
import torch
import warnings
from tqdm import tqdm
#from transformers import AutoModelWithLMHead, GPT2Tokenizer
from transformers import AutoModelWithLMHead, GPT2Tokenizer, LlamaTokenizer, AutoTokenizer, XGLMTokenizer

from selfcond.generation import force_units_hooks, generate_sentence, set_seed
from selfcond.models import PytorchTransformersModel
try:
    from transformers import SeamlessM4TProcessor, SeamlessM4TForTextToText
except Exception:
    SeamlessM4TProcessor = SeamlessM4TForTextToText = None
#library version choice for seamlessM4Tprocessor is tough
import json
import soundfile as sf
import numpy as np
import torchaudio

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")



def _load_wav_any(path: str, target_sr: int) -> np.ndarray:
    """Try torchaudio first; fallback to soundfile. Returns float32 mono at target_sr."""
    try:
        wav, sr = torchaudio.load(path)
        if wav.dim() > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0).float().cpu().numpy()
    except Exception:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            # simple linear resample via torch
            t = torch.tensor(data).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            data = t.squeeze(0).cpu().numpy()
        return data.astype("float32")


def argument_parser(prev_args: t.Optional[str] = None):
    parser = argparse.ArgumentParser(prev_args)
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name from the HuggingFace Transformers.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="Models cached directory.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Context given to the model to start generation.",
        default="EOS",
    )
    parser.add_argument(
        "--prompt_format_id_for_translation",
        type=int,
        help="prompt_format_id_for_translation",
        default=1,
    )
    parser.add_argument(
        "--length", type=int, default=20, help="Number of new tokens to be generated."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Logits softmax temperature."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k tokens taken into account at generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help=(
            "Only those tokens whose probabilities add up to 0.9 are "
            "taken into account for generation (nucleus sampling)."
        ),
    )
    parser.add_argument("--device", type=str, required=False, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=[1],
        nargs="*",
        help=(
            "Random seed for initialization. If 2 seeds are passed, all seeds in between are swept."
        ),
    )
    parser.add_argument("--expertise", type=pathlib.Path, help="Expertise results as CSV file.")
    parser.add_argument(
        "--metric",
        type=str,
        default="ap",
        help="Metric to use to rank experts for generation.",
    )
    parser.add_argument("--forcing", type=str, nargs="*", default=["on_p50"], help="Forcing value.")
    parser.add_argument(
        "--num-units",
        type=int,
        default=[1],
        nargs="+",
        help=(
            "Number of units (top experts in terms of --metric) to be intervened on during"
            " generation"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help=(
            "Which set of top units to use. If set to 1, units from [0, --num-units] are used. "
            "If set to 2, units from [--num-units, 2*--num-units] are used. And so on. "
            "If set to 0, --num-units random units are selected."
        ),
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="If set, force --num-units per layer at a time.",
    )

    parser.add_argument("--eos", action="store_true", help="Trim the sentence if EOS is generated.")
    parser.add_argument("--verbose", action="store_true", help="Show more information")
    parser.add_argument("--no-save", action="store_true", help="If set, nothing is saved.")
    parser.add_argument(
        "--only-last-token",
        action="store_true",
        help="If set, only the last token of the sequence is intervened upon.",
    )
    parser.add_argument(
        "--results-file",
        type=pathlib.Path,
        default=None,
        help=(
            "If set, the results file will have this name, otherwise a generic naming is applied.."
        ),
    )
    parser.add_argument("--seamless", action="store_true",
                    help="Use SeamlessM4T seq2seq path (processor + generate).")
    parser.add_argument("--src-lang", type=str, default=None,
                    help="Source language code for Seamless, e.g., eng, jpn, deu.")
    parser.add_argument("--tgt-lang", type=str, default=None,
                    help="Target language code for Seamless, e.g., deu, jpn, cmn.")
    parser.add_argument("--task", type=str, default="t2t_translation",
                    help="Task type: ASR | s2t_translation | t2t_translation")
    #4 above are added to adapt to seamless
    
    return parser.parse_args()


def generate(args):
    assert len(args.seed) in [1, 2]

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    #n_gpu = torch.cuda.device_count() if device is not "cpu" else 0
    n_gpu = torch.cuda.device_count() if device != "cpu" else 0
    print(f"Device {device} ({n_gpu})")

    expertise = pd.read_csv(args.expertise)
    concept = expertise["concept"].values[0]

    # Load model and tokenizer
    #tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    #tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    
    # --- Load tokenizer/processor + model wrapper ---
    if args.seamless:
        assert SeamlessM4TProcessor is not None and SeamlessM4TForTextToText is not None, \
            "Transformers build lacks SeamlessM4T.*; please upgrade transformers."
        assert args.src_lang and args.tgt_lang, \
            "--src-lang and --tgt-lang are required for Seamless."

        # Explicit processor (not AutoProcessor)
        processor = SeamlessM4TProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

        # Wrap HF model in your readable wrapper so unit hooks still work
        readable_model = PytorchTransformersModel(
            model_name=args.model_name_or_path,
            seq_len=128,
            cache_dir=args.cache_dir,
            device=device,
        )
        hf_model = readable_model.module  # expected to be SeamlessM4TForTextToText
        tokenizer = None  # not used in Seamless text2text path
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        readable_model = PytorchTransformersModel(
            model_name=args.model_name_or_path,
            seq_len=128,
            cache_dir=args.cache_dir,
            device=device,
        )
        hf_model = readable_model.module
        processor = None


    layer_names = (
        list(expertise.sort_values("layer").layer.unique())
        if args.per_layer
        else [
            None,
        ]
    )
    forcing_values = args.forcing
    generation_results = []

    sweep_seed = range(args.seed[0], args.seed[1]) if len(args.seed) == 2 else args.seed
    for forcing_value in forcing_values:
        for top_n in args.top_n:
            for force_layer in layer_names:
                for num_units in args.num_units:
                    pbar = tqdm(
                        total=len(sweep_seed),
                        desc=(
                            "Generating"
                            f" [force={forcing_value} units={num_units}/{len(expertise)} ({100 * num_units / len(expertise):0.3f}%)"
                            f" top_n={top_n} layers={force_layer}]"
                        ),
                    )

                    # Set units to forcing value
                    mean_metric = 0
                    if num_units > 0:
                        model, df_force = force_units_hooks(
                            model=readable_model,
                            expertise=expertise,
                            value=forcing_value,
                            metric=args.metric,
                            num_units=num_units,
                            top_n=top_n,
                            use_layers=force_layer,
                            only_last_token=args.only_last_token,
                        )
                        mean_metric = float(df_force[args.metric].mean())
                    # Added 2023.10.09
                    else: 
                        model=readable_model
                        mean_metric = 0.0

                    for seed in sweep_seed:
                        # Sample sentence from a nn.Module (that might have been forced)
                        if args.verbose:
                            print("\n")
                            print(f"{concept} s={seed} f={num_units}:")

                        # Random seed for full reproducibility
                        set_seed(seed, gpu=device != 'cpu')
                        #set_seed(None, gpu=device != "cpu")

                        # For machine translation task
                        if "translation_text_" in args.prompt:
                            with open(args.prompt, 'rb') as f:
                                source_text = pickle.load(f)[seed-1]['en']
                            
                        
                        if args.seamless:
                            # --- Seamless text-to-text path using SeamlessM4TProcessor ---
                            # Encode with src_lang; generate with tgt_lang.
                            if args.task.upper() in ("ASR", "S2T_TRANSLATION"):
                                # args.prompt は concept JSON パスを想定
                                with open(args.prompt, "r", encoding="utf-8") as f:
                                    obj = json.load(f)
                                clips = obj.get("clips", [])
                                if len(clips) == 0:
                                    raise ValueError(f"No clips in JSON: {args.prompt}")
                                idx = (seed - 1) % len(clips)
                                clip = clips[idx]

                                wav_path = clip.get("path_VC") or clip.get("path")
                                if not wav_path:
                                    raise ValueError("Neither 'path_VC' nor 'path' found in clip.")
                                # sampling_rate: JSONの各clipにあればそれを、なければ Expertise 側や引数で決める
                                sr = int(clip.get("sampling_rate", 16000))

                                audio_1d = _load_wav_any(wav_path, sr)

                                # processor でエンコード（ここでは processor に tgt_lang も渡す＝対称に）
                                tgt = args.tgt_lang or args.src_lang
                                feats = processor(
                                    audios=[audio_1d],
                                    sampling_rate=sr,
                                    src_lang=args.src_lang,
                                    tgt_lang=tgt,
                                    return_tensors="pt",
                                )
                                feats = {k: v.to(device) for k, v in feats.items()}

                                do_sample = args.temperature > 0.0
                                gen_kwargs = dict(max_new_tokens=int(args.length), do_sample=do_sample)
                                if do_sample:
                                    if args.top_k: gen_kwargs["top_k"] = int(args.top_k)
                                    if args.top_p: gen_kwargs["top_p"] = float(args.top_p)
                                    gen_kwargs["temperature"] = float(args.temperature)

                                #gen_kwargs = _apply_task_defaults(args, gen_kwargs)
                                with torch.no_grad():
                                    # tgt_langは processor に渡しているので generate には渡さない
                                    #generated = hf_model.generate(**feats, tgt_lang=tgt, **gen_kwargs)
                                    generated = hf_model.generate(
                                        **feats,
                                        tgt_lang=tgt,
                                        generate_speech=False,   # <- TEXT ONLY
                                        **gen_kwargs
                                    )

                                sentence = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
                                #sentence = processor.batch_decode(generated, skip_special_tokens=True)[0]
                                perplexity = None
                                #perplexity not implemented. think about whether its necessary

                            elif args.task.upper() == "T2T_TRANSLATION":
                                # 2 系統をサポート:
                                # (A) 旧来の PKL から 'en' を読む
                                # (B) concept JSON から 'transcription' を読む
                                if args.prompt.endswith(".pkl"):
                                    with open(args.prompt, "rb") as f:
                                        source_text = pickle.load(f)[seed-1]["en"]
                                else:
                                    with open(args.prompt, "r", encoding="utf-8") as f:
                                        obj = json.load(f)
                                    clips = obj.get("clips", [])
                                    if len(clips) == 0:
                                        raise ValueError(f"No clips in JSON: {args.prompt}")
                                    idx = (seed - 1) % len(clips)
                                    source_text = clips[idx].get("transcription", "")
                                    if not source_text:
                                        raise ValueError("No 'transcription' found for t2t_translation.")

                                feats = processor(
                                    text=[source_text],
                                    src_lang=args.src_lang,
                                    tgt_lang=args.tgt_lang,   # 対称に processor 側で設定
                                    return_tensors="pt",
                                )
                                feats = {k: v.to(device) for k, v in feats.items()}

                                do_sample = args.temperature > 0.0
                                gen_kwargs = dict(max_new_tokens=int(args.length), do_sample=do_sample)
                                tgt = args.tgt_lang or args.src_lang
                                if do_sample:
                                    if args.top_k: gen_kwargs["top_k"] = int(args.top_k)
                                    if args.top_p: gen_kwargs["top_p"] = float(args.top_p)
                                    gen_kwargs["temperature"] = float(args.temperature)
                                #gen_kwargs = _apply_task_defaults(args, gen_kwargs)
                                with torch.no_grad():
                                    generated = hf_model.generate(
                                        **feats,
                                        tgt_lang=tgt,            # safe & recommended
                                        generate_speech=False,   # text-only output
                                        **gen_kwargs
                                    )
                                    #generated = hf_model.generate(**feats, tgt_lang=tgt, **gen_kwargs)
                                    #generated = hf_model.generate(**feats, **gen_kwargs)

                                sentence = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
                                #sentence = processor.batch_decode(generated, skip_special_tokens=True)[0]
                                perplexity = None

                            else:
                                raise ValueError(f"Unsupported task for Seamless: {args.task}")

                            
                        else:
                            # --- Original decoder-only path ---
                            if "translation_text_" in args.prompt:
                                # map prompt_format_id to instruction (unchanged)
                                if args.prompt_format_id_for_translation == 0:
                                    prompt = f'Translate a sentence from English to a target language.\nEnglish: {source_text}\nTarget Language:'
                                elif args.prompt_format_id_for_translation == 1:
                                    prompt = f'Translate English to a target language.\nEnglish: {source_text}\nTarget Language:'
                                elif args.prompt_format_id_for_translation == 2:
                                    prompt = f'Translate an English sentence into a target language.\nEnglish: {source_text}\nTarget Language:'
                                elif args.prompt_format_id_for_translation == 3:
                                    prompt = f'Translate an English sentence into German.\nEnglish: {source_text}\nGerman:'
                                elif args.prompt_format_id_for_translation == 4:
                                    prompt = f'Translate an English sentence into Japanese.\nEnglish: {source_text}\nJapanese:'
                                elif args.prompt_format_id_for_translation == 5:
                                    prompt = f'Translate an English sentence into French.\nEnglish: {source_text}\nFrench:'
                                elif args.prompt_format_id_for_translation == 6:
                                    prompt = f'Translate an English sentence into Spanish.\nEnglish: {source_text}\nSpanish:'
                                elif args.prompt_format_id_for_translation == 7:
                                    prompt = f'Translate an English sentence into Chinese.\nEnglish: {source_text}\nChinese:'
                                else:
                                    raise ValueError("error! prompt_format_id_for_translation is not properly defined!")
                            else:
                                prompt = source_text

                        if not args.seamless:
                            sentence, perplexity = generate_sentence(
                                model=readable_model.module,
                                tokenizer=tokenizer,
                                prompt=prompt, #args.prompt,
                                length=args.length,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                temperature=args.temperature,
                                eos=args.eos,
                                device=device,
                                verbose=args.verbose,
                            )
                        # Store generation results
                        generation_results.append(
                            [
                                forcing_value,
                                num_units,
                                top_n,
                                seed,
                                sentence,
                                mean_metric,
                                force_layer,
                                perplexity,
                            ]
                        )
                        pbar.update()

                    # Restore units to the original values!
                    if num_units > 0:
                        readable_model.restore_units()

                    pbar.close()

    if args.results_file is None:
        results_file: pathlib.Path = (
            args.expertise.parent / f'forced_sentences_{concept}_{args.prompt.replace("_", "")}.csv'
        )
    else:
        results_file: pathlib.Path = args.results_file

    generated_df = pd.DataFrame(
        columns=[
            "forcing_value",
            "num_units",
            "top_n",
            "seed",
            "sentence",
            "mean_metric",
            "forced_layer",
            "perplexity",
        ],
        data=generation_results,
    )
    generated_df["context"] = [args.prompt] * len(generated_df)
    generated_df["concept"] = [concept] * len(generated_df)

    #if results_file.exists():
    #    previous_df = pd.read_csv(results_file, index_col=0)
    #    generated_df = previous_df.append(generated_df, ignore_index=True)

    if not args.no_save:
        generated_df.to_csv(results_file)
    else:
        print(generated_df)
        for units, units_df in generated_df.groupby(by="num_units", sort=False):
            for i, sentence in zip(range(len(generated_df)), units_df["sentence"].values):
                print(f"{i} [{units}] {sentence}")


if __name__ == "__main__":
    args = argument_parser()
    generate(args)
