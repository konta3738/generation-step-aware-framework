import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from TTS.api import TTS  # Coqui TTS
#added newly below
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
import torch.serialization as ts
ts.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig])

INPUT_JSON = "/content/assets/Speech/sense/ja_speech.json"
OUT_AUDIO_ROOT = Path("./assets/Speech/fleurs_VC")
OUT_JSON_ROOT  = Path("./assets/Speech/sense")

GLOBAL_REFERENCE_WAV = "/content/audiofile.wav"  # <-- SET THIS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def lang_from_config(config_str: str) -> Optional[str]:
    return config_str.split("_", 1)[0] if config_str else None

def pick_any_existing_clip_wav(all_clips: List[dict]) -> Optional[str]:
    for c in all_clips:
        p = c.get("path")
        if p and Path(p).exists():
            return p
    return None

def map_lang(code: str) -> str:
    """
    Map ISO-ish codes to XTTS language IDs.
    XTTS v2 accepts short codes like 'en','de','ja','es','fr','it','ru','tr','pl','pt','pt-br','zh-cn','ko','ar', etc.
    We keep it simple for your dataset: ja/en/de -> themselves; fallback to 'en'.
    """
    if not code:
        return "en"
    code = code.lower()
    # Normalize some common variants
    if code in {"pt_br", "pt-br"}:
        return "pt-br"
    if code in {"zh", "zh-cn", "zh_cn"}:
        return "zh-cn"
    # most FLEURS configs like 'ja_jp','en_us','de_de' -> 'ja','en','de'
    if "_" in code:
        code = code.split("_", 1)[0]
    return code

def get_supported_languages(tts: TTS) -> Optional[set]:
    """
    Try to read supported languages from the model; API differs by version.
    Returns a set of strings or None if not available.
    """
    try:
        lm = getattr(tts.synthesizer.tts_model, "language_manager", None)
        if lm is not None:
            names = getattr(lm, "language_names", None) or getattr(lm, "name_to_id", {}).keys()
            return set(names)
    except Exception:
        pass
    return None

def main():
    data = load_json(INPUT_JSON)

    # flatten
    all_clips: List[dict] = []
    for items in data.get("clips", {}).values():
        all_clips.extend(items)

    # language buckets
    clips_by_lang: Dict[str, List[dict]] = {}
    for c in all_clips:
        lang = c.get("language") or lang_from_config(c.get("hf", {}).get("config", ""))
        if lang:
            c["language"] = lang
            clips_by_lang.setdefault(lang, []).append(c)

    # reference voice (global)
    if GLOBAL_REFERENCE_WAV and Path(GLOBAL_REFERENCE_WAV).exists():
        speaker_wav_global = GLOBAL_REFERENCE_WAV
    else:
        speaker_wav_global = pick_any_existing_clip_wav(all_clips)
        if speaker_wav_global is None:
            raise FileNotFoundError("Set GLOBAL_REFERENCE_WAV or make sure at least one clip path exists.")
        print(f"[WARN] Using first available clip as global reference: {speaker_wav_global}")

    # load model once
    tts = TTS(MODEL_NAME).to(DEVICE)
    supported = get_supported_languages(tts)
    if supported:
        # normalize supported names to lowercase for comparisons
        supported = {s.lower() for s in supported}
        # common alternates
        if "zh" in supported and "zh-cn" not in supported:
            supported.add("zh-cn")

    # process per language (same global speaker)
    for lang, clips in clips_by_lang.items():
        print(f"\n=== Processing language: {lang} ({len(clips)} clips) ===")
        lang_out_dir = OUT_AUDIO_ROOT / lang
        ensure_dir(lang_out_dir)

        for clip in tqdm(clips, desc=f"{lang}"):
            in_path = clip.get("path")
            text = clip.get("transcription", "")

            if not in_path or not Path(in_path).exists():
                print(f"[SKIP] Missing audio file: {in_path}")
                continue
            if not text:
                print(f"[SKIP] Missing transcription for: {in_path}")
                continue

            # filename
            hf_id = clip.get("hf", {}).get("id")
            out_name = f"{hf_id}.wav" if hf_id is not None else (Path(in_path).stem + ".vc.wav")
            out_path = lang_out_dir / out_name

            # language pick
            lang_candidate = clip.get("language") or lang_from_config(clip.get("hf", {}).get("config", ""))
            lang_id = map_lang(lang_candidate)

            # if model exposes supported set, guard it
            if supported and lang_id not in supported:
                print(f"[WARN] Language '{lang_id}' not in model supported set {sorted(supported)}. Falling back to 'en'.")
                lang_id = "en"

            # synthesize
            try:
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav_global,
                    file_path=str(out_path),
                    language=lang_id
                )
            except TypeError:
                # older api without 'language'
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav_global,
                    file_path=str(out_path)
                )

            # record VC path
            clip["path_VC"] = out_path.as_posix()

        # write per-language JSON
        out_json_obj = {
            "concept": f"{lang}_speech",
            "group": data.get("group", "sense"),
            "source": data.get("source", "speech"),
            "clips": {
                "positive": [c for c in data.get("clips", {}).get("positive", []) if c.get("language") == lang],
                "negative": [c for c in data.get("clips", {}).get("negative", []) if c.get("language") == lang],
            }
        }
        out_json_path = OUT_JSON_ROOT / f"{lang}_speech_VC.json"
        save_json(out_json_obj, out_json_path)
        print(f"[OK] Wrote {out_json_path}")

if __name__ == "__main__":
    main()
