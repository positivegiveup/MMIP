"""ASR worker - run in a SEPARATE process so that all GPU memory (CTranslate2 keeps its own allocator that
torch.cuda.empty_cache() cannot release) is returned to the OS when it exits.
usage: python -m Codes.asr_worker <cache_dir> <video1> [<video2> ...]   (language: env ASR_LANG, default auto)

Backends are tried in order and the worker moves to the next one when a backend cannot run on this machine
(e.g. the ctranslate2 wheel is built for CUDA 12 -> "libcublas.so.12 is not found" on a CUDA-13-only env):
  1. faster-whisper large-v3, GPU (float16)
  2. transformers Whisper large-v3-turbo, GPU (uses torch's own CUDA runtime)
  3. faster-whisper large-v3, CPU (int8; slow but needs no CUDA libraries)
Audio is decoded with PyAV (bundled with faster-whisper), so no system ffmpeg is required."""
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from opencc import OpenCC
    _cc = OpenCC("s2twp")
    to_tw = _cc.convert
except Exception:
    to_tw = lambda x: x

SR = 16000
# errors that mean "this backend cannot run here" rather than "this video is broken"
BACKEND_ERR = re.compile(r"lib\w+\.so|cuda|cublas|cudnn|out of memory|no kernel image|driver", re.I)


def load_audio(path: str) -> np.ndarray:
    """16 kHz mono float32; empty array if the file has no audio stream."""
    from faster_whisper.audio import decode_audio
    try:
        return decode_audio(path, sampling_rate=SR)
    except (IndexError, StopIteration, ValueError):     # no audio stream
        return np.zeros(0, dtype=np.float32)


def fw_backend(device: str, lang):
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8",
                         cpu_threads=max(1, (os.cpu_count() or 4) - 2))

    def run(audio):
        segs, _ = model.transcribe(audio, vad_filter=True, language=lang, beam_size=5)
        return [dict(start=float(s.start), end=float(s.end), text=to_tw(s.text.strip())) for s in segs]
    return run, model


def hf_backend(lang):
    import torch
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                    dtype=torch.float16, device="cuda")
    gen = {"language": lang} if lang else {}

    def run(audio, max_len=8.0, max_gap=0.8):
        # word-level timestamps, regrouped into short sentences (segment-level chunks are ~30 s long,
        # which is too coarse for 10 s clips)
        out = pipe({"raw": audio, "sampling_rate": SR}, chunk_length_s=30, batch_size=8,
                   return_timestamps="word", generate_kwargs=gen)
        total = len(audio) / SR
        segs, cur = [], None
        for w in out.get("chunks", []):
            t = w["text"]
            if not t.strip():
                continue
            st, en = w["timestamp"]
            st = float(st or (cur["end"] if cur else 0.0))
            en = float(en if en is not None else min(st + 0.5, total))
            if cur and (en - cur["start"] > max_len or st - cur["end"] > max_gap):
                segs.append(cur)
                cur = None
            if cur is None:
                cur = dict(start=st, end=en, text=t.strip())
            else:
                cur["end"], cur["text"] = en, cur["text"] + t
            if cur["text"].rstrip().endswith((".", "?", "!", "。", "？", "！")):
                segs.append(cur)
                cur = None
        if cur:
            segs.append(cur)
        return [dict(start=s["start"], end=s["end"], text=to_tw(s["text"].strip())) for s in segs]
    return run, pipe


BACKENDS = [
    ("faster-whisper large-v3 (cuda)", lambda lang: fw_backend("cuda", lang)),
    ("transformers whisper-large-v3-turbo (cuda)", hf_backend),
    ("faster-whisper large-v3 (cpu, int8)", lambda lang: fw_backend("cpu", lang)),
]


def main():
    cache, lang, vids = Path(sys.argv[1]), (os.environ.get("ASR_LANG") or None), sys.argv[2:]
    b, run, handle = 0, None, None

    def next_backend(reason=None):
        nonlocal b, run, handle
        if reason is not None:
            print(f"[ASR] backend '{BACKENDS[b][0]}' unusable ({reason}) -> switching", flush=True)
            b += 1
        run, handle = None, None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        while b < len(BACKENDS):
            try:
                run, handle = BACKENDS[b][1](lang)
                print(f"[ASR] using {BACKENDS[b][0]}", flush=True)
                return True
            except Exception as e:
                print(f"[ASR] cannot load {BACKENDS[b][0]}: {type(e).__name__}: {e}", flush=True)
                b += 1
        return False

    if not next_backend():
        print("[ASR] no usable backend; nothing transcribed", flush=True)
        return
    for v in vids:
        p = Path(v)
        try:
            audio = load_audio(str(p))
        except Exception as e:
            print(f"[ASR] {p.name}: cannot decode audio ({type(e).__name__}: {e}) -> not cached", flush=True)
            continue
        while True:
            try:
                segs = run(audio) if len(audio) else []
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if BACKEND_ERR.search(msg) and b + 1 < len(BACKENDS):
                    if not next_backend(msg.splitlines()[0][:160]):
                        print("[ASR] no usable backend left", flush=True)
                        return
                    continue                       # retry the same video with the next backend
                segs = None
                print(f"[ASR] {p.name}: FAILED ({msg[:200]}) -> not cached", flush=True)
                break
        if segs is None:
            continue
        (cache / f"asr_{p.stem}.json").write_text(json.dumps(segs, ensure_ascii=False), "utf-8")
        print(f"[ASR] {p.name}: {len(segs)} segments, {sum(len(s['text']) for s in segs)} chars", flush=True)


if __name__ == "__main__":
    main()
