"""Shared utilities for Quiz1-4 (paths, model wrappers, judges, metrics, plotting).

Target hardware: a single RTX 3090 (24 GB).
    * Text LLM  : Qwen2.5-7B-Instruct (bf16, ~15 GB)
    * VLM       : Qwen2.5-VL-7B-Instruct (bf16, ~16 GB)
    * Embedding : BAAI/bge-m3 (fp16, ~1.2 GB)
Only ONE of the 7B models is kept on the GPU at a time -> call free_gpu() after `del`.
"""
from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "Outputs"
for _d in (DATA_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TEXT_LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLM_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBED_NAME = "BAAI/bge-m3"
SEED = 42


# --------------------------------------------------------------------------- misc helpers
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def free_gpu() -> None:
    """Release cached GPU memory (call after `del model`)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _cuda_modules():
    """All live torch.nn.Module objects that have parameters on the GPU -> [(module, GB)] (top-level only)."""
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # isinstance() on gc objects can touch deprecated attrs
    mods = []
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.nn.Module):
                ps = list(o.parameters())
                if ps and ps[0].is_cuda:
                    mods.append(o)
        except Exception:
            pass
    children = {id(c) for m in mods for c in m.modules() if c is not m}
    out = []
    for m in mods:
        if id(m) not in children:
            gb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 ** 3
            out.append((m, gb))
    return sorted(out, key=lambda x: -x[1])


def gpu_inventory(min_gb: float = 0.05) -> None:
    """Debug helper: which models are still on the GPU?"""
    gpu_report(" now")
    for m, gb in _cuda_modules():
        if gb >= min_gb:
            print(f"   {type(m).__name__:45s} {gb:5.1f} GB")


def unload_models(keep=()) -> None:
    """Drop the weights of every ChatLLM / VLM / ChineseCLIP wrapper still alive (except `keep`), clear the last
    exception (its traceback pins half-loaded models after an OOM) and free the cache.
    Called automatically before ChatLLM() / VLM() load, so a leaked model can never cause OOM."""
    import sys
    sys.last_type = sys.last_value = sys.last_traceback = None
    try:  # IPython also keeps the last traceback
        ip = get_ipython()  # noqa: F821
        ip.InteractiveTB.tb = None
        sys.last_exc = None
    except Exception:
        pass
    gc.collect()
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    keep_ids = {id(k) for k in keep}
    for o in gc.get_objects():
        try:
            if id(o) in keep_ids:
                continue
            if (isinstance(o, (ChatLLM, VLM)) or type(o).__name__ == "ChineseCLIP") and \
                    getattr(o, "model", None) is not None:
                print(f"[unload] {type(o).__name__} ({getattr(o, 'name', '')})")
                o.model = None
        except Exception:
            pass
    free_gpu()


def release(ns: dict, *names: str) -> None:
    """Close Gradio objects, drop the named globals, unload every big model wrapper, free the cache.
    Usage:  release(globals(), "ui1", "chat", "llm").  The Embedder (bge-m3, ~1 GB) is kept."""
    for n in names:
        obj = ns.pop(n, None)
        if obj is not None and hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass
        del obj
    unload_models()
    gpu_report(" after release")


def _check_free(need_gb: float, what: str) -> None:
    import torch
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if free < need_gb:
        print(f"[warn] only {free:.1f} GB free before loading {what} (needs ~{need_gb:.0f} GB). Still on GPU:")
        gpu_inventory()


def gpu_report(tag: str = "") -> None:
    """Prints PyTorch-allocated memory AND the real device-wide usage (includes CTranslate2 / other libs)."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024 ** 3
            free, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info())
            print(f"[GPU]{tag} torch-allocated {used:.1f} GB | device used {total - free:.1f} / {total:.1f} GB")
    except Exception:
        pass


_OPENCC = None


def to_tw(text: str) -> str:
    """Simplified -> Traditional (Taiwan) if opencc is installed, otherwise unchanged."""
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
            _OPENCC = OpenCC("s2twp")
        except Exception:
            _OPENCC = False
    if _OPENCC is False:
        return text
    try:
        return _OPENCC.convert(text)
    except Exception:
        return text


def extract_json(text: str):
    """Return the first valid JSON object/array found inside an LLM output (or None)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------- plotting
def set_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
    })


def save_fig(name: str, fig=None, show: bool = True, dpi: int = 200) -> Path:
    """Save to Outputs/{name}.png (names follow Q1_xxx / Q2_xxx ...)."""
    import matplotlib.pyplot as plt
    fig = fig or plt.gcf()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {path}")
    return path


# --------------------------------------------------------------------------- model wrappers
class ChatLLM:
    """Plain-text chat LLM (HF transformers) with batched greedy generation."""

    def __init__(self, name: str = TEXT_LLM_NAME):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        unload_models()
        _check_free(16, name)
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def generate_batch(self, batch_messages: List[List[dict]], max_new_tokens: int = 512,
                       temperature: float = 0.0, batch_size: int = 8) -> List[str]:
        import torch
        outs: List[str] = []
        # sort by length to reduce padding, then restore the original order
        order = sorted(range(len(batch_messages)),
                       key=lambda i: sum(len(m["content"]) for m in batch_messages[i]))
        results = {}
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            prompts = [self.tok.apply_chat_template(batch_messages[i], tokenize=False,
                                                    add_generation_prompt=True) for i in ids]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, t in zip(ids, self.tok.batch_decode(new, skip_special_tokens=True)):
                results[i] = t.strip()
        return [results[i] for i in range(len(batch_messages))]

    def generate(self, messages: List[dict], max_new_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        return self.generate_batch([messages], max_new_tokens, temperature, 1)[0]


class VLM:
    """Qwen2.5-VL wrapper. Images are passed as PIL objects in the order of the
    {"type": "image"} placeholders found in `messages`."""

    def __init__(self, name: str = VLM_NAME, min_pixels: int = 128 * 28 * 28,
                 max_pixels: int = 320 * 28 * 28):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.name = name
        unload_models()
        _check_free(17, name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            name, min_pixels=min_pixels, max_pixels=max_pixels)

    def generate(self, messages: List[dict], images: Optional[list] = None,
                 max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        import torch
        msgs = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            msgs.append({"role": m["role"], "content": c})
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt",
                                padding=True).to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, **kw)
        new = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def generate_batch(self, batch_messages, max_new_tokens: int = 512,
                       temperature: float = 0.0, batch_size: int = 1) -> List[str]:
        """Text-only calls, sequential (API-compatible with ChatLLM)."""
        return [self.generate(m, None, max_new_tokens, temperature) for m in batch_messages]


class Embedder:
    """bge-m3 dense embedder (L2-normalised, so dot product == cosine)."""

    def __init__(self, name: str = EMBED_NAME, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cuda")
        self.model.max_seq_length = max_seq_length
        self.model.half()

    def encode(self, texts: Sequence[str], batch_size: int = 32, show: bool = False) -> np.ndarray:
        e = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=show)
        return e.astype(np.float32)


# --------------------------------------------------------------------------- query rewriting
REWRITE_SYS = ("你是對話查詢改寫器。給定對話紀錄與使用者最新的問題，請把最新問題改寫成"
               "『不需要看對話紀錄也能完整理解』的獨立問題：補上被代名詞或省略的主題，保留原意，"
               "不要回答問題。若問題本來就完整，原樣輸出。只輸出改寫後的問題。")


def rewrite_messages(history: List[dict], question: str, max_turns: int = 3) -> List[dict]:
    hist = history[-2 * max_turns:]
    cite = re.compile(r"\[[^\]]*\]")
    lines = []
    for m in hist:
        who = "使用者" if m["role"] == "user" else "助理"
        lines.append(f"{who}：{cite.sub('', m['content'])[:300]}")
    h = "\n".join(lines)
    return [{"role": "system", "content": REWRITE_SYS},
            {"role": "user", "content": f"【對話紀錄】\n{h}\n\n【最新問題】{question}\n\n改寫後的獨立問題："}]


def rewrite_query(llm, history: List[dict], question: str) -> str:
    if not history:
        return question
    out = llm.generate(rewrite_messages(history, question), max_new_tokens=96)
    out = out.strip().split("\n")[0].strip().strip("「」\"'")
    return out or question


# --------------------------------------------------------------------------- metrics
_PUNCT = re.compile(r"[\s，。！？、；：「」『』（）()\[\]{}《》,.!?;:\"'\-—_/\\]")


def char_f1(pred: str, gold: str) -> float:
    """Character-level F1 (suitable for Chinese)."""
    p = Counter(_PUNCT.sub("", pred or ""))
    g = Counter(_PUNCT.sub("", gold or ""))
    if not p or not g:
        return 0.0
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def token_f1(pred: str, gold: str) -> float:
    """Word-level F1 (suitable for English)."""
    p = Counter(re.findall(r"\w+(?:\.\w+)?", (pred or "").lower()))
    g = Counter(re.findall(r"\w+(?:\.\w+)?", (gold or "").lower()))
    if not p or not g:
        return 0.0
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def ndcg_at_k(rels: Sequence[int], k: int, n_relevant: Optional[int] = None) -> float:
    rels = list(rels)[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    n_rel = n_relevant if n_relevant is not None else sum(rels)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_rel)))
    return dcg / ideal if ideal > 0 else 0.0


REFUSAL_KEYS = ["資料不足", "無法回答", "未提及", "沒有提到", "沒有提及", "找不到", "無相關",
                "不足以", "影片中找不到", "cannot answer", "not enough information", "insufficient"]


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in REFUSAL_KEYS)


# --------------------------------------------------------------------------- LLM-as-judge
JUDGE_SYS = "You are a strict and consistent grader. Output JSON only."


def _parse_score(out: str) -> float:
    obj = extract_json(out)
    if isinstance(obj, dict) and "score" in obj:
        try:
            return float(min(5, max(1, float(obj["score"]))))
        except (TypeError, ValueError):
            pass
    m = re.search(r"[1-5]", out or "")
    return float(m.group()) if m else 1.0


def judge_correctness(llm, triples: Sequence[tuple], batch_size: int = 8) -> List[float]:
    """triples = [(question, gold_answer, predicted_answer), ...] -> scores in [1, 5]."""
    prompts = []
    for q, gold, pred in triples:
        user = (f"Question: {q}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
                "Rate how well the candidate matches the reference answer in FACTUAL content "
                "(ignore style, language and length). Be strict: compare the key facts one by one; "
                "any missing or wrong key fact must lower the score.\n"
                "5 = every key fact matches the reference; 4 = correct with one minor omission; "
                "3 = partially correct (about half of the key facts); 2 = mostly incorrect or very incomplete; "
                "1 = wrong, irrelevant, or refuses to answer.\n"
                'Output ONLY JSON in the form {"score": <integer 1-5>}.')
        prompts.append([{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}])
    outs = llm.generate_batch(prompts, max_new_tokens=24, batch_size=batch_size)
    return [_parse_score(o) for o in outs]


def batch_json(llm, prompts: List[List[dict]], max_new_tokens: int = 256, batch_size: int = 8):
    outs = llm.generate_batch(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [extract_json(o) for o in outs]


def make_chatbot(gr, **kw):
    """gr.Chatbot across Gradio 4/5 (needs type="messages") and 6 (type argument removed)."""
    try:
        return gr.Chatbot(type="messages", **kw)
    except TypeError:
        return gr.Chatbot(**kw)


def launch_blocks(demo, gr, **kw):
    """demo.launch with theme where supported (Gradio 6 moved `theme` from Blocks() to launch()).
    If `server_port` is busy (e.g. an earlier UI in the same kernel), the next free port is used."""
    import socket
    port = kw.get("server_port")
    if port:
        for p in range(int(port), int(port) + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    continue
            if p != port:
                print(f"[launch] port {port} busy -> using {p}  (http://127.0.0.1:{p})")
            kw["server_port"] = p
            break
    try:
        demo.launch(theme=gr.themes.Soft(), **kw)
    except TypeError:
        demo.launch(**kw)
    return demo


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
