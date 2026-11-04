"""Shared utilities for Quiz1 / Quiz2 / Quiz3.

Project layout
    main.ipynb
    Codes/   (this folder)
    Data/    (datasets are downloaded here automatically if missing)
    Outputs/ (figures Q{n}_*.png, models q{n}_*, csv / json results)
"""
from __future__ import annotations

import base64
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "Outputs"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 12})


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(prefix: str, name: str, dpi: int = 150) -> Path:
    """Save the current figure as Outputs/{prefix}_{name}.png, then display it."""
    path = OUT / f"{prefix}_{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close("all")
    print("saved ->", path)
    return path


def mb(x: float) -> float:
    return x / 2 ** 20


def warn_if_gpu_not_empty(threshold_mb: float = 300) -> float:
    """Return MiB currently allocated; warn if an earlier model is still referenced (inflates memory measurements)."""
    cur = mb(torch.cuda.memory_allocated())
    if cur > threshold_mb:
        print(f"[warning] {cur:.0f} MiB already allocated BEFORE loading - an earlier model/tensor is still referenced "
              "(del it, or restart the kernel) otherwise the memory numbers are inflated.")
    return cur


def report_param_memory(model):
    """Print how many MiB of parameters / buffers live in each dtype (diagnoses unexpectedly large quantized models)."""
    from collections import defaultdict
    d = defaultdict(float)
    for p in model.parameters():
        d[f"param {p.dtype}"] += p.numel() * p.element_size()
    for b in model.buffers():
        d[f"buffer {b.dtype}"] += b.numel() * b.element_size()
    for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
        print(f"{k:28s} {mb(v):9.1f} MiB")
    print(f"{'torch.cuda.memory_allocated':28s} {mb(torch.cuda.memory_allocated()):9.1f} MiB "
          f"(params+buffers = {mb(sum(d.values())):.1f} MiB)")


def cleanup():
    """Release GPU memory (call after `del model`)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def adapter_ctx(model, on: bool = True):
    """Context manager: use LoRA adapter (on=True) or the original base weights (on=False)."""
    if on or not hasattr(model, "disable_adapter"):
        return contextlib.nullcontext()
    return model.disable_adapter()


# --------------------------------------------------------------------------- #
# LLM loading / generation
# --------------------------------------------------------------------------- #
def load_llm(name: str, precision: str = "bf16"):
    """precision in {'bf16', 'int8', 'int4'} (int8/int4 use bitsandbytes)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": {"": torch.cuda.current_device()}}   # follows torch.cuda.set_device(...)
    if precision == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif precision == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["torch_dtype"] = torch.float16
    elif precision == "int4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(precision)
    cleanup()
    before = warn_if_gpu_not_empty()
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    gc.collect(); torch.cuda.empty_cache()          # drop un-collected temporaries from (quantized) loading
    torch.cuda.synchronize()
    info = dict(precision=precision, model_mb=mb(torch.cuda.memory_allocated()) - before)
    print(f"[load_llm] {name} @ {precision}: weights on GPU = {info['model_mb']:.0f} MiB")
    return model, tok, info


@torch.no_grad()
def chat_generate(model, tok, messages, max_new_tokens: int = 512):
    """Greedy generation for ONE conversation, with GPU-memory / latency measurement."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    new = out[0, inp["input_ids"].shape[1]:]
    return dict(text=tok.decode(new, skip_special_tokens=True).strip(),
                n_tokens=int(len(new)), prompt_tokens=int(inp["input_ids"].shape[1]),
                latency=dt, base_mb=mb(base), peak_mb=mb(torch.cuda.max_memory_allocated()))


@torch.no_grad()
def batch_generate(model, tok, messages_list, max_new_tokens: int = 256, bs: int = 8):
    outs = []
    for i in tqdm(range(0, len(messages_list), bs), desc="generate", leave=False):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_list[i:i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


# --------------------------------------------------------------------------- #
# Commercial-LLM judge. Uses Gemini if GEMINI_API_KEY is set (pip install google-genai),
# otherwise an OpenAI-compatible API if OPENAI_API_KEY is set. JUDGE_MODEL overrides the model name.
# --------------------------------------------------------------------------- #
_JUDGE_CACHE = DATA / "judge_cache.json"
_judge_warned = False
_judge_disabled = False      # set when the DAILY quota is exhausted
_last_judge_call = 0.0
JUDGE_MIN_INTERVAL = float(os.environ.get("JUDGE_MIN_INTERVAL", 5.0))   # seconds between calls (free tier ~10-15 RPM)
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def judge_backend():
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def judge_available() -> bool:
    return judge_backend() is not None


def _parse_score(txt: str) -> float:
    try:
        return float(json.loads(txt)["score"])
    except Exception:
        m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', txt)
        if not m:
            raise ValueError(f"cannot parse judge reply: {txt[:120]}")
        return float(m.group(1))


def llm_judge(question: str, answer: str, reference: str | None = None, image=None,
              rubric: str = "Score correctness and instruction-following.") -> float:
    """Return a 1-10 score from a commercial LLM, or NaN if no API key / failure.
    Results are cached in Data/judge_cache.json so re-running the notebook is free."""
    global _judge_warned, _judge_disabled, _last_judge_call
    backend = judge_backend()
    model = os.environ.get("JUDGE_MODEL") or (DEFAULT_GEMINI_MODEL if backend == "gemini" else DEFAULT_OPENAI_MODEL)
    key = hashlib.sha1(json.dumps([question, answer, reference, rubric, model, image is not None],
                                  ensure_ascii=False).encode()).hexdigest()
    cache = json.loads(_JUDGE_CACHE.read_text()) if _JUDGE_CACHE.exists() else {}
    if key in cache:
        return cache[key]
    if _judge_disabled:
        return float("nan")
    if backend is None:
        if not _judge_warned:
            print("[judge] no GEMINI_API_KEY / OPENAI_API_KEY -> LLM-judge scores will be NaN (auto metrics still work).")
            _judge_warned = True
        return float("nan")
    prompt = (
        "You are a strict evaluator. Rate the ASSISTANT ANSWER on a 1-10 scale.\n"
        f"Rubric: {rubric}\n\n[QUESTION]\n{question}\n\n"
        + (f"[REFERENCE / GROUND TRUTH]\n{reference}\n\n" if reference else "")
        + f"[ASSISTANT ANSWER]\n{answer}\n\n"
        'Reply with JSON only: {"score": <integer 1-10>, "reason": "<short>"}')
    score = float("nan")
    for attempt in range(4):                      # retry on rate-limit / transient errors
        wait = JUDGE_MIN_INTERVAL - (time.time() - _last_judge_call)
        if wait > 0:
            time.sleep(wait)                      # throttle: keep under the per-minute quota
        _last_judge_call = time.time()
        try:
            if backend == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
                contents = [prompt]
                if image is not None:
                    im = image.convert("RGB"); im.thumbnail((768, 768)); contents.append(im)
                r = client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                score = _parse_score(r.text)
            else:
                from openai import OpenAI
                client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
                content = [{"type": "text", "text": prompt}]
                if image is not None:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64(image)}"}})
                r = client.chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                                   messages=[{"role": "user", "content": content}])
                score = _parse_score(r.choices[0].message.content)
            break
        except Exception as e:
            msg = str(e)
            if re.search(r"PerDay|per day|daily", msg, re.I):
                print(f"[judge:{backend}] DAILY quota exhausted -> judge disabled for this session. "
                      "Re-run later with Quiz*.rejudge_run(...) to fill the NaN scores.")
                _judge_disabled = True
                return float("nan")
            m = re.search(r"retry(?:Delay)?\W+(?:in\s+)?(\d+(?:\.\d+)?)\s*s", msg, re.I)
            delay = min(75.0, float(m.group(1)) + 1 if m else 10.0 * (attempt + 1))
            print(f"[judge:{backend}] attempt {attempt + 1} failed ({msg[:70]}...) -> waiting {delay:.0f}s")
            time.sleep(delay)
    if score != score:
        return score
    cache[key] = score
    _JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return score


# --------------------------------------------------------------------------- #
# LoRA + DPO (custom loop: works for text LLMs and VLMs, supports early stopping)
# --------------------------------------------------------------------------- #
def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Attach LoRA to the language-model linear layers (vision tower excluded)."""
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    model = get_peft_model(model, cfg)
    for p in model.parameters():           # keep adapters in fp32 for stable optimisation
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def prepare_eval(model):
    model.eval()
    with contextlib.suppress(Exception):
        model.gradient_checkpointing_disable()
    return model


def seq_logp(model, enc: dict) -> torch.Tensor:
    """Sum of log-probs of the response tokens (labels != -100)."""
    labels = enc["labels"]
    inputs = {k: v for k, v in enc.items() if k != "labels"}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits
    logits, tgt = logits[:, :-1], labels[:, 1:]
    mask = tgt != -100
    sel = logits[mask].float()
    lp = torch.log_softmax(sel, -1).gather(-1, tgt[mask].unsqueeze(-1)).squeeze(-1)
    return lp.sum()


def train_dpo(model, train_items, val_items, make_pair, beta=0.1, lr=2e-5, epochs=3, grad_accum=8,
              eval_every=5, patience=3, min_delta=1e-3, seed=42):
    """DPO with a frozen reference = the same model with the LoRA adapter disabled.

    make_pair(item) -> (enc_chosen, enc_rejected); each enc has input_ids / attention_mask / labels (+ pixel values).
    Early stopping: stop when validation DPO-loss has not improved for `patience` evaluations; the best LoRA
    weights are restored at the end.  Returns the history (list of dict)."""
    from transformers import get_linear_schedule_with_warmup
    random.seed(seed)
    dev = model.device

    def to_dev(enc):
        return {k: (v.to(dev, dtype=torch.bfloat16) if v.is_floating_point() else v.to(dev)) for k, v in enc.items()}

    # 1) reference log-probs (adapter disabled)
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for it in tqdm(train_items + val_items, desc="reference log-probs"):
            ec, er = make_pair(it)
            it["ref_c"], it["ref_r"] = float(seq_logp(model, to_dev(ec))), float(seq_logp(model, to_dev(er)))

    def evaluate(items):
        model.eval()
        L, A, M = [], [], []
        with torch.no_grad():
            for it in items:
                ec, er = make_pair(it)
                z = beta * ((seq_logp(model, to_dev(ec)) - it["ref_c"]) - (seq_logp(model, to_dev(er)) - it["ref_r"]))
                L.append(-F.logsigmoid(z).item()); A.append(float(z.item() > 0)); M.append(z.item() / beta)
        model.train()
        return float(np.mean(L)), float(np.mean(A)), float(np.mean(M))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_upd = epochs * math.ceil(len(train_items) / grad_accum)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * n_upd)), n_upd)

    v0 = evaluate(val_items)
    history = [dict(step=0, epoch=0.0, train_loss=float("nan"), val_loss=v0[0], val_acc=v0[1], val_margin=v0[2])]
    best, best_state, bad, step, stop = v0[0], {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}, 0, 0, False
    model.train()
    run_loss, run_n = 0.0, 0
    for ep in range(epochs):
        random.shuffle(train_items)
        for i, it in enumerate(tqdm(train_items, desc=f"epoch {ep + 1}/{epochs}", leave=False)):
            ec, er = make_pair(it)
            z = beta * ((seq_logp(model, to_dev(ec)) - it["ref_c"]) - (seq_logp(model, to_dev(er)) - it["ref_r"]))
            loss = -F.logsigmoid(z)
            (loss / grad_accum).backward()
            run_loss += loss.item(); run_n += 1
            if (i + 1) % grad_accum == 0 or i == len(train_items) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % eval_every == 0:
                    vl, va, vm = evaluate(val_items)
                    history.append(dict(step=step, epoch=ep + (i + 1) / len(train_items), train_loss=run_loss / max(1, run_n),
                                        val_loss=vl, val_acc=va, val_margin=vm))
                    print(f"step {step:3d} | train {run_loss / max(1, run_n):.4f} | val loss {vl:.4f} | val acc {va:.2f} | margin {vm:.2f}")
                    run_loss, run_n = 0.0, 0
                    if vl < best - min_delta:
                        best, bad = vl, 0
                        best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
                    else:
                        bad += 1
                        if bad >= patience:
                            print(f"Early stopping at step {step} (best val loss {best:.4f})")
                            stop = True
                            break
        if stop:
            break
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in best_state:
                p.data.copy_(best_state[k].to(p.device, p.dtype))
    prepare_eval(model)
    return history


def plot_training(history, prefix: str, name: str, title: str):
    import pandas as pd
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h.step, h.val_loss, "o-", label="validation"); ax[0].plot(h.step, h.train_loss, "s--", label="train (running)")
    best = h.val_loss.idxmin(); ax[0].axvline(h.step[best], color="red", ls=":", label="best checkpoint")
    ax[0].set(title="DPO loss", xlabel="optimizer step", ylabel="loss"); ax[0].legend()
    ax[1].plot(h.step, h.val_acc, "o-", color="green"); ax[1].set(title="Validation preference accuracy", xlabel="optimizer step", ylim=(0, 1.05))
    ax[2].plot(h.step, h.val_margin, "o-", color="purple"); ax[2].set(title="Validation reward margin (log-ratio)", xlabel="optimizer step")
    fig.suptitle(title, y=1.03)
    return savefig(prefix, name)
