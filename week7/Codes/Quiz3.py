"""Quiz 3 - Preference fine-tuning (DPO, an RLHF-style method) of an LLM and a VLM.

LLM aspect : "be concise"  (short, to-the-point answers without being told to)
             -> general ability check on 100 unrelated MMLU questions
VLM aspect : "do not mention / confirm objects that are not in the image" (object hallucination)
             -> image-quality robustness experiment (brightness, contrast, blur, JPEG, noise, resolution)
"""
from __future__ import annotations

from common import *  # noqa: F401,F403
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import Quiz2
from Quiz2 import load_image, mentioned_cats, parse_yesno, vlm_generate

PREFIX = "Q3"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
VLM_NAME = Quiz2.VLM_A
CONCISE_SYS = "Answer the question directly and concisely, in at most two short sentences. Do not add extra background."


# =========================================================================== #
# PART A - LLM
# =========================================================================== #
def prepare_alpaca_prompts(n_train=170, n_test=50, seed=0):
    p = DATA / "q3_llm_prompts.json"
    if p.exists():
        return json.loads(p.read_text())
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ok = {"what", "why", "how", "explain", "describe", "who", "when", "where", "which", "tell", "give", "define", "compare",
          "is", "are", "can", "do", "does", "summarize"}
    pool = [r["instruction"].strip() for r in ds if not r["input"] and 20 < len(r["instruction"]) < 200
            and r["instruction"].split()[0].lower() in ok]
    rng = random.Random(seed); rng.shuffle(pool)
    d = {"train": pool[:n_train], "test": pool[n_train:n_train + n_test]}
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"[alpaca] {len(d['train'])} train prompts, {len(d['test'])} test prompts -> {p}")
    return d


def build_llm_pairs(model, tok, prompts, val_size=20, max_chosen=120, max_rejected=300):
    """chosen = answer produced WITH a 'be concise' system prompt; rejected = the model's default (verbose) answer."""
    p = DATA / "q3_llm_dpo_pairs.json"
    if p.exists():
        d = json.loads(p.read_text()); print(f"[pairs] loaded {len(d['train'])} train / {len(d['val'])} val pairs"); return d
    ch = batch_generate(model, tok, [[{"role": "system", "content": CONCISE_SYS}, {"role": "user", "content": q}] for q in prompts], max_chosen, bs=8)
    rj = batch_generate(model, tok, [[{"role": "user", "content": q}] for q in prompts], max_rejected, bs=8)
    pairs = [dict(prompt=q, chosen=c, rejected=r) for q, c, r in zip(prompts, ch, rj)
             if len(c.split()) <= 0.7 * len(r.split()) and len(c.split()) >= 3]
    d = {"train": pairs[val_size:], "val": pairs[:val_size]}
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"[pairs] {len(d['train'])} train + {len(d['val'])} val pairs -> {p}")
    return d


def llm_make_pair(tok):
    def f(it):
        ptxt = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True)
        pi = tok(ptxt, add_special_tokens=False).input_ids
        out = []
        for resp in (it["chosen"], it["rejected"]):
            ri = tok(resp + tok.eos_token, add_special_tokens=False).input_ids
            ids = torch.tensor([pi + ri])
            out.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                            labels=torch.tensor([[-100] * len(pi) + ri])))
        return out[0], out[1]
    return f


def eval_conciseness(model, tok, prompts, tag, adapter=True, max_new_tokens=400, judge=True):
    with adapter_ctx(model, adapter):
        outs = batch_generate(model, tok, [[{"role": "user", "content": q}] for q in prompts], max_new_tokens, bs=8)
    df = pd.DataFrame(dict(prompt=prompts, response=outs))
    df["n_tokens"] = [len(tok(o).input_ids) for o in outs]
    df["n_words"] = [len(o.split()) for o in outs]
    df["truncated"] = df.n_tokens >= max_new_tokens - 1
    df["judge"] = [llm_judge(q, o, None, rubric="Score helpfulness and correctness of the answer (1-10). "
                                                "Do NOT penalise brevity as long as the question is answered.")
                   if judge else float("nan") for q, o in zip(prompts, outs)]
    df["tag"] = tag
    return df


def prepare_mmlu(n=100, seed=0):
    p = DATA / "q3_mmlu100.json"
    if p.exists():
        return json.loads(p.read_text())
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    items = [dict(question=ds[i]["question"], choices=ds[i]["choices"], answer=int(ds[i]["answer"]), subject=ds[i]["subject"]) for i in idx]
    p.write_text(json.dumps(items, ensure_ascii=False, indent=1)); print(f"[mmlu] {n} questions -> {p}")
    return items


@torch.no_grad()
def eval_mmlu(model, tok, items, adapter=True, bs=4):
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    correct = []
    with adapter_ctx(model, adapter):
        for i in tqdm(range(0, len(items), bs), desc="MMLU", leave=False):
            batch = items[i:i + bs]
            texts = []
            for it in batch:
                q = it["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", it["choices"])) + "\nAnswer with only the letter (A, B, C or D)."
                texts.append(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            logits = model(**enc).logits[:, -1, letter_ids]
            correct += [int(a == it["answer"]) for a, it in zip(logits.argmax(-1).tolist(), batch)]
    return np.array(correct)


def plot_llm_dataset(d):
    tr = d["train"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist([len(x["chosen"].split()) for x in tr], bins=20, alpha=.7, label="chosen (concise)")
    ax[0].hist([len(x["rejected"].split()) for x in tr], bins=20, alpha=.7, label="rejected (verbose)")
    ax[0].set(xlabel="# words", ylabel="# pairs", title=f"LLM DPO Dataset: Response Length ({len(tr)} train pairs)"); ax[0].legend()
    ax[1].axis("off"); ex = tr[0]
    ax[1].text(0, 1, f"Example pair\n\nPROMPT: {ex['prompt'][:120]}\n\nCHOSEN: {ex['chosen'][:260]}\n\nREJECTED: {ex['rejected'][:260]}...",
               va="top", wrap=True, fontsize=9)
    ax[1].set_title("Example Preference Pair")
    return savefig(PREFIX, "llm_dataset_stats")


def plot_llm_conciseness(before, after):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].boxplot([before.n_words, after.n_words], showmeans=True); ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["Base", "DPO-tuned"])
    ax[0].set(ylabel="# words per answer", title="Answer Length (optimisation target)")
    for a, v, t in zip(ax[1:], [[(before.n_words <= 60).mean(), (after.n_words <= 60).mean()],
                               [before.judge.mean(), after.judge.mean()]],
                       ["Share of answers <= 60 words", "LLM-judge helpfulness (1-10)"]):
        a.bar(["Base", "DPO-tuned"], v, color=["gray", "tab:green"]); a.set_title(t)
        for i, x in enumerate(v): a.text(i, x, f"{x:.2f}", ha="center", va="bottom")
    fig.suptitle("Q3 LLM: Optimisation Aspect - Conciseness (50 held-out prompts)", y=1.03)
    return savefig(PREFIX, "llm_conciseness_before_after")


def plot_mmlu(b, a):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["Base", "DPO-tuned"], [b.mean(), a.mean()], color=["gray", "tab:green"]); ax[0].set_ylim(0, 1)
    for i, x in enumerate([b.mean(), a.mean()]): ax[0].text(i, x, f"{x:.2%}", ha="center", va="bottom")
    ax[0].set_title("General Benchmark (100 MMLU questions)")
    up, down = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    ax[1].bar(["wrong->right", "right->wrong", "unchanged"], [up, down, len(a) - up - down], color=["tab:green", "tab:red", "lightgray"])
    ax[1].set_title("Per-question Changes"); ax[1].set_ylabel("# questions")
    fig.suptitle("Q3 LLM: Did General Ability Change?", y=1.03)
    return savefig(PREFIX, "llm_general_benchmark")


# =========================================================================== #
# PART B - VLM
# =========================================================================== #
def _list_text(cats):
    cs = ["a " + c for c in cats]
    return "I can see " + (cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1]) + "."


LIST_Q = "List the main objects you can see in this image."


def build_vlm_pairs(metas, seed=0):
    """Per image: positive-existence pair, negative-existence pair, object-listing pair (hallucinated objects = rejected)."""
    rng = random.Random(seed)
    pairs = []
    for m in metas:
        cats = m["cats"]
        vis = sorted([c for c, v in cats.items() if v["frac"] >= 0.03], key=lambda c: -cats[c]["frac"])
        if not vis: continue
        c = rng.choice(vis)
        pairs.append(dict(image_path=m["path"], kind="pos", question=f"Is there a {c} in this image? Answer with Yes or No.",
                          chosen=f"Yes, there is a {c} in the image.", rejected=f"No, there is no {c} in the image."))
        a = rng.choice([x for x in Quiz2.COMMON if x not in cats])
        pairs.append(dict(image_path=m["path"], kind="neg", question=f"Is there a {a} in this image? Answer with Yes or No.",
                          chosen=f"No, there is no {a} in the image.", rejected=f"Yes, there is a {a} in the image."))
        keep = vis[:4]
        fake = rng.sample([x for x in Quiz2.COMMON if x not in cats], rng.choice([1, 2]))
        mixed = keep + fake; rng.shuffle(mixed)
        pairs.append(dict(image_path=m["path"], kind="list", question=LIST_Q, chosen=_list_text(keep), rejected=_list_text(mixed)))
    return pairs


def vlm_make_pair(processor):
    def f(it):
        img = load_image(it["image_path"])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["question"]}]}]
        ptxt = processor.apply_chat_template(msgs, add_generation_prompt=True)
        plen = processor(text=[ptxt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        out = []
        for resp in (it["chosen"], it["rejected"]):
            enc = dict(processor(text=[ptxt + resp + "<|im_end|>"], images=[img], return_tensors="pt"))
            lab = enc["input_ids"].clone(); lab[:, :plen] = -100
            enc["labels"] = lab; out.append(enc)
        return out[0], out[1]
    return f


def plot_vlm_dataset(pairs):
    kinds = pd.Series([p["kind"] for p in pairs]).value_counts()
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 4, 1); kinds.plot.bar(ax=ax, color="steelblue"); ax.set_title(f"VLM DPO Pairs by Type (n={len(pairs)})")
    shown = [next(p for p in pairs if p["kind"] == k) for k in ("pos", "neg", "list")]
    for i, p in enumerate(shown):
        ax = fig.add_subplot(1, 4, 2 + i); ax.imshow(load_image(p["image_path"])); ax.axis("off")
        ax.set_title(f"[{p['kind']}] {p['question'][:45]}\n+ {p['chosen'][:55]}\n- {p['rejected'][:55]}", fontsize=8, loc="left")
    fig.suptitle("Q3 VLM Preference Dataset (chosen +, rejected -)", y=1.0)
    return savefig(PREFIX, "vlm_dataset_samples")


def eval_existence(model, processor, cfg, items, adapter=True, max_new_tokens=6):
    rows = []
    with adapter_ctx(model, adapter):
        for it in tqdm(items, desc="existence", leave=False):
            r = vlm_generate(model, processor, cfg, load_image(it["image_path"]), it["question"], max_new_tokens)
            rows.append(dict(id=it["id"], gt=it["answer"], pred=parse_yesno(r["text"]), text=r["text"]))
    return pd.DataFrame(rows)


def existence_metrics(df):
    y = df["gt"] == "yes"; p = df["pred"] == "yes"
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(accuracy=float((df["gt"] == df["pred"]).mean()), precision=prec, recall=rec,
                f1=2 * prec * rec / max(1e-9, prec + rec), yes_ratio=float(p.mean()), false_positive_rate=fp / max(1, int((~y).sum())))


def eval_caption(model, processor, cfg, metas, adapter=True):
    rows = []
    with adapter_ctx(model, adapter):
        for m in tqdm(metas, desc="caption", leave=False):
            r = vlm_generate(model, processor, cfg, load_image(m["path"]), "Describe this image in one or two sentences.", 96)
            gt, men = set(m["cats"]), mentioned_cats(r["text"])
            rows.append(dict(image=m["path"], text=r["text"], recall=len(gt & men) / max(1, len(gt)),
                             halluc=float(len(men - gt) > 0), n_halluc=len(men - gt), halluc_objs=sorted(men - gt)))
    return pd.DataFrame(rows)


def plot_vlm_before_after(b_ex, a_ex, b_cap, a_cap):
    mb_, ma_ = existence_metrics(b_ex), existence_metrics(a_ex)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ks = ["accuracy", "f1", "yes_ratio", "false_positive_rate"]; x = np.arange(len(ks))
    ax[0].bar(x - .2, [mb_[k] for k in ks], .4, label="Base", color="gray"); ax[0].bar(x + .2, [ma_[k] for k in ks], .4, label="DPO-tuned", color="tab:green")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["Accuracy", "F1", "Yes-ratio", "False-pos. rate"]); ax[0].legend(); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Existence Questions (held-out images)")
    for a, col, t in zip(ax[1:], ["halluc", "recall"], ["Caption hallucination rate (CHAIR-s, lower = better)", "Caption recall of real objects"]):
        v = [b_cap[col].mean(), a_cap[col].mean()]
        a.bar(["Base", "DPO-tuned"], v, color=["gray", "tab:green"]); a.set_title(t, fontsize=10); a.set_ylim(0, 1.05)
        for i, y in enumerate(v): a.text(i, y, f"{y:.2f}", ha="center", va="bottom")
    fig.suptitle("Q3 VLM: Optimisation Aspect - Less Object Hallucination", y=1.03)
    return savefig(PREFIX, "vlm_hallucination_before_after")


# ---- image-quality perturbations ------------------------------------------ #
DEGRADATIONS = {   # kind -> (levels, index of the 'identity' level)
    "brightness": [0.2, 0.5, 1.0, 1.6, 2.5],
    "contrast": [0.2, 0.5, 1.0, 1.6, 2.5],
    "blur_radius": [0, 1, 2, 4, 8],
    "jpeg_quality": [100, 50, 20, 10, 5],
    "gaussian_noise_sigma": [0, 10, 25, 50, 80],
    "resolution_scale": [1.0, 0.5, 0.25, 0.125, 0.0625],
}


def degrade(img: Image.Image, kind: str, level) -> Image.Image:
    if kind == "brightness": return ImageEnhance.Brightness(img).enhance(level)
    if kind == "contrast": return ImageEnhance.Contrast(img).enhance(level)
    if kind == "blur_radius": return img.filter(ImageFilter.GaussianBlur(level)) if level > 0 else img
    if kind == "jpeg_quality":
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(level)); buf.seek(0); return Image.open(buf).convert("RGB")
    if kind == "gaussian_noise_sigma":
        a = np.asarray(img).astype(np.float32) + np.random.RandomState(0).normal(0, level, np.asarray(img).shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "resolution_scale":
        w, h = img.size
        return img.resize((max(8, int(w * level)), max(8, int(h * level))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    raise ValueError(kind)


def eval_degradation(model, processor, cfg, items, modes=("tuned", "base")):
    """Existence accuracy/F1 under every (kind, level) for the tuned model and the base model (adapter disabled)."""
    rows = []
    for kind, levels in DEGRADATIONS.items():
        for lv in levels:
            for mode in modes:
                preds = []
                with adapter_ctx(model, mode == "tuned"):
                    for it in items:
                        img = degrade(load_image(it["image_path"]), kind, lv)
                        preds.append(parse_yesno(vlm_generate(model, processor, cfg, img, it["question"], 6)["text"]))
                df = pd.DataFrame(dict(gt=[i["answer"] for i in items], pred=preds))
                m = existence_metrics(df); m.update(kind=kind, level=lv, mode=mode); rows.append(m)
            print(f"{kind}={lv}: " + " | ".join(f"{r['mode']} acc {r['accuracy']:.2f}" for r in rows[-len(modes):]))
    return pd.DataFrame(rows)


def plot_degradation_curves(res):
    kinds = list(DEGRADATIONS)
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))
    for a, k in zip(axs.ravel(), kinds):
        for mode, c in zip(("tuned", "base"), ("tab:green", "gray")):
            d = res[(res.kind == k) & (res["mode"] == mode)]
            if len(d):
                a.plot(range(len(d)), d.accuracy, "o-", color=c, label=f"{mode} accuracy")
                a.plot(range(len(d)), d.yes_ratio, "x--", color=c, alpha=.5, label=f"{mode} yes-ratio")
        a.set_xticks(range(len(DEGRADATIONS[k]))); a.set_xticklabels([str(l) for l in DEGRADATIONS[k]]); a.set_ylim(0, 1.05)
        a.set_title(k.replace("_", " ").title()); a.set_xlabel("level (identity = brightness/contrast 1.0, blur 0, JPEG 100, noise 0, scale 1.0)", fontsize=7)
        a.legend(fontsize=7)
    fig.suptitle("Q3 VLM: Effect of Image Quality on Existence-QA Performance", y=1.0)
    return savefig(PREFIX, "vlm_degradation_curves")


def plot_degradation_examples(img_path):
    img = load_image(img_path)
    fig, axs = plt.subplots(len(DEGRADATIONS), 5, figsize=(14, 17))
    for r, (k, lvs) in enumerate(DEGRADATIONS.items()):
        for c, lv in enumerate(lvs):
            axs[r, c].imshow(degrade(img, k, lv)); axs[r, c].axis("off"); axs[r, c].set_title(f"{k}={lv}", fontsize=8)
    fig.suptitle("Q3 VLM: Image Degradation Examples", y=0.995)
    return savefig(PREFIX, "vlm_degradation_examples")


def save_adapter(model, tok_or_proc, name):
    path = OUT / name
    model.save_pretrained(path); tok_or_proc.save_pretrained(path)
    print("saved ->", path)
    return path
