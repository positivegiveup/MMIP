"""Quiz 2 - Deploy two open-source VLMs and compare them on a 20-item vision benchmark."""
from __future__ import annotations

from common import *  # noqa: F401,F403
import pandas as pd
from PIL import Image

import warnings
warnings.filterwarnings("ignore", message=".*min_pixels.*max_pixels.*", category=FutureWarning)

PREFIX = "Q2"
VLM_A = "Qwen/Qwen2-VL-2B-Instruct"      # bf16 ~4.5 GB
VLM_B = "HuggingFaceTB/SmolVLM-Instruct"   # bf16 ~4.5 GB

VLM_CONFIGS = {
    VLM_A: dict(short="qwen2vl2b", proc_kwargs=dict(min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28), nested=False),
    VLM_B: dict(short="smolvlm2b", proc_kwargs=dict(do_image_splitting=False), nested=True),
}

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
          "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
          "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
COMMON = ["person", "car", "chair", "dog", "cat", "bottle", "cup", "bench", "bus", "tv", "laptop", "bicycle", "umbrella",
          "truck", "horse", "bird", "clock", "bed", "pizza", "train", "airplane", "boat", "giraffe", "elephant", "zebra",
          "cow", "sheep", "skateboard", "kite", "vase"]
SYN = {"person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid", "kids",
                  "player", "lady", "guy", "skier", "surfer", "rider"],
       "bicycle": ["bicycle", "bicycles", "bike", "bikes"], "car": ["car", "cars", "sedan", "taxi"],
       "motorcycle": ["motorcycle", "motorcycles", "motorbike", "scooter"], "airplane": ["airplane", "plane", "jet", "aircraft"],
       "bus": ["bus", "buses"], "truck": ["truck", "trucks"], "boat": ["boat", "boats", "ship", "sailboat"],
       "tv": ["tv", "television"], "couch": ["couch", "sofa"], "dining table": ["dining table", "table"],
       "cell phone": ["cell phone", "phone", "smartphone"], "potted plant": ["potted plant", "plant", "plants"],
       "sports ball": ["ball"], "wine glass": ["wine glass"], "hot dog": ["hot dog"], "teddy bear": ["teddy bear", "teddy"],
       "traffic light": ["traffic light", "stoplight"], "fire hydrant": ["fire hydrant", "hydrant"],
       "skis": ["skis", "ski"], "refrigerator": ["refrigerator", "fridge"], "remote": ["remote", "remote control"]}
_PAT = {c: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in SYN.get(c, [c, c + "s"])) + r")\b", re.I) for c in COCO80}


# --------------------------------------------------------------------------- #
# Dataset: COCO val (streamed from the HF hub, cached under Data/)
# --------------------------------------------------------------------------- #
def prepare_coco(n_images: int = 420, seed: int = 0, min_side: int = 200):
    """Download n_images COCO val images + object annotations if not present; return list of meta dicts."""
    meta_p = DATA / "coco_val_meta.json"
    if meta_p.exists():
        metas = json.loads(meta_p.read_text())
        if len(metas) >= n_images:
            print(f"[coco] using cached {len(metas)} images")
            return metas[:n_images]
    from datasets import load_dataset
    img_dir = DATA / "coco_val"; img_dir.mkdir(exist_ok=True)
    ds = load_dataset("detection-datasets/coco", split="val", streaming=True).shuffle(seed=seed, buffer_size=1000)
    names = None
    try:
        names = ds.features["objects"].feature["category"].names
    except Exception:
        pass
    names = names or COCO80
    metas = []
    for ex in tqdm(ds, total=n_images, desc="downloading COCO val"):
        W, H = ex["width"], ex["height"]
        objs, cats = ex["objects"], {}
        areas = objs["area"] if "area" in objs else [(b[2] - b[0]) * (b[3] - b[1]) for b in objs["bbox"]]
        for c, a in zip(objs["category"], areas):
            name = names[c] if isinstance(c, int) else c
            d = cats.setdefault(name, {"n": 0, "frac": 0.0}); d["n"] += 1; d["frac"] = min(1.0, d["frac"] + a / (W * H))
        if min(W, H) < min_side or len(cats) < 2:
            continue
        rel = f"coco_val/{ex['image_id']}.jpg"
        if not (DATA / rel).exists():
            ex["image"].convert("RGB").save(DATA / rel, quality=95)
        metas.append(dict(image_id=ex["image_id"], path=rel, width=W, height=H, cats=cats))
        if len(metas) >= n_images:
            break
    meta_p.write_text(json.dumps(metas))
    print(f"[coco] saved {len(metas)} images to {img_dir}")
    return metas


def load_image(path_rel: str) -> Image.Image:
    return Image.open(DATA / path_rel).convert("RGB")


# --------------------------------------------------------------------------- #
# Benchmark construction (task types: yes / no / count / mcq / caption)
# --------------------------------------------------------------------------- #
# Easy types (used by Q3's existence test) + HARD types (Q2 benchmark v2)
TYPE_NAMES = {"yes": "Existence (present)", "no": "Existence (absent)", "count": "Counting", "mcq": "Classification (MCQ)",
              "yes_small": "Small-object existence", "no_adv": "Adversarial absent object", "count_hard": "Counting (4-9)",
              "mcq_hard": "Prominence MCQ (present distractors)", "spatial": "Spatial left/right", "caption": "Image description"}
TYPE_ORDER = ["yes", "no", "yes_small", "no_adv", "count", "count_hard", "mcq", "mcq_hard", "spatial", "caption"]
# 20 items; rare / strict types first so they can claim the images they need
HARD_QUOTAS = {"spatial": 3, "count_hard": 3, "mcq_hard": 3, "yes_small": 3, "no_adv": 4, "caption": 4}
# annotated-but-often-missed categories are never used as "absent" targets (COCO annotation is incomplete for them)
_UNSAFE_ABSENT = {"person", "dining table", "chair", "book", "potted plant", "cup", "bottle", "bowl", "handbag", "backpack", "sports ball"}
_CTX = {}


def cooccurrence(metas):
    """P(c present | p present) estimated on the downloaded COCO images (used to pick adversarial absent objects)."""
    n, pair = {}, {}
    for m in metas:
        cs = list(m["cats"])
        for p in cs:
            n[p] = n.get(p, 0) + 1
            for c in cs:
                if c != p:
                    pair[(p, c)] = pair.get((p, c), 0) + 1
    return n, pair


def _coco_names(ds):
    try:
        return ds.features["objects"].feature["category"].names
    except Exception:
        return COCO80


def attach_boxes(metas):
    """Add normalised boxes m['boxes'] = {category: [[x1,y1,x2,y2], ...]} (needed by the spatial task).
    One-time streaming pass over COCO val; cached in Data/coco_val_boxes.json."""
    p = DATA / "coco_val_boxes.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    pending = {str(m["image_id"]) for m in metas} - set(cache)
    if pending:
        from datasets import load_dataset
        from datasets import Image as HFImage
        ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
        names = _coco_names(ds)
        try:
            ds = ds.cast_column("image", HFImage(decode=False))      # we only need annotations
        except Exception:
            pass
        raw = {}
        for ex in tqdm(ds, desc="collecting COCO boxes (one-time)"):
            iid = str(ex["image_id"])
            if iid in pending:
                raw[iid] = (ex["width"], ex["height"], ex["objects"]); pending.discard(iid)
                if not pending:
                    break
        # detect the bbox format ([x1,y1,x2,y2] vs [x,y,w,h]) by comparing with the annotated 'area'
        votes = [0, 0]
        for W, H, o in list(raw.values())[:60]:
            if "area" not in o: continue
            for bb, ar in zip(o["bbox"], o["area"]):
                a1, a2 = (bb[2] - bb[0]) * (bb[3] - bb[1]), bb[2] * bb[3]
                if a1 > 0 and a2 > 0 and ar > 0:
                    votes[abs(math.log(ar / a1)) > abs(math.log(ar / a2))] += 1
        xyxy = votes[0] >= votes[1]
        print(f"[boxes] bbox format detected: {'xyxy' if xyxy else 'xywh'} (votes {votes})")
        for iid, (W, H, o) in raw.items():
            d = {}
            for c, bb in zip(o["category"], o["bbox"]):
                x1, y1, x2, y2 = (bb[0], bb[1], bb[2], bb[3]) if xyxy else (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                d.setdefault(names[c] if isinstance(c, int) else c, []).append([x1 / W, y1 / H, x2 / W, y2 / H])
            cache[iid] = d
        p.write_text(json.dumps(cache))
    for m in metas:
        m["boxes"] = cache.get(str(m["image_id"]))
    print(f"[boxes] attached to {sum(m['boxes'] is not None for m in metas)}/{len(metas)} images")
    return metas


def _make_item(typ, m, rng):
    cats = m["cats"]
    base = dict(type=typ, image_id=m["image_id"], image_path=m["path"], gt_cats=sorted(cats))
    # ---------------- easy types ----------------
    if typ == "yes":
        c = [c for c, v in cats.items() if v["frac"] >= 0.03]
        if not c: return None
        c = rng.choice(sorted(c))
        return dict(base, question=f"Is there a {c} in this image? Answer with Yes or No.", answer="yes", target=c)
    if typ == "no":
        c = rng.choice([c for c in COMMON if c not in cats])
        return dict(base, question=f"Is there a {c} in this image? Answer with Yes or No.", answer="no", target=c)
    if typ == "count":
        c = [c for c, v in cats.items() if 1 <= v["n"] <= 5 and v["frac"] >= 0.02]
        if not c: return None
        c = rng.choice(sorted(c))
        return dict(base, question=f"How many instances of '{c}' are visible in this image? Answer with a single number.",
                    answer=str(cats[c]["n"]), target=c)
    if typ == "mcq":
        ranked = sorted(cats.items(), key=lambda kv: -kv[1]["frac"])
        if ranked[0][1]["frac"] < 0.15 or (len(ranked) > 1 and ranked[1][1]["frac"] > 0.6 * ranked[0][1]["frac"]): return None
        right = ranked[0][0]
        opts = [right] + rng.sample([c for c in COMMON if c not in cats], 3)
        rng.shuffle(opts)
        q = "Which object is the most prominent in this image?\n" + "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts)) \
            + "\nAnswer with the letter only."
        return dict(base, question=q, answer="ABCD"[opts.index(right)], target=right)
    # ---------------- hard types ----------------
    if typ == "yes_small":      # a single, small (0.3%-2% of the image) annotated object
        c = [c for c, v in cats.items() if v["n"] == 1 and 0.003 <= v["frac"] <= 0.02]
        if not c: return None
        c = rng.choice(sorted(c))
        return dict(base, question=f"Is there a {c} in this image? Answer with Yes or No.", answer="yes", target=c)
    if typ == "no_adv":         # absent object that very often co-occurs with what IS in the image
        n, pair = _CTX["cooc"]
        sc = {}
        for c in COCO80:
            if c in cats or c in _UNSAFE_ABSENT or n.get(c, 0) < 5: continue
            sc[c] = max((pair.get((p, c), 0) / n[p] for p in cats if n.get(p, 0) >= 5), default=0.0)
        top = sorted(sc, key=sc.get, reverse=True)[:3]
        if not top or sc[top[0]] < 0.12: return None
        c = rng.choice(top)
        return dict(base, question=f"Is there a {c} in this image? Answer with Yes or No.", answer="no", target=c,
                    cooc=round(sc[c], 3))
    if typ == "count_hard":     # 4-9 instances of a clearly visible category
        c = [c for c, v in cats.items() if 4 <= v["n"] <= 9 and v["frac"] >= 0.03]
        if not c: return None
        c = rng.choice(sorted(c))
        return dict(base, question=f"How many instances of '{c}' are visible in this image? Answer with a single number.",
                    answer=str(cats[c]["n"]), target=c)
    if typ == "mcq_hard":       # distractors are objects that ARE in the image, only one is the most prominent
        ranked = sorted(cats.items(), key=lambda kv: -kv[1]["frac"])
        if len(ranked) < 3 or ranked[0][1]["frac"] < 0.12 or ranked[1][1]["frac"] > 0.5 * ranked[0][1]["frac"]: return None
        right = ranked[0][0]
        opts = [right] + [c for c, _ in ranked[1:4]]
        opts += rng.sample([c for c in COMMON if c not in cats], 4 - len(opts))
        rng.shuffle(opts)
        q = "Which of these objects occupies the largest part of the image?\n" + "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts)) \
            + "\nAnswer with the letter only."
        return dict(base, question=q, answer="ABCD"[opts.index(right)], target=right)
    if typ == "spatial":        # two single-instance objects that are horizontally separated
        bx = m.get("boxes")
        if not bx: return None
        single = [c for c, v in cats.items() if v["n"] == 1 and v["frac"] >= 0.01 and c in bx and len(bx[c]) == 1]
        pairs = [(a, b) for a in single for b in single if a != b and bx[a][0][2] <= bx[b][0][0] - 0.05]   # a entirely left of b
        if not pairs: return None
        a, b = rng.choice(sorted(pairs))
        if rng.random() < 0.5:
            q, ans = f"In this image, is the {a} on the left or on the right of the {b}? Answer with Left or Right.", "left"
        else:
            q, ans = f"In this image, is the {b} on the left or on the right of the {a}? Answer with Left or Right.", "right"
        return dict(base, question=q, answer=ans, target=f"{a}|{b}")
    if typ == "caption":
        if len(cats) < (4 if m.get("_hard_caption") else 1): return None
        return dict(base, question="Describe this image in one or two sentences.", answer="", target="")
    raise ValueError(typ)


def build_vlm_items(metas, quotas=None, seed=0, cooc_from=None):
    """metas = candidate image pool; cooc_from = (bigger) list of metas used to estimate object co-occurrence."""
    quotas = quotas or HARD_QUOTAS
    _CTX["cooc"] = cooccurrence(cooc_from or metas)
    rng = random.Random(seed)
    pool = [dict(m, _hard_caption=True) for m in metas]
    rng.shuffle(pool)
    items, used = [], set()
    for typ, q in quotas.items():
        c = 0
        for m in pool:
            if c >= q: break
            if m["image_id"] in used: continue
            it = _make_item(typ, m, rng)
            if it:
                it["id"] = len(items) + 1; items.append(it); used.add(m["image_id"]); c += 1
        if c < q:
            print(f"[warning] only {c}/{q} eligible images for type '{typ}' - enlarge the image pool")
    return items


def build_existence_set(metas, seed=0):
    """For each image: one 'present' and one 'absent' yes/no question (POPE-style)."""
    rng = random.Random(seed)
    items = []
    for m in metas:
        for typ in ("yes", "no"):
            it = _make_item(typ, m, rng)
            if it:
                it["id"] = len(items) + 1; items.append(it)
    return items


def save_items(items, name):
    p = DATA / name
    p.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print("saved ->", p)


# --------------------------------------------------------------------------- #
# VLM loading / inference
# --------------------------------------------------------------------------- #
def load_vlm(name, precision="bf16", proc_overrides=None):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    cfg = dict(VLM_CONFIGS.get(name, dict(short=name.split("/")[-1].lower(), proc_kwargs={}, nested=False)))
    processor = AutoProcessor.from_pretrained(name, **{**cfg["proc_kwargs"], **(proc_overrides or {})})
    kw = {"device_map": {"": torch.cuda.current_device()}, "torch_dtype": torch.bfloat16}
    if precision == "int4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    cleanup()
    before = warn_if_gpu_not_empty()
    model = AutoVLM.from_pretrained(name, **kw).eval()
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize()
    info = dict(model=name, precision=precision, model_mb=mb(torch.cuda.memory_allocated()) - before)
    print(f"[load_vlm] {name}: weights on GPU = {info['model_mb']:.0f} MiB")
    return model, processor, cfg, info


@torch.no_grad()
def vlm_generate(model, processor, cfg, image, prompt, max_new_tokens=64):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    imgs = [[image]] if cfg.get("nested") else [image]
    inp = processor(text=[text], images=imgs, return_tensors="pt").to(model.device, dtype=model.dtype)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    new = out[:, inp["input_ids"].shape[1]:]
    txt = processor.batch_decode(new, skip_special_tokens=True)[0].strip()
    return dict(text=txt, n_tokens=int(new.shape[1]), latency=dt, base_mb=mb(base), peak_mb=mb(torch.cuda.max_memory_allocated()))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_NUMW = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def parse_yesno(t):
    m = re.match(r"\W*(yes|no)\b", t.strip().lower())
    return m.group(1) if m else None


def parse_number(t):
    m = re.search(r"\d+", t)
    if m: return m.group(0)
    for w, i in _NUMW.items():
        if re.search(rf"\b{w}\b", t.lower()): return str(i)
    return None


def parse_letter(t):
    m = re.match(r"\W*\(?([A-D])\b", t.strip()) or re.search(r"\b([A-D])\b", t)
    return m.group(1) if m else None


def mentioned_cats(text):
    return {c for c, p in _PAT.items() if p.search(text)}


def parse_leftright(t):
    m = re.search(r"\b(left|right)\b", t.lower())
    return m.group(1) if m else None


def score_item(item, resp):
    typ = item["type"]
    out = dict(correct=float("nan"), recall=float("nan"), halluc=float("nan"), abs_err=float("nan"))
    if typ in ("yes", "no", "yes_small", "no_adv"):
        out["correct"] = float(parse_yesno(resp) == item["answer"])
    elif typ in ("count", "count_hard"):
        pred = parse_number(resp)
        out["correct"] = float(pred == item["answer"])
        out["abs_err"] = abs(int(pred) - int(item["answer"])) if pred is not None else float("nan")
    elif typ in ("mcq", "mcq_hard"):
        out["correct"] = float(parse_letter(resp) == item["answer"])
    elif typ == "spatial":
        out["correct"] = float(parse_leftright(resp) == item["answer"])
    if typ == "caption" or item.get("force_caption_metrics"):
        gt, men = set(item["gt_cats"]), mentioned_cats(resp)
        out["recall"] = len(gt & men) / max(1, len(gt))
        out["halluc"] = float(len(men - gt) > 0)          # CHAIR_s-style: any mentioned-but-absent object
        out["n_halluc_obj"] = len(men - gt)
    return out


def run_vlm(model, processor, cfg, items, tag, model_mb=float("nan"), judge=True):
    vlm_generate(model, processor, cfg, load_image(items[0]["image_path"]), "Hi", 4)   # warm-up
    rows = []
    for it in tqdm(items, desc=tag):
        img = load_image(it["image_path"])
        r = vlm_generate(model, processor, cfg, img, it["question"], 96 if it["type"] == "caption" else 12)
        r.update(score_item(it, r["text"]))
        r.update(id=it["id"], type=it["type"], question=it["question"], answer=it["answer"], model=tag, model_mb=model_mb)
        if judge:
            ref = it["answer"] if it["type"] != "caption" else "Objects annotated in the image: " + ", ".join(it["gt_cats"])
            r["judge"] = llm_judge(it["question"], r["text"], ref, image=img,
                                   rubric="Look at the image. Score correctness (and, for descriptions, accuracy without "
                                          "hallucinated objects, plus completeness).")
        else:
            r["judge"] = float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def rejudge_run(df, items):
    """Fill NaN judge scores of an existing VLM run without re-running the model."""
    df = df.copy()
    for i in df.index[df.judge.isna()]:
        it = next(x for x in items if x["id"] == df.at[i, "id"])
        ref = it["answer"] if it["type"] != "caption" else "Objects annotated in the image: " + ", ".join(it["gt_cats"])
        df.at[i, "judge"] = llm_judge(it["question"], df.at[i, "text"], ref, image=load_image(it["image_path"]),
                                      rubric="Look at the image. Score correctness (and, for descriptions, accuracy without "
                                             "hallucinated objects, plus completeness).")
    print(f"remaining NaN judge scores: {int(df.judge.isna().sum())}")
    return df


def plot_benchmark_samples(items):
    rows = math.ceil(len(items) / 5)
    fig, axs = plt.subplots(rows, 5, figsize=(16, 3.3 * rows))
    for a in axs.ravel(): a.axis("off")
    for a, it in zip(axs.ravel(), items):
        a.imshow(load_image(it["image_path"]))
        a.set_title(f"#{it['id']} {TYPE_NAMES[it['type']]}", fontsize=8)
    fig.suptitle(f"Q2 Custom Vision Benchmark: {len(items)} Items (hard version)", y=0.93)
    return savefig(PREFIX, "benchmark_samples")


def _objective_types(dfs):
    present = set().union(*[set(d.type) for d in dfs.values()])
    return [t for t in TYPE_ORDER if t in present and t != "caption"]


def plot_task_accuracy(dfs: dict, name="accuracy_by_task"):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    types = _objective_types(dfs); x = np.arange(len(types)); w = 0.8 / len(dfs)
    for i, (n, d) in enumerate(dfs.items()):
        v = [d[d.type == t].correct.mean() for t in types]
        pos = x + i * w - 0.4 + w / 2
        ax[0].bar(pos, v, w, label=f"{n} (overall {d.correct.mean():.2f})")
        for xi, vi in zip(pos, v): ax[0].text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].axhline(0.5, color="gray", ls=":", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels([TYPE_NAMES[t].replace(" (", "\n(") for t in types], fontsize=7)
    ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("Accuracy"); ax[0].legend(fontsize=8); ax[0].set_title("Accuracy by Task Type")
    names = ["GT-object recall", "Hallucination rate (CHAIR-s)"]
    for i, (n, d) in enumerate(dfs.items()):
        c = d[d.type == "caption"]; v = [c.recall.mean(), c.halluc.mean()]
        pos = np.arange(2) + i * w - 0.4 + w / 2
        ax[1].bar(pos, v, w, label=n)
        for xi, vi in zip(pos, v): ax[1].text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylim(0, 1.1); ax[1].legend(fontsize=8)
    ax[1].set_title("Image Description Quality")
    return savefig(PREFIX, name)


def summarize_vlm(dfs: dict) -> pd.DataFrame:
    rows = []
    for n, d in dfs.items():
        r = {"model": n}
        for t in _objective_types(dfs):
            r[f"acc_{t}"] = d[d.type == t].correct.mean()
        r["acc_overall"] = d.correct.mean()
        cnt = d[d.type.isin(["count", "count_hard"])].abs_err
        r["count_MAE"] = cnt.mean() if len(cnt) else float("nan")
        c = d[d.type == "caption"]
        r["cap_recall"], r["cap_halluc_rate"] = c.recall.mean(), c.halluc.mean()
        yn = d[d.type.isin(["yes", "yes_small", "no", "no_adv"])]
        r["yes_rate"] = float((yn.text.str.lower().str.match(r"\W*yes")).mean()) if len(yn) else float("nan")
        r["judge_avg"], r["mem_mean_GiB"], r["latency_s"] = d.judge.mean(), d.peak_mb.mean() / 1024, d.latency.mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("model").round(3)


def plot_efficiency(dfs: dict):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    names = list(dfs)
    for a, v, t in zip(ax, [[d.model_mb.iloc[0] / 1024 for d in dfs.values()], [d.peak_mb.mean() / 1024 for d in dfs.values()],
                            [d.latency.mean() for d in dfs.values()]],
                       ["Weights on GPU (GiB)", "Mean peak GPU memory per inference (GiB)", "Mean latency per item (s)"]):
        a.bar(names, v, color=["tab:blue", "tab:orange"][:len(names)]); a.set_title(t, fontsize=10)
        for i, vi in enumerate(v): a.text(i, vi, f"{vi:.2f}", ha="center", va="bottom")
    fig.suptitle("Q2 VLM Efficiency Comparison", y=1.03)
    return savefig(PREFIX, "efficiency")


def plot_judge(dfs: dict):
    if all(d.judge.isna().all() for d in dfs.values()):
        print("No LLM-judge scores (no judge API key set) -> skip judge plot"); return None
    types = [t for t in TYPE_ORDER if t in set().union(*[set(d.type) for d in dfs.values()])]
    x = np.arange(len(types)); w = 0.8 / len(dfs)
    plt.figure(figsize=(13, 4.5))
    for i, (n, d) in enumerate(dfs.items()):
        plt.bar(x + i * w - 0.4 + w / 2, [d[d.type == t].judge.mean() for t in types], w, label=f"{n} (avg {d.judge.mean():.2f})")
    plt.xticks(x, [TYPE_NAMES[t].replace(" (", "\n(") for t in types], fontsize=7); plt.ylim(0, 10.5); plt.ylabel("Gemini judge score (1-10)")
    plt.legend(); plt.title("Q2 Commercial-LLM Judge Scores by Task Type")
    return savefig(PREFIX, "judge_scores")


def plot_qualitative(items, dfs: dict, ids=None):
    """Show one example per task type (default), with the answer of every model."""
    if ids is None:
        seen, ids = set(), []
        for it in items:
            if it["type"] not in seen:
                seen.add(it["type"]); ids.append(it["id"])
    fig, axs = plt.subplots(1, len(ids), figsize=(4.6 * len(ids), 6.5))
    axs = np.atleast_1d(axs)
    for a, i in zip(axs, ids):
        it = next(x for x in items if x["id"] == i); a.imshow(load_image(it["image_path"])); a.axis("off")
        txt = f"[{TYPE_NAMES[it['type']]}]\nQ: {it['question'].splitlines()[0][:60]}\nGT: {it['answer'] or '-'}\n" + "\n".join(
            f"{n}: {d[d.id == i].text.iloc[0][:70]}" for n, d in dfs.items())
        a.set_title(txt, fontsize=7, loc="left")
    fig.suptitle("Q2 Qualitative Examples (one per task type)", y=1.02)
    return savefig(PREFIX, "qualitative_examples")


def run_both(items, judge=False, models=(VLM_A, VLM_B)):
    """Load each VLM in turn, run the same items, free the GPU. Returns {short_name: DataFrame}."""
    dfs = {}
    for name in models:
        model, proc, cfg, info = load_vlm(name)
        dfs[cfg["short"]] = run_vlm(model, proc, cfg, items, cfg["short"], model_mb=info["model_mb"], judge=judge)
        del model, proc; cleanup()
    return dfs
