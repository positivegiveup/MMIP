"""Quiz 3 - Evaluating RAG systems.

Part A  Evaluate the Quiz-1 text RAG with RAGAS-style metrics.
        (The official `ragas` package defaults to OpenAI judges; to stay fully local on one 3090 we
        re-implement the RAGAS metric definitions with Qwen2.5-7B as the judge:
        faithfulness, answer relevancy, context precision, context recall  + answer correctness.)
Part B  Extend the idea to VideoRAG: modality-stratified retrieval (speech vs silent), temporal-tolerant
        clip hits, modality ablation / alpha sweep, visual faithfulness judged by the VLM on raw frames,
        citation (reference) accuracy and multi-turn query-rewrite ablation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from Codes.common import *  # noqa: F401,F403
from Codes import Quiz1 as Q1
from Codes import Quiz2 as Q2

set_style()

# =============================================================================
# Part A - RAGAS-style evaluation of the text RAG
# =============================================================================


def build_rag_records(llm, index, qa_eval, k: int, technique: str) -> list:
    qs = [q["question"] for q in qa_eval]
    hits_list = index.search_many(qs, k)
    answers, _ = Q1.answer_batch(llm, technique, qs, hits_list)
    recs = []
    for q, hits, a in zip(qa_eval, hits_list, answers):
        flags = [int(Q1.chunk_hits_evidence(h["chunk"], q["doc_id"], q["ev_start"], q["ev_end"])) for h in hits]
        recs.append(dict(qid=q["id"], doc_id=q["doc_id"], question=q["question"], gold=q["answer"], answer=a,
                         contexts=[h["chunk"]["text"] for h in hits], flags=flags))
    return recs


def _ctx_block(ctxs, limit: int = 6000) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(ctxs))[:limit]


def _msg(user: str):
    return [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}]


def _decompose_prompt(question, text):
    return _msg("Break the ANSWER into a list of atomic, self-contained factual statements (resolve pronouns, "
                "keep numbers and names, at most 6 statements). If the answer only says that information is "
                "insufficient / unknown, return an empty list.\n"
                f"Question: {question}\nAnswer: {text}\n"
                'Output ONLY JSON: {"statements": ["...", "..."]}')


def _verify_prompt(ctxs, statements):
    lst = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    return _msg(f"Context:\n{_ctx_block(ctxs)}\n\nFor each statement decide whether it can be directly inferred from "
                "the context (1 = yes, 0 = no).\n"
                f"Statements:\n{lst}\n"
                'Output ONLY JSON: {"verdicts": [1, 0, ...]} with exactly one entry per statement.')


def _as_list(obj, key):
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        return obj[key]
    return []


def _verdicts(obj, n):
    v = []
    for x in _as_list(obj, "verdicts")[:n]:
        try:
            v.append(1 if int(x) == 1 else 0)
        except (TypeError, ValueError):
            v.append(0)
    return v + [0] * (n - len(v))


def ragas_style_eval(llm, embedder, recs: list) -> pd.DataFrame:
    n = len(recs)
    # 1. statements of answer / gold ------------------------------------------------------------
    ans_st = [[str(s) for s in _as_list(o, "statements")] for o in
              batch_json(llm, [_decompose_prompt(r["question"], r["answer"]) for r in recs], 400)]
    gold_st = [[str(s) for s in _as_list(o, "statements")] for o in
               batch_json(llm, [_decompose_prompt(r["question"], r["gold"]) for r in recs], 400)]
    # 2. faithfulness: answer statements supported by retrieved contexts ---------------------------
    faith = [np.nan] * n
    idx = [i for i in range(n) if ans_st[i]]
    outs = batch_json(llm, [_verify_prompt(recs[i]["contexts"], ans_st[i]) for i in idx], 200)
    for i, o in zip(idx, outs):
        faith[i] = float(np.mean(_verdicts(o, len(ans_st[i]))))
    # 3. context recall: gold statements attributable to contexts ----------------------------------
    recall = [np.nan] * n
    idx = [i for i in range(n) if gold_st[i]]
    outs = batch_json(llm, [_verify_prompt(recs[i]["contexts"], gold_st[i]) for i in idx], 200)
    for i, o in zip(idx, outs):
        recall[i] = float(np.mean(_verdicts(o, len(gold_st[i]))))
    # 4. context precision (RAGAS average precision of LLM relevance judgements) --------------------
    prompts = []
    for r in recs:
        lst = "\n\n".join(f"[{i + 1}] {c[:700]}" for i, c in enumerate(r["contexts"]))
        prompts.append(_msg(f"Question: {r['question']}\nReference answer: {r['gold']}\n\nContexts:\n{lst}\n\n"
                            "For every numbered context output 1 if it contains information useful for deriving the "
                            'reference answer, else 0. Output ONLY JSON: {"relevance": [1, 0, ...]}'))
    cprec = []
    for r, o in zip(recs, batch_json(llm, prompts, 120)):
        rel = [1 if str(x).strip() in ("1", "True", "true") else 0 for x in _as_list(o, "relevance")][:len(r["contexts"])]
        rel += [0] * (len(r["contexts"]) - len(rel))
        if sum(rel) == 0:
            cprec.append(0.0)
            continue
        prec = [sum(rel[:k + 1]) / (k + 1) for k in range(len(rel))]
        cprec.append(float(sum(p * x for p, x in zip(prec, rel)) / sum(rel)))
    # 5. answer relevancy: similarity between the question and questions regenerated from the answer
    refused = [is_refusal(r["answer"]) for r in recs]
    gen_prompts = [_msg(f"Generate 3 different questions that the following answer directly answers.\nAnswer: {r['answer']}\n"
                        'Output ONLY JSON: {"questions": ["...", "...", "..."]}') for r in recs]
    rel_ans = []
    for r, o, ref in zip(recs, batch_json(llm, gen_prompts, 200), refused):
        qs = [str(q) for q in _as_list(o, "questions")][:3]
        if ref or not qs:
            rel_ans.append(0.0)          # RAGAS: non-committal answers get 0
            continue
        e = embedder.encode([r["question"]] + qs)
        rel_ans.append(float((e[1:] @ e[0]).mean()))
    # 6. answer correctness (LLM judge + char F1 + semantic similarity) ----------------------------
    judge = judge_correctness(llm, [(r["question"], r["gold"], r["answer"]) for r in recs])
    ea, eg = embedder.encode([r["answer"] for r in recs]), embedder.encode([r["gold"] for r in recs])
    sem = (ea * eg).sum(1)
    # 7. retrieval metrics from gold evidence ------------------------------------------------------
    rows = []
    for i, r in enumerate(recs):
        fl = r["flags"]
        first = next((k + 1 for k, f in enumerate(fl) if f), None)
        rows.append(dict(qid=r["qid"], doc_id=r["doc_id"], hit=float(any(fl)), mrr=(1 / first if first else 0.0),
                         ndcg=ndcg_at_k(fl, len(fl), max(1, sum(fl))), ctx_precision=cprec[i], ctx_recall=recall[i],
                         faithfulness=faith[i], answer_relevancy=rel_ans[i], answer_correctness=judge[i] / 5,
                         char_f1=char_f1(r["answer"], r["gold"]), sem_sim=float(sem[i]), refused=float(refused[i]),
                         question=r["question"], gold=r["gold"], answer=r["answer"]))
    return pd.DataFrame(rows)


RAGAS_COLS = ["hit", "mrr", "ctx_precision", "ctx_recall", "faithfulness", "answer_relevancy", "answer_correctness"]


def plot_ragas_radar(df: pd.DataFrame, name="Q3_ragas_radar"):
    vals = [float(np.nanmean(df[c])) for c in RAGAS_COLS]
    ang = np.linspace(0, 2 * np.pi, len(RAGAS_COLS), endpoint=False).tolist()
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(ang + ang[:1], vals + vals[:1], color="#4C78A8", lw=2)
    ax.fill(ang + ang[:1], vals + vals[:1], color="#4C78A8", alpha=0.25)
    ax.set_xticks(ang)
    ax.set_xticklabels([c.replace("_", "\n") for c in RAGAS_COLS], fontsize=9)
    ax.set_ylim(0, 1)
    for a, v in zip(ang, vals):
        ax.text(a, min(v + 0.08, 1.0), f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("RAGAS-style Scores of the Quiz-1 RAG")
    return save_fig(name, fig)


def plot_metric_distributions(df: pd.DataFrame, name="Q3_ragas_metric_distributions"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = [df[c].dropna().values for c in RAGAS_COLS]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(RAGAS_COLS) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in RAGAS_COLS])
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=8, alpha=0.4, color="#E45756")
    ax.set_title("Per-question Metric Distributions (box = quartiles, triangle = mean)")
    ax.set_ylabel("Score")
    return save_fig(name, fig)


def plot_retrieval_vs_generation(df: pd.DataFrame, name="Q3_hit_vs_generation"):
    g = df.groupby("hit")[["faithfulness", "answer_correctness", "answer_relevancy", "refused"]].mean()
    g.index = ["retrieval miss" if i == 0 else "retrieval hit" for i in g.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    g.T.plot.bar(ax=ax, color=["#E45756", "#54A24B"][:len(g)], rot=15)
    ax.set_title("Does a Retrieval Hit Lead to a Better Answer?")
    ax.set_ylabel("Mean score")
    return save_fig(name, fig)


def plot_metrics_by_doc(df: pd.DataFrame, name="Q3_correctness_by_document"):
    g = df.groupby("doc_id")[["answer_correctness", "faithfulness", "hit"]].mean()
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(g)), 4))
    g.plot.bar(ax=ax, width=0.8)
    ax.set_title("Quality per Source Document")
    ax.set_ylabel("Mean score")
    return save_fig(name, fig)


# ---- multi-turn (query rewriting) ablation ----------------------------------------------------
def evaluate_multiturn_text(llm, index, mt_items, k: int = 5, technique: str = "grounded_strict") -> pd.DataFrame:
    followups = [m["followup"] for m in mt_items]
    concat = [f"{m['history'][0]['content']} {m['history'][1]['content'][:80]} {m['followup']}" for m in mt_items]
    rewrote = [o.strip().split("\n")[0].strip("「」\"'") or m["followup"] for o, m in zip(
        llm.generate_batch([rewrite_messages(m["history"], m["followup"]) for m in mt_items], max_new_tokens=96), mt_items)]
    variants = {"no_rewrite (raw follow-up)": followups, "history_concat": concat,
                "llm_rewrite": rewrote, "oracle_standalone": [m["standalone"] for m in mt_items]}
    rows = []
    for name, qs in variants.items():
        hits = index.search_many(qs, k)
        answers, _ = Q1.answer_batch(llm, technique, followups, hits, histories=[m["history"] for m in mt_items])
        judge = judge_correctness(llm, [(m["standalone"], m["answer"], a) for m, a in zip(mt_items, answers)])
        for m, h, a, j in zip(mt_items, hits, answers, judge):
            fl = [Q1.chunk_hits_evidence(x["chunk"], m["doc_id"], m["ev_start"], m["ev_end"]) for x in h]
            first = next((i + 1 for i, f in enumerate(fl) if f), None)
            rows.append(dict(variant=name, kind=m.get("kind", "pronoun"), qid=m["id"], hit=float(any(fl)),
                             mrr=1 / first if first else 0.0, correctness=j / 5))
    df = pd.DataFrame(rows)
    if df["kind"].nunique() > 1:
        print("by follow-up kind (pronoun = same-topic follow-up, switch = topic change):")
        display_df = df.groupby(["kind", "variant"])[["hit", "mrr", "correctness"]].mean().round(3)
        try:
            from IPython.display import display
            display(display_df)
        except Exception:
            print(display_df)
    return df


def plot_multiturn_ablation(df: pd.DataFrame, name="Q3_multiturn_rewrite_ablation", title="Text RAG"):
    cols = [c for c in ("hit", "mrr", "correctness") if c in df]
    groups = [("All follow-ups", df)]
    if "kind" in df and df["kind"].nunique() > 1:
        groups += [(f"{k.capitalize()} follow-ups (n={df[df.kind == k].qid.nunique()})", df[df.kind == k])
                   for k in sorted(df["kind"].unique())]
    fig, axes = plt.subplots(1, len(groups), figsize=(9 * len(groups) / (1.6 if len(groups) > 1 else 1), 4.2),
                             sharey=True, squeeze=False)
    for ax, (ttl, d) in zip(axes[0], groups):
        d.groupby("variant")[cols].mean().plot.bar(ax=ax, rot=15, legend=ax is axes[0][0])
        ax.set_title(ttl if len(groups) > 1 else f"Multi-turn Query Handling Ablation ({title})")
        ax.set_ylabel("Mean score")
        ax.set_xlabel("")
    if len(groups) > 1:
        fig.suptitle(f"Multi-turn Query Handling Ablation ({title})")
    return save_fig(name, fig)


# =============================================================================
# Part B - VideoRAG evaluation
# =============================================================================
VQ_SPEECH = """你會看到一段影片片段的語音逐字稿。請出 1 個「只有看過這段影片才答得出來」的問題。
規則：問題必須獨立可理解，不可出現「這段」「片段」「逐字稿」；請改用不同的說法提問（不要逐字照抄逐字稿）；答案 1~2 句。
只輸出 JSON：{{"question": "...", "answer": "..."}}
逐字稿：{transcript}"""

VQ_VISUAL = """以上是同一段影片依時間順序抽出的畫面。請出 1 個關於「畫面中的操作或物件」的問題，
要求只有看過這段畫面才答得出來。
規則：問題必須獨立可理解，不可出現「這段」「畫面」「片段」等指代；答案 1~2 句，具體（例如按了哪個按鈕、畫面上的文字）。
只輸出 JSON：{"question": "...", "answer": "..."}"""

VF_PROMPT = """以下是影片的第一輪問答，以及該片段的描述。請接著出「第二輪追問」。
規則：followup 要使用代名詞或省略主題（例如「它」「那個步驟」），單看無法知道在問什麼；
standalone 是 followup 改寫成完整獨立的問題；answer 為標準答案（1~2 句）。
只輸出 JSON：{{"followup": "...", "standalone": "...", "answer": "..."}}
【畫面描述】{caption}
【逐字稿】{transcript}
【第一輪】Q: {q1}\nA: {a1}"""


VF_RETRY = """

注意：上一次的 followup 與 standalone 幾乎一樣，這是錯的。
followup 必須「短」且「依賴第一輪」才看得懂，例如「那它下一步要按哪裡？」「那個函式怎麼輸入？」「它的缺點呢？」；
不可出現影片主題或功能名稱。standalone 才寫出完整主題。"""

_VF_PRONOUN = re.compile(r"它|它們|其|這個|這些|那個|那些|那|該|此|這|上述|剛剛|剛才|前面|接下來|然後|下一步")


def _is_real_followup(f: str, sa: str) -> bool:
    """A follow-up must differ from its standalone form, be shorter, and refer back (pronoun / ellipsis)."""
    norm = lambda s: re.sub(r"[\s？?。，,！!「」\"']", "", s)
    return (norm(f) != norm(sa) and len(norm(f)) < len(norm(sa)) and bool(_VF_PRONOUN.search(f))
            and 3 <= len(f) <= 40)


def _round_robin(clips, n, seed):
    rng = random.Random(seed)
    by_vid = {}
    for c in clips:
        by_vid.setdefault(c["video_id"], []).append(c)
    for v in by_vid.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by_vid.values()):
        for k in list(by_vid):
            if by_vid[k] and len(out) < n:
                out.append(by_vid[k].pop())
    return out


def build_video_benchmark(vlm, clips, n_speech: int = 20, n_visual: int = 20, seed: int = SEED, force=False) -> list:
    """Silver-label benchmark. Speech questions come from transcripts; visual questions are generated by the VLM
    looking at the raw frames (NOT at the caption that is used for indexing -> less leakage).
    >>> Open Data/q2_video_qa.json and fix wrong Q/A by hand to turn it into a gold benchmark. <<<"""
    path = DATA_DIR / "q2_video_qa.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    sp = [c for c in clips if c["video_has_speech"] and len(c["transcript"]) >= 25]
    vis = [c for c in clips if not c["video_has_speech"]] or clips
    items = []
    for c in _round_robin(sp, n_speech, seed):
        o = extract_json(vlm.generate([{"role": "user", "content": VQ_SPEECH.format(transcript=c["transcript"])}],
                                      max_new_tokens=220))
        if isinstance(o, dict) and o.get("question") and o.get("answer"):
            items.append(dict(modality="speech", question=to_tw(str(o["question"])), answer=to_tw(str(o["answer"])), c=c))
    for c in _round_robin(vis, n_visual, seed + 1):
        imgs = [Image.open(p).convert("RGB") for p in c["frames"]]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": VQ_VISUAL}]
        o = extract_json(vlm.generate([{"role": "user", "content": content}], images=imgs, max_new_tokens=220))
        if isinstance(o, dict) and o.get("question") and o.get("answer"):
            items.append(dict(modality="visual", question=to_tw(str(o["question"])), answer=to_tw(str(o["answer"])), c=c))
    out = []
    for i, it in enumerate(items):
        c = it["c"]
        out.append(dict(id=f"vq{i:03d}", modality=it["modality"], question=it["question"], answer=it["answer"],
                        clip_id=c["clip_id"], video_id=c["video_id"], clip_idx=c["clip_idx"], start=c["start"],
                        end=c["end"], video_has_speech=c["video_has_speech"], label_source="silver"))
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"[VideoQA] {len(out)} questions ({sum(o['modality'] == 'speech' for o in out)} speech / "
          f"{sum(o['modality'] == 'visual' for o in out)} visual) -> {path}")
    return out


def build_video_multiturn(vlm, clips, items, n: int = 20, seed: int = SEED, force=False) -> list:
    path = DATA_DIR / "q2_video_multiturn.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    cmap = {c["clip_id"]: c for c in clips}
    out, n_bad = [], 0
    # iterate over ALL items (shuffled) until n valid ones: a follow-up identical to its standalone question
    # (the VLM often ignores the "use a pronoun" rule) makes the rewrite ablation meaningless -> retry / skip
    for it in random.Random(seed).sample(items, len(items)):
        if len(out) >= n:
            break
        c = cmap[it["clip_id"]]
        base = VF_PROMPT.format(caption=c["caption"], transcript=c["transcript"] or "（無語音）",
                                q1=it["question"], a1=it["answer"])
        for attempt in range(2):
            prompt = base if attempt == 0 else base + VF_RETRY
            o = extract_json(vlm.generate([{"role": "user", "content": prompt}], max_new_tokens=260))
            if not (isinstance(o, dict) and all(o.get(k) for k in ("followup", "standalone", "answer"))):
                continue
            f, sa, a = (to_tw(str(o[k])).strip() for k in ("followup", "standalone", "answer"))
            if _is_real_followup(f, sa):
                out.append(dict(id=f"vm{len(out):03d}", modality=it["modality"], clip_id=it["clip_id"],
                                video_id=it["video_id"], clip_idx=it["clip_idx"],
                                history=[{"role": "user", "content": it["question"]},
                                         {"role": "assistant", "content": it["answer"]}],
                                followup=f, standalone=sa, answer=a))
                break
            n_bad += 1
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(f"[VideoMT] {len(out)} two-turn items ({n_bad} rejected: follow-up was not a pronoun/ellipsis "
          f"question) -> {path}")
    return out


# ---- retrieval metrics ----------------------------------------------------------------------------
def _rank_rows(index, items, S, config):
    pos_of = {c["clip_id"]: i for i, c in enumerate(index.clips)}
    by_vid = {}
    for j, c in enumerate(index.clips):
        by_vid.setdefault(c["video_id"], []).append(j)
    rows = []
    for qi, it in enumerate(items):
        order = np.argsort(-S[qi])
        pos = np.empty_like(order)
        pos[order] = np.arange(len(order))
        g = pos_of[it["clip_id"]]
        gidx = index.clips[g]["clip_idx"]
        near = [j for j in by_vid[it["video_id"]] if abs(index.clips[j]["clip_idx"] - gidx) <= 1]
        rows.append(dict(config=config, qid=it["id"], modality=it.get("modality", "all"),
                         clip_rank=int(pos[g]) + 1, near_rank=int(min(pos[j] for j in near)) + 1,
                         video_rank=int(min(pos[j] for j in by_vid[it["video_id"]])) + 1))
    return rows


def eval_video_retrieval(index, items, alphas=tuple(np.round(np.linspace(0, 1, 11), 2))) -> pd.DataFrame:
    """Rank of the gold clip (exact / +-1 neighbour clip / same video) for text, visual and fusion(alpha)."""
    St, Sv = index.score_matrices([it["question"] for it in items])
    rows = _rank_rows(index, items, index.fuse(St, Sv, "text"), "text_only")
    rows += _rank_rows(index, items, index.fuse(St, Sv, "visual"), "visual_only")
    for a in alphas:
        rows += _rank_rows(index, items, index.fuse(St, Sv, "fusion", float(a)), f"fusion_a{a:.2f}")
    return pd.DataFrame(rows)


def summarize_video_retrieval(df: pd.DataFrame, ks=(1, 3, 5)) -> pd.DataFrame:
    out = []
    for (cfg, mod), g in list(df.groupby(["config", "modality"])) + [((c, "all"), g) for c, g in df.groupby("config")]:
        r = dict(config=cfg, modality=mod, n=len(g), mrr=float((1 / g["clip_rank"]).mean()))
        for k in ks:
            r[f"clip_hit@{k}"] = float((g["clip_rank"] <= k).mean())
            r[f"near_hit@{k}"] = float((g["near_rank"] <= k).mean())
            r[f"video_hit@{k}"] = float((g["video_rank"] <= k).mean())
        out.append(r)
    return pd.DataFrame(out)


def plot_video_retrieval_by_modality(summary: pd.DataFrame, alpha: float, name="Q3_video_retrieval_by_modality"):
    cfgs = ["text_only", "visual_only", f"fusion_a{alpha:.2f}"]
    labels = ["text only", "visual only", f"fusion (a={alpha:.2f})"]
    metrics = ["clip_hit@3", "near_hit@3", "video_hit@3", "mrr"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4), sharey=True)
    for ax, m in zip(axes, metrics):
        w = 0.25
        for i, mod in enumerate(["speech", "visual", "all"]):
            vals = [summary[(summary.config == c) & (summary.modality == mod)][m].mean() for c in cfgs]
            ax.bar(np.arange(len(cfgs)) + (i - 1) * w, vals, w, label=f"{mod} questions")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(m.replace("_", " "))
    axes[0].legend(fontsize=8)
    fig.suptitle("VideoRAG Retrieval: Modality Ablation, Stratified by Question Type")
    return save_fig(name, fig)


def plot_alpha_sweep(summary: pd.DataFrame, name="Q3_video_alpha_sweep"):
    s = summary[summary.config.str.startswith("fusion_a")].copy()
    s["alpha"] = s.config.str.replace("fusion_a", "").astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mod in ["speech", "visual", "all"]:
        g = s[s.modality == mod].sort_values("alpha")
        axes[0].plot(g.alpha, g["near_hit@3"], marker="o", label=mod)
        axes[1].plot(g.alpha, g["mrr"], marker="s", label=mod)
    axes[0].set_title("Near-clip Hit@3 vs fusion alpha (0 = visual, 1 = text)")
    axes[1].set_title("Clip MRR vs fusion alpha")
    for a in axes:
        a.set_xlabel("alpha (text weight)")
        a.legend()
    return save_fig(name, fig)


def best_alpha(summary: pd.DataFrame) -> float:
    s = summary[(summary.modality == "all") & summary.config.str.startswith("fusion_a")].copy()
    s["alpha"] = s.config.str.replace("fusion_a", "").astype(float)
    s["score"] = s["mrr"] + s["near_hit@3"]
    return float(s.sort_values("score", ascending=False).iloc[0]["alpha"])


# ---- generation metrics (VLM as judge on raw frames) -----------------------------------------------
def _evidence_content(hits, n_frames: int = 2):
    content, images = [{"type": "text", "text": "Evidence clips:\n"}], []
    for i, h in enumerate(hits, 1):
        c = h["clip"]
        content.append({"type": "text", "text": f"[V{i}] transcript: {c['transcript'] or '(no speech)'}\n"})
        for p in Q2._pick(c["frames"], n_frames):
            content.append({"type": "image"})
            images.append(Image.open(p).convert("RGB"))
    return content, images


def video_faithfulness(vlm, hits_list, answers, n_frames: int = 2):
    """Share of answer statements that are supported by the retrieved RAW frames + transcripts (not captions)."""
    out = []
    for hits, ans in zip(hits_list, answers):
        o = extract_json(vlm.generate(_decompose_prompt("", ans), max_new_tokens=300))
        st = [str(s) for s in _as_list(o, "statements")]
        if not st or is_refusal(ans):
            out.append(np.nan)
            continue
        content, imgs = _evidence_content(hits, n_frames)
        lst = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(st))
        content.append({"type": "text", "text": f"\nFor each statement decide whether the evidence (frames and transcripts) "
                                                f"directly supports it (1 = yes, 0 = no).\nStatements:\n{lst}\n"
                                                'Output ONLY JSON: {"verdicts": [1, 0, ...]}'})
        v = extract_json(vlm.generate([{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": content}],
                                      images=imgs, max_new_tokens=120))
        out.append(float(np.mean(_verdicts(v, len(st)))))
    return out


def eval_video_generation(vlm, index, items, gen_modes=("full", "text_only", "frames_only"), k: int = 3,
                          alpha: float = Q2.ALPHA, n_frames: int = 2) -> pd.DataFrame:
    """Same retrieval, different generation inputs -> isolates how much frames / captions / ASR matter."""
    hits_all = [index.retrieve(it["question"], k, "fusion", alpha) for it in items]
    cmap = {c["clip_id"]: c for c in index.clips}
    rows = []
    for mode in gen_modes:
        answers = []
        for it, h in zip(items, hits_all):
            msgs, imgs = Q2.build_video_messages(it["question"], h, None, mode, n_frames)
            answers.append(vlm.generate(msgs, images=imgs or None, max_new_tokens=300))
        judge = judge_correctness(vlm, [(it["question"], it["answer"], a) for it, a in zip(items, answers)])
        faith = video_faithfulness(vlm, hits_all, answers, n_frames)
        for it, h, a, j, f in zip(items, hits_all, answers, judge, faith):
            gidx = cmap[it["clip_id"]]["clip_idx"]
            near = [x for x in h if x["clip"]["video_id"] == it["video_id"] and abs(x["clip"]["clip_idx"] - gidx) <= 1]
            cited = {int(n) for n in re.findall(r"\[V(\d+)\]", a)}
            rows.append(dict(gen_mode=mode, qid=it["id"], modality=it["modality"], correctness=j / 5, faithfulness=f,
                             retrieval_hit=float(bool(near)),
                             ref_hit=float(any(x["rank"] in cited for x in near)),   # cites a correct clip
                             refused=float(is_refusal(a)), answer=a))
        print(f"[VideoGen] {mode} done")
    return pd.DataFrame(rows)


def plot_video_generation(df: pd.DataFrame, name="Q3_video_generation_ablation"):
    metrics = ["correctness", "faithfulness", "ref_hit", "retrieval_hit"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4), sharey=True)
    for ax, m in zip(axes, metrics):
        w = 0.27
        for i, mod in enumerate(["speech", "visual"]):
            g = df[df.modality == mod].groupby("gen_mode")[m].mean().reindex(["full", "text_only", "frames_only"])
            ax.bar(np.arange(len(g)) + (i - 0.5) * w, g.values, w, label=f"{mod} questions")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["frames+text", "text only", "frames only"], rotation=15)
        ax.set_title(m.replace("_", " "))
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("VideoRAG Generation Ablation (same retrieved clips, different inputs to the VLM)")
    return save_fig(name, fig)


def eval_video_multiturn(vlm, index, mt_items, k: int = 3, alpha: float = Q2.ALPHA) -> pd.DataFrame:
    cmap = {c["clip_id"]: c for c in index.clips}
    variants = {"no_rewrite": [m["followup"] for m in mt_items],
                "history_concat": [f"{m['history'][0]['content']} {m['history'][1]['content'][:80]} {m['followup']}"
                                   for m in mt_items],
                "vlm_rewrite": [Q2.rewrite_query(vlm, m["history"], m["followup"]) for m in mt_items],
                "oracle_standalone": [m["standalone"] for m in mt_items]}
    rows = []
    for name, qs in variants.items():
        St, Sv = index.score_matrices(qs)
        items = [dict(id=m["id"], modality=m["modality"], clip_id=m["clip_id"], video_id=m["video_id"]) for m in mt_items]
        r = pd.DataFrame(_rank_rows(index, items, index.fuse(St, Sv, "fusion", alpha), name))
        r["variant"] = name
        rows.append(r)
    df = pd.concat(rows)
    df["near_hit"] = (df["near_rank"] <= k).astype(float)
    df["mrr"] = 1 / df["clip_rank"]
    return df


def plot_video_multiturn(df: pd.DataFrame, name="Q3_video_multiturn_rewrite_ablation"):
    return plot_multiturn_ablation(df.rename(columns={"near_hit": "hit"})[["variant", "hit", "mrr"]], name, "VideoRAG")
