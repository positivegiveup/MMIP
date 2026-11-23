"""Quiz 4 - GraphRAG on one AI paper: "Attention Is All You Need" (arXiv:1706.03762).

Pipeline (GraphRAG, Microsoft-style, fully local):
  PDF -> text chunks -> LLM entity/relation extraction -> merge -> networkx graph
      -> Louvain communities -> LLM community summaries
  Query modes : naive (vector RAG baseline) | local (entity-centred sub-graph) | global (map-reduce over
                community summaries) | hybrid (local + naive chunks)
  Benchmark   : 25 hand-written Q/A (single-fact / multi-hop / global)  ->  LLM-judge, F1, semantic sim, latency
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from Codes.common import *  # noqa: F401,F403

set_style()

PAPER_URL = "https://arxiv.org/pdf/1706.03762"
PAPER_PATH = DATA_DIR / "q4_paper_attention_is_all_you_need.pdf"

# =============================================================================
# 1. Paper -> text -> chunks
# =============================================================================


def download_paper(force: bool = False) -> Path:
    if PAPER_PATH.exists() and not force:
        return PAPER_PATH
    import requests
    r = requests.get(PAPER_URL, headers={"User-Agent": "RAG-Course-Homework/1.0"}, timeout=60)
    r.raise_for_status()
    PAPER_PATH.write_bytes(r.content)
    print(f"[downloaded] {PAPER_PATH} ({len(r.content) / 1e3:.0f} KB)")
    return PAPER_PATH


def extract_paper_text(path: Path = PAPER_PATH) -> str:
    """Block-wise extraction (one block ~ one paragraph), drops everything after 'References'."""
    paras = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        for page in doc:
            for b in page.get_text("blocks", sort=True):
                t = b[4] if len(b) > 4 else ""
                paras.append(t)
    except ImportError:
        from pypdf import PdfReader
        for page in PdfReader(str(path)).pages:
            paras += (page.extract_text() or "").split("\n\n")
    out = []
    for t in paras:
        t = re.sub(r"-\n(?=[a-z])", "", t)          # de-hyphenate
        t = re.sub(r"\s*\n\s*", " ", t).strip()
        t = t.replace("ﬁ", "fi").replace("ﬂ", "fl")
        if re.fullmatch(r"(?:\d+\s*)?references", t, flags=re.I):   # heading is short (<25 chars) -> check BEFORE the length filter
            break
        if len(t) >= 25:
            out.append(t)
    return "\n\n".join(out)


def chunk_paper(text: str, chunk_size: int = 1200, overlap: int = 150) -> list:
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > chunk_size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p if overlap else p
        else:
            cur = (cur + "\n" + p) if cur else p
        while len(cur) > chunk_size * 1.6:          # extremely long paragraph
            chunks.append(cur[:chunk_size])
            cur = cur[chunk_size - overlap:]
    if cur.strip():
        chunks.append(cur)
    return [dict(chunk_id=i, text=c.strip()) for i, c in enumerate(chunks)]


# =============================================================================
# 2. Benchmark (hand-written; verify against the paper before reporting!)
# =============================================================================
BENCHMARK = [
    # ---- single-fact -----------------------------------------------------------------------------
    ("single", "How many identical layers does the Transformer encoder stack contain?",
     "The encoder is a stack of N = 6 identical layers."),
    ("single", "What is the model dimension d_model used by the base Transformer?",
     "d_model = 512; all sub-layers and embedding layers produce outputs of dimension 512."),
    ("single", "How many attention heads does the base Transformer use, and what are d_k and d_v per head?",
     "h = 8 parallel attention heads, with d_k = d_v = 64."),
    ("single", "What is the inner-layer dimensionality of the position-wise feed-forward network?",
     "The inner layer has dimensionality d_ff = 2048 (input and output are 512)."),
    ("single", "Why are the dot products scaled by 1/sqrt(d_k) in scaled dot-product attention?",
     "For large d_k the dot products grow large in magnitude, pushing the softmax into regions with extremely small "
     "gradients; scaling by 1/sqrt(d_k) counteracts this."),
    ("single", "Which optimizer and hyperparameters were used to train the models?",
     "The Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10^-9."),
    ("single", "How many warmup steps are used in the learning rate schedule?",
     "warmup_steps = 4000: the learning rate increases linearly for the first 4000 steps and then decreases "
     "proportionally to the inverse square root of the step number."),
    ("single", "What kind of positional encoding does the Transformer use?",
     "Sinusoidal positional encodings: sine and cosine functions of different frequencies."),
    ("single", "What BLEU score did the big Transformer reach on the WMT 2014 English-to-German task?",
     "28.4 BLEU, improving over the previous best results (including ensembles) by over 2 BLEU."),
    ("single", "What hardware was used to train the models?",
     "One machine with 8 NVIDIA P100 GPUs."),
    ("single", "How long did it take to train the base models and for how many steps?",
     "The base models were trained for 100,000 steps, about 12 hours (roughly 0.4 seconds per step)."),
    ("single", "What dropout rate is used by the base model?",
     "P_drop = 0.1, applied to the output of each sub-layer and to the sums of embeddings and positional encodings."),
    ("single", "What label smoothing value was employed during training?",
     "Label smoothing with epsilon_ls = 0.1, which hurts perplexity but improves accuracy and BLEU."),
    ("single", "Which dataset was used for the English constituency parsing experiments?",
     "The Wall Street Journal (WSJ) portion of the Penn Treebank, about 40K training sentences."),
    ("single", "What training data and vocabulary were used for WMT 2014 English-German?",
     "About 4.5 million sentence pairs encoded with byte-pair encoding, with a shared source-target vocabulary of "
     "about 37,000 tokens."),
    # ---- multi-hop -------------------------------------------------------------------------------
    ("multi", "How is the per-head dimension d_k of the base model determined from other hyperparameters, and what is its value?",
     "d_k = d_v = d_model / h = 512 / 8 = 64."),
    ("multi", "In encoder-decoder attention, where do the queries, keys and values come from?",
     "The queries come from the previous decoder layer, while the keys and values come from the output of the encoder."),
    ("multi", "How does the decoder preserve the auto-regressive property?",
     "Decoder self-attention is masked: softmax inputs corresponding to illegal (future) connections are set to -infinity, "
     "and the output embeddings are offset by one position, so predictions depend only on known earlier outputs."),
    ("multi", "How do the residual connections constrain the output dimension of the sub-layers?",
     "Each sub-layer output is LayerNorm(x + Sublayer(x)); to make the residual additions possible, all sub-layers and "
     "embedding layers output vectors of dimension d_model = 512."),
    ("multi", "How do self-attention layers compare with recurrent layers in sequential operations and maximum path length?",
     "Self-attention needs O(1) sequential operations and has O(1) maximum path length, whereas recurrent layers need "
     "O(n) sequential operations and have O(n) maximum path length."),
    ("multi", "How are embedding weights shared in the model and what scaling is applied?",
     "The same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation, "
     "and the embedding weights are multiplied by sqrt(d_model)."),
    # ---- global ----------------------------------------------------------------------------------
    ("global", "What are the main components of the Transformer architecture described in the paper?",
     "An encoder-decoder architecture with stacks of 6 layers: encoder layers have multi-head self-attention and a "
     "position-wise feed-forward network; decoder layers add masked self-attention and encoder-decoder attention; "
     "residual connections with layer normalization, embeddings with positional encodings, and a final linear + softmax."),
    ("global", "What advantages does the paper claim for self-attention over recurrent and convolutional layers?",
     "Lower total computational complexity per layer (when n < d), more parallelizable computation, shorter path "
     "lengths between long-range dependencies, and potentially more interpretable models."),
    ("global", "Which regularization techniques are used when training the Transformer?",
     "Residual dropout (applied to sub-layer outputs and embedding sums) and label smoothing."),
    ("global", "Summarize the main contribution of the paper and its key experimental results.",
     "It proposes the Transformer, a sequence transduction model based solely on attention that dispenses with recurrence "
     "and convolutions; it achieves state-of-the-art translation quality on WMT 2014 (28.4 BLEU English-German) with "
     "much less training cost and generalizes to English constituency parsing."),
]


def load_benchmark() -> list:
    path = DATA_DIR / "q4_benchmark.json"
    if not path.exists():
        data = [dict(id=f"q{i:02d}", category=c, question=q, answer=a) for i, (c, q, a) in enumerate(BENCHMARK)]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    return json.loads(path.read_text("utf-8"))


# =============================================================================
# 3. Graph construction
# =============================================================================
ENTITY_TYPES = ["MODEL", "COMPONENT", "HYPERPARAMETER", "DATASET", "METRIC", "METHOD", "TASK", "HARDWARE",
                "CONCEPT", "RESULT"]
EXTRACT_PROMPT = """You are building a knowledge graph from a machine-learning paper.
Extract entities and relations from the TEXT.
Entity types: {types}
Return ONLY JSON:
{{"entities": [{{"name": "", "type": "", "description": "one sentence"}}],
  "relations": [{{"source": "", "target": "", "relation": "short verb phrase", "description": "one sentence"}}]}}
Rules: use canonical short names (e.g. "multi-head attention", "WMT 2014 English-German"); put numeric values
(e.g. "d_model = 512", "28.4 BLEU") inside the descriptions; relation endpoints must be listed in entities;
at most 15 entities and 20 relations.

TEXT:
{text}"""


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def extract_graph_elements(llm, chunks, batch_size: int = 6, force: bool = False):
    path = DATA_DIR / "q4_extraction.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    prompts = [[{"role": "user", "content": EXTRACT_PROMPT.format(types=", ".join(ENTITY_TYPES), text=c["text"])}]
               for c in chunks]
    outs = llm.generate_batch(prompts, max_new_tokens=1400, batch_size=batch_size)
    res = []
    for c, o in zip(chunks, outs):
        obj = extract_json(o)
        if not isinstance(obj, dict):
            obj = {}
        res.append(dict(chunk_id=c["chunk_id"], entities=obj.get("entities", []) or [],
                        relations=obj.get("relations", []) or []))
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    n_e = sum(len(r["entities"]) for r in res)
    n_r = sum(len(r["relations"]) for r in res)
    print(f"[extract] {n_e} raw entities / {n_r} raw relations from {len(chunks)} chunks")
    return res


def build_graph(extractions, embedder=None, merge_thr: float = 0.92) -> nx.Graph:
    """Aggregate extractions into a graph; optionally merge near-duplicate entity names (same type)."""
    nodes = {}
    for ex in extractions:
        for e in ex["entities"]:
            if not isinstance(e, dict) or not e.get("name"):
                continue
            k = norm(e["name"])
            n = nodes.setdefault(k, dict(name=str(e["name"]).strip(), types=Counter(), descs=[], chunks=set(), mentions=0))
            n["types"][str(e.get("type", "CONCEPT")).upper()] += 1
            d = str(e.get("description", "")).strip()
            if d and d not in n["descs"]:
                n["descs"].append(d)
            n["chunks"].add(ex["chunk_id"])
            n["mentions"] += 1
    alias = {k: k for k in nodes}
    if embedder is not None and len(nodes) > 1 and merge_thr < 1:
        keys = list(nodes)
        E = embedder.encode([nodes[k]["name"] for k in keys])
        S = E @ E.T
        ttype = [nodes[k]["types"].most_common(1)[0][0] for k in keys]
        log = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if S[i, j] >= merge_thr and ttype[i] == ttype[j]:
                    a, b = keys[i], keys[j]
                    ra, rb = alias[a], alias[b]
                    if ra != rb:
                        keep = ra if nodes[ra]["mentions"] >= nodes[rb]["mentions"] else rb
                        drop = rb if keep == ra else ra
                        for kk, v in alias.items():
                            if v == drop:
                                alias[kk] = keep
                        log.append((nodes[keep]["name"], nodes[drop]["name"], float(S[i, j])))
        (DATA_DIR / "q4_merge_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
        print(f"[merge] merged {len(log)} near-duplicate entities (see Data/q4_merge_log.json)")
    G = nx.Graph()
    for k, n in nodes.items():
        r = alias[k]
        if r not in G:
            G.add_node(r, name=nodes[r]["name"], types=Counter(), descs=[], chunks=set(), mentions=0)
        g = G.nodes[r]
        g["types"].update(n["types"])
        g["mentions"] += n["mentions"]
        g["chunks"] |= n["chunks"]
        for d in n["descs"]:
            if d not in g["descs"]:
                g["descs"].append(d)
    for ex in extractions:
        for rel in ex["relations"]:
            if not isinstance(rel, dict):
                continue
            s, t = alias.get(norm(rel.get("source", ""))), alias.get(norm(rel.get("target", "")))
            if not s or not t or s == t:
                continue
            if G.has_edge(s, t):
                G[s][t]["weight"] += 1
            else:
                G.add_edge(s, t, weight=1, relations=[])
            G[s][t]["relations"].append(dict(src=G.nodes[s]["name"], tgt=G.nodes[t]["name"],
                                             relation=str(rel.get("relation", "related to")),
                                             description=str(rel.get("description", "")), chunk=ex["chunk_id"]))
    for n in G.nodes:
        G.nodes[n]["type"] = G.nodes[n]["types"].most_common(1)[0][0]
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0 and G.nodes[n]["mentions"] < 2])
    print(f"[graph] {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = SEED) -> list:
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted((sorted(c) for c in comms), key=len, reverse=True)
    print(f"[community] {len(comms)} communities; sizes = {[len(c) for c in comms]}")
    return comms


def summarize_communities(llm, G, comms, force: bool = False) -> list:
    path = DATA_DIR / "q4_community_summaries.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    prompts = []
    for cid, c in enumerate(comms):
        ents = "\n".join(f"- {G.nodes[n]['name']} ({G.nodes[n]['type']}): {' '.join(G.nodes[n]['descs'][:2])}"[:300]
                         for n in sorted(c, key=lambda x: -G.degree(x))[:25])
        rels = []
        for u, v, d in G.subgraph(c).edges(data=True):
            r = d["relations"][0]
            rels.append(f"- {r['src']} --{r['relation']}--> {r['tgt']}: {r['description']}"[:240])
        prompts.append([{"role": "user", "content":
                         "Write a coherent report (<= 140 words) describing what this group of entities from the "
                         "'Attention Is All You Need' paper represents and how they relate. Keep concrete numbers.\n\n"
                         f"ENTITIES:\n{ents}\n\nRELATIONS:\n" + "\n".join(rels[:35])}])
    outs = llm.generate_batch(prompts, max_new_tokens=300, batch_size=4)
    summ = [dict(community=cid, size=len(c), nodes=c, summary=o.strip()) for cid, (c, o) in enumerate(zip(comms, outs))]
    path.write_text(json.dumps(summ, ensure_ascii=False, indent=1), "utf-8")
    return summ


# ---- persistence ("model" artefacts, all prefixed q4_) ---------------------------------------------
def save_graph(G: nx.Graph, comms, node_emb, chunk_emb, chunks):
    nodes = [dict(id=n, name=d["name"], type=d["type"], descs=d["descs"], chunks=sorted(d["chunks"]),
                  mentions=d["mentions"]) for n, d in G.nodes(data=True)]
    edges = [dict(source=u, target=v, weight=d["weight"], relations=d["relations"]) for u, v, d in G.edges(data=True)]
    (OUT_DIR / "q4_graph.json").write_text(json.dumps(dict(nodes=nodes, edges=edges, communities=comms),
                                                      ensure_ascii=False), "utf-8")
    H = nx.Graph()   # GraphML copy (flat attributes only) for Gephi / Neo4j import
    for n, d in G.nodes(data=True):
        H.add_node(n, name=d["name"], type=d["type"], description=" | ".join(d["descs"])[:800], mentions=d["mentions"])
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, weight=d["weight"], relation="; ".join(r["relation"] for r in d["relations"])[:300])
    nx.write_graphml(H, OUT_DIR / "q4_graph.graphml")
    np.savez_compressed(OUT_DIR / "q4_embeddings.npz", node_emb=node_emb, chunk_emb=chunk_emb)
    (OUT_DIR / "q4_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), "utf-8")
    print(f"[saved] q4_graph.json / q4_graph.graphml / q4_embeddings.npz / q4_chunks.json in {OUT_DIR}")


# =============================================================================
# 4. GraphRAG engine
# =============================================================================
ANS_SYS = ("Answer the question using ONLY the provided context. Be precise: keep exact names and numbers. "
           "Answer in at most 4 sentences. If the context does not contain the answer, say so.")


class GraphRAG:
    def __init__(self, llm, embedder, chunks, chunk_emb, G, comm_summaries):
        self.llm, self.emb, self.chunks, self.chunk_emb = llm, embedder, chunks, chunk_emb
        self.G, self.summ = G, comm_summaries
        self.nodes = list(G.nodes)
        self.node_emb = embedder.encode([f"{G.nodes[n]['name']}: {(G.nodes[n]['descs'] or [''])[0]}" for n in self.nodes])
        self.node2comm = {n: s["community"] for s in comm_summaries for n in s["nodes"]}

    def _llm_answer(self, question, context, max_new_tokens=300):
        return self.llm.generate([{"role": "system", "content": ANS_SYS},
                                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
                                 max_new_tokens=max_new_tokens)

    # ---- baseline --------------------------------------------------------------------------------
    def naive(self, q, k: int = 4):
        t0 = time.perf_counter()
        s = self.chunk_emb @ self.emb.encode([q])[0]
        top = np.argsort(-s)[:k]
        ctx = "\n\n".join(f"[chunk {i}] {self.chunks[i]['text']}" for i in top)
        return dict(answer=self._llm_answer(q, ctx), context=ctx, seeds=[], nodes=[], latency=time.perf_counter() - t0)

    # ---- local: entity-centred sub-graph ------------------------------------------------------------
    def local_context(self, q, n_seed: int = 6, n_neigh: int = 10, n_chunks: int = 3, n_comm: int = 2):
        qe = self.emb.encode([q])[0]
        top = np.argsort(-(self.node_emb @ qe))[:n_seed]
        seeds = [self.nodes[i] for i in top]
        neigh = Counter()
        for s in seeds:
            for nb in self.G.neighbors(s):
                if nb not in seeds:
                    neigh[nb] += self.G[s][nb]["weight"]
        keep = seeds + [n for n, _ in neigh.most_common(n_neigh)]
        ent_lines = [f"- {self.G.nodes[n]['name']} ({self.G.nodes[n]['type']}): {' '.join(self.G.nodes[n]['descs'][:2])}"
                     for n in keep]
        rel_lines = []
        for u, v, d in self.G.subgraph(keep).edges(data=True):
            for r in d["relations"][:2]:
                rel_lines.append(f"- {r['src']} --{r['relation']}--> {r['tgt']}: {r['description']}")
        cc = Counter(c for s in seeds for c in self.G.nodes[s]["chunks"])
        chunk_ids = [c for c, _ in cc.most_common(n_chunks)]
        comm_ids = [c for c, _ in Counter(self.node2comm[s] for s in seeds if s in self.node2comm).most_common(n_comm)]
        ctx = ("## Entities\n" + "\n".join(ent_lines) + "\n\n## Relations\n" + "\n".join(rel_lines[:45]) +
               "\n\n## Source text\n" + "\n\n".join(f"[chunk {c}] {self.chunks[c]['text']}" for c in chunk_ids) +
               "\n\n## Community reports\n" + "\n\n".join(self.summ[c]["summary"] for c in comm_ids))
        return ctx, seeds, keep

    def local(self, q, **kw):
        t0 = time.perf_counter()
        ctx, seeds, keep = self.local_context(q, **kw)
        return dict(answer=self._llm_answer(q, ctx), context=ctx, seeds=seeds, nodes=keep, latency=time.perf_counter() - t0)

    # ---- global: map-reduce over community reports ------------------------------------------------------
    def global_(self, q, top_points: int = 8):
        t0 = time.perf_counter()
        prompts = [[{"role": "user", "content":
                     f"Community report:\n{s['summary']}\n\nQuestion: {q}\n\nExtract the points from the report that help "
                     'answer the question. Output ONLY JSON: {"points": "...", "score": 0-100 (helpfulness)}'}]
                   for s in self.summ]
        outs = batch_json(self.llm, prompts, max_new_tokens=250, batch_size=8)
        pts = []
        for o in outs:
            if isinstance(o, dict) and o.get("points"):
                try:
                    sc = float(o.get("score", 0))
                except (TypeError, ValueError):
                    sc = 0.0
                if sc > 15:
                    pts.append((sc, str(o["points"])))
        pts.sort(reverse=True)
        ctx = "\n".join(f"- (helpfulness {s:.0f}) {p}" for s, p in pts[:top_points]) or "(no relevant community found)"
        return dict(answer=self._llm_answer(q, ctx, 350), context=ctx, seeds=[], nodes=[], latency=time.perf_counter() - t0)

    def hybrid(self, q, k: int = 3):
        t0 = time.perf_counter()
        ctx, seeds, keep = self.local_context(q)
        s = self.chunk_emb @ self.emb.encode([q])[0]
        extra = "\n\n".join(f"[chunk {i}] {self.chunks[i]['text']}" for i in np.argsort(-s)[:k])
        ctx = ctx + "\n\n## Top vector-retrieved chunks\n" + extra
        return dict(answer=self._llm_answer(q, ctx), context=ctx, seeds=seeds, nodes=keep, latency=time.perf_counter() - t0)

    def answer(self, method: str, q: str):
        return {"naive": self.naive, "local": self.local, "global": self.global_, "hybrid": self.hybrid}[method](q)


# =============================================================================
# 5. Benchmark evaluation
# =============================================================================
METHODS = ["naive", "local", "global", "hybrid"]


def evaluate_graphrag(rag: GraphRAG, bench, methods=METHODS) -> pd.DataFrame:
    rows, preds = [], {}
    for m in methods:
        preds[m] = [rag.answer(m, b["question"]) for b in bench]
        print(f"[bench] {m} done (avg {np.mean([p['latency'] for p in preds[m]]):.1f}s)")
    for m in methods:
        judge = judge_correctness(rag.llm, [(b["question"], b["answer"], p["answer"]) for b, p in zip(bench, preds[m])])
        ea = rag.emb.encode([p["answer"] for p in preds[m]])
        eg = rag.emb.encode([b["answer"] for b in bench])
        sem = (ea * eg).sum(1)
        for b, p, j, s in zip(bench, preds[m], judge, sem):
            rows.append(dict(method=m, qid=b["id"], category=b["category"], judge=j, correct=float(j >= 4),
                             token_f1=token_f1(p["answer"], b["answer"]), sem_sim=float(s), latency=p["latency"],
                             ctx_chars=len(p["context"]), answer=p["answer"]))
    return pd.DataFrame(rows)


# =============================================================================
# 6. Figures
# =============================================================================
def plot_graph_overview(G, comms, name="Q4_graph_overview", n_labels: int = 35):
    node2c = {n: i for i, c in enumerate(comms) for n in c}
    pos = nx.spring_layout(G, k=0.6 / np.sqrt(max(1, G.number_of_nodes())) * 3, seed=SEED, iterations=200)
    deg = dict(G.degree())
    fig, ax = plt.subplots(figsize=(14, 11))
    nx.draw_networkx_edges(G, pos, alpha=0.18, width=0.6, ax=ax)
    cmap = plt.get_cmap("tab20")
    nx.draw_networkx_nodes(G, pos, node_size=[40 + 35 * deg[n] for n in G], node_color=[cmap(node2c.get(n, 0) % 20) for n in G],
                           alpha=0.9, linewidths=0.4, edgecolors="k", ax=ax)
    top = sorted(G, key=lambda n: -deg[n])[:n_labels]
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["name"] for n in top}, font_size=8, ax=ax)
    ax.set_title(f"Knowledge Graph of 'Attention Is All You Need' ({G.number_of_nodes()} entities, "
                 f"{G.number_of_edges()} relations, {len(comms)} communities)")
    ax.axis("off")
    return save_fig(name, fig)


def plot_graph_statistics(G, comms, name="Q4_graph_statistics"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    deg = sorted((d for _, d in G.degree()), reverse=True)
    axes[0].loglog(range(1, len(deg) + 1), [max(d, 0.8) for d in deg], marker="o", ms=3)
    axes[0].set_title("Degree Distribution (rank-degree, log-log)")
    axes[0].set_xlabel("Node rank")
    axes[0].set_ylabel("Degree")
    types = Counter(G.nodes[n]["type"] for n in G)
    axes[1].barh(list(types.keys()), list(types.values()), color="#4C78A8")
    axes[1].set_title("Entity Types")
    axes[2].bar([f"C{i}" for i in range(len(comms))], [len(c) for c in comms], color="#F58518")
    axes[2].set_title("Community Sizes (Louvain)")
    return save_fig(name, fig)


def plot_subgraph(G, nodes, seeds, title, name="Q4_local_search_subgraph"):
    H = G.subgraph(nodes)
    pos = nx.spring_layout(H, seed=SEED, k=0.9)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    nx.draw_networkx_edges(H, pos, alpha=0.4, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=["#E45756" if n in seeds else "#9ecae9" for n in H], node_size=420,
                           edgecolors="k", linewidths=0.5, ax=ax)
    nx.draw_networkx_labels(H, pos, labels={n: H.nodes[n]["name"] for n in H}, font_size=8, ax=ax)
    el = {(u, v): d["relations"][0]["relation"][:22] for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=el, font_size=6, ax=ax)
    ax.set_title(title + "  (red = seed entities matched to the query)")
    ax.axis("off")
    return save_fig(name, fig)


def plot_benchmark(df, name_overall="Q4_benchmark_overall", name_cat="Q4_benchmark_by_category"):
    order = [m for m in METHODS if m in df.method.unique()]
    g = df.groupby("method").agg(judge=("judge", "mean"), correct=("correct", "mean"), f1=("token_f1", "mean"),
                                 sem=("sem_sim", "mean"), lat=("latency", "mean")).reindex(order)
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, (col, ttl, c) in zip(axes, [("judge", "LLM-judge score (1-5) ↑", "#4C78A8"), ("correct", "Accuracy (judge>=4) ↑", "#54A24B"),
                                        ("f1", "Token F1 ↑", "#F58518"), ("sem", "Semantic similarity ↑", "#B279A2"),
                                        ("lat", "Latency / question (s) ↓", "#BAB0AC")]):
        ax.bar(order, g[col].values, color=c)
        ax.set_title(ttl)
        for i, v in enumerate(g[col].values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("GraphRAG Benchmark - Overall (25 questions)")
    save_fig(name_overall, fig)
    cats = ["single", "multi", "global"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, col, ttl in zip(axes, ["judge", "correct"], ["LLM-judge score by question type", "Accuracy by question type"]):
        p = df.pivot_table(index="category", columns="method", values=col, aggfunc="mean").reindex(cats)[order]
        p.plot.bar(ax=ax, rot=0)
        ax.legend(loc="lower right", ncol=2, fontsize=8)
        ax.set_title(ttl)
    fig.suptitle("GraphRAG vs Naive RAG by Question Type (single-fact / multi-hop / global)")
    save_fig(name_cat, fig)
    return g
