"""Quiz 1 - Multi-turn RAG chatbot (+ advanced: chunk/overlap/top-k search, prompting, UI).

Topic   : Artificial Intelligence / Machine Learning (Traditional-Chinese Wikipedia)
Pipeline: Wikipedia download -> sentence-aware chunking -> bge-m3 embeddings -> cosine retrieval
          -> query rewriting (multi-turn) -> prompted Qwen2.5-7B-Instruct -> answer + references
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Codes.common import *  # noqa: F401,F403  (paths, ChatLLM, Embedder, save_fig, ...)

set_style()

# =============================================================================
# 1. Corpus download (Wikipedia, zh)
# =============================================================================
WIKI_API = "https://zh.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Course-Homework/1.0 (educational use)"}
WIKI_TITLES = [
    "人工智慧", "機器學習", "深度學習", "人工神經網路", "自然語言處理", "大型語言模型",
    "卷積神經網路", "循環神經網路", "強化學習", "監督式學習", "非監督式學習", "生成對抗網路",
    "電腦視覺", "語音辨識", "推薦系統", "資料探勘", "決策樹", "支持向量機", "聚類分析",
    "自編碼器", "知識圖譜", "圖靈測試", "ChatGPT", "反向傳播算法", "過適", "梯度下降法",
    "特徵工程", "遷移學習", "貝氏定理", "邏輯迴歸", "線性迴歸", "隨機森林",
]
_DROP_SECTIONS = {"參見", "參考文獻", "參考資料", "外部連結", "註釋", "注釋", "延伸閱讀",
                  "相關條目", "腳註", "參考", "引用", "來源", "文獻"}


def clean_wiki_text(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.+?)\s*\1\s*$", line.strip())
        if m:
            skip = m.group(2).strip() in _DROP_SECTIONS
            continue
        if skip:
            continue
        line = line.replace("​", "").strip()
        if len(line) >= 6:
            out.append(line)
    return "\n".join(out)


def download_corpus(titles=WIKI_TITLES, force: bool = False, min_chars: int = 800) -> list:
    """Download once, then cache to Data/q1_corpus.json."""
    path = DATA_DIR / "q1_corpus.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    import requests
    params = dict(action="query", format="json", prop="extracts", explaintext=1, maxlag=5,
                  exsectionformat="wiki", redirects=1, converttitles=1, variant="zh-tw")

    def fetch(t, tries=5):
        # Wikimedia throttles bursts of anonymous requests (HTTP 429 / maxlag); the first run only got the
        # first 10 titles because of this -> pause between calls and back off on errors.
        for a in range(tries):
            try:
                r = requests.get(WIKI_API, headers=HEADERS, timeout=30, params={**params, "titles": t})
                if r.status_code == 429:
                    raise RuntimeError("throttled (HTTP 429)")
                r.raise_for_status()
                js = r.json()
                if "error" in js:
                    raise RuntimeError(f"API error: {js['error'].get('code')}")
                return next(iter(js["query"]["pages"].values()))
            except Exception as e:
                if a == tries - 1:
                    raise
                wait = 3 * 2 ** a
                print(f"[retry] {t}: {e} -> wait {wait}s")
                time.sleep(wait)

    corpus, seen = [], set()
    for t in dict.fromkeys(titles):
        time.sleep(1.0)
        try:
            page = fetch(t)
            if "missing" in page:
                print(f"[skip] {t}: page not found")
                continue
            title = to_tw(page.get("title", t))
            text = clean_wiki_text(to_tw(page.get("extract", "")))
        except Exception as e:  # network / parse errors
            print(f"[skip] {t}: {e}")
            continue
        if len(text) < min_chars or title in seen:
            print(f"[skip] {t}: {'duplicate of ' + title if title in seen else f'too short ({len(text)} chars)'}")
            continue
        seen.add(title)
        corpus.append(dict(doc_id=f"D{len(corpus):02d}", title=title, text=text))
        print(f"[ok] {corpus[-1]['doc_id']} {title}: {len(text)} chars")
    if not corpus:
        raise RuntimeError("Wikipedia download failed for every title - check your network.")
    path.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), "utf-8")
    return corpus


def plot_corpus_stats(corpus):
    ids = [d["doc_id"] for d in corpus]
    lens = [len(d["text"]) for d in corpus]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(ids)), 4))
    ax.bar(ids, lens, color="#4C78A8")
    ax.set_title("Corpus Size per Document (characters)")
    ax.set_xlabel("Document ID")
    ax.set_ylabel("Characters")
    plt.setp(ax.get_xticklabels(), rotation=70)
    return save_fig("Q1_corpus_chars_per_document", fig)


# =============================================================================
# 2. Sentence-aware chunking (chunk_size / overlap are in characters)
# =============================================================================
_SENT = re.compile(r"[^。！？!?\n]*[。！？!?]+[」』）)”]*|[^。！？!?\n]+")


def sentence_spans(text: str, max_len: int):
    spans = []
    for m in _SENT.finditer(text):
        s, e = m.start(), m.end()
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        if e <= s:
            continue
        while e - s > max_len:  # hard-split very long "sentences"
            spans.append((s, s + max_len))
            s += max_len
        spans.append((s, e))
    return spans


def chunk_spans(text: str, chunk_size: int, overlap: int):
    """Greedy sentence packing. Consecutive chunks share <= `overlap` characters of whole sentences."""
    spans = sentence_spans(text, chunk_size)
    out, i = [], 0
    while i < len(spans):
        j = i
        while j < len(spans) and spans[j][1] - spans[i][0] <= chunk_size:
            j += 1
        if j == i:
            j = i + 1
        out.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        k = j
        while k - 1 > i and overlap > 0 and spans[j - 1][1] - spans[k - 1][0] <= overlap:
            k -= 1
        i = k
    return out


def build_chunks(corpus, chunk_size: int = 300, overlap: int = 50) -> list:
    chunks = []
    for d in corpus:
        for s, e in chunk_spans(d["text"], chunk_size, overlap):
            chunks.append(dict(chunk_id=len(chunks), doc_id=d["doc_id"], title=d["title"],
                               start=s, end=e, text=d["text"][s:e]))
    return chunks


def chunk_hits_evidence(chunk: dict, doc_id: str, ev_s: int, ev_e: int, thr: float = 0.5) -> bool:
    """A chunk 'hits' the gold evidence if it is in the same document and covers >= thr of it.
    Char-offset based, hence comparable across different chunk sizes."""
    if chunk["doc_id"] != doc_id:
        return False
    ov = min(chunk["end"], ev_e) - max(chunk["start"], ev_s)
    return ov >= thr * max(1, ev_e - ev_s)


def plot_chunk_stats(chunks, title_suffix=""):
    lens = [len(c["text"]) for c in chunks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=30, color="#F58518", edgecolor="white")
    ax.axvline(np.mean(lens), color="k", ls="--", label=f"mean={np.mean(lens):.0f}")
    ax.set_title(f"Chunk Length Distribution {title_suffix}(N={len(lens)})")
    ax.set_xlabel("Characters per chunk")
    ax.set_ylabel("Count")
    ax.legend()
    return save_fig("Q1_chunk_length_distribution", fig)


# =============================================================================
# 3. Vector index
# =============================================================================
class VectorIndex:
    def __init__(self, chunks, emb, embedder):
        self.chunks, self.emb, self.embedder = chunks, emb, embedder

    @classmethod
    def build(cls, chunks, embedder, show=False):
        texts = [f"{c['title']}\n{c['text']}" for c in chunks]  # title header = cheap context
        return cls(chunks, embedder.encode(texts, show=show), embedder)

    def search_many(self, queries, k: int = 5):
        Q = self.embedder.encode(list(queries))
        S = Q @ self.emb.T
        k = min(k, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        out = []
        for i in range(len(queries)):
            order = idx[i][np.argsort(-S[i, idx[i]])]
            out.append([dict(rank=r + 1, score=float(S[i, j]), idx=int(j), chunk=self.chunks[j])
                        for r, j in enumerate(order)])
        return out

    def search(self, query: str, k: int = 5):
        return self.search_many([query], k)[0]

    def save(self, name: str = "q1_index"):
        np.savez_compressed(OUT_DIR / f"{name}.npz", emb=self.emb)
        (OUT_DIR / f"{name}_chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), "utf-8")
        print(f"[saved] {OUT_DIR / (name + '.npz')} + {name}_chunks.json")

    @classmethod
    def load(cls, embedder, name: str = "q1_index"):
        emb = np.load(OUT_DIR / f"{name}.npz")["emb"]
        chunks = json.loads((OUT_DIR / f"{name}_chunks.json").read_text("utf-8"))
        return cls(chunks, emb, embedder)


def plot_embedding_space(index: VectorIndex, max_docs_in_legend: int = 12):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = index.emb
    X50 = PCA(n_components=min(50, X.shape[0] - 1), random_state=SEED).fit_transform(X)
    Z = TSNE(n_components=2, perplexity=min(30, max(5, X.shape[0] // 4)), init="pca",
             random_state=SEED).fit_transform(X50)
    docs = sorted({c["doc_id"] for c in index.chunks})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for n, d in enumerate(docs):
        m = np.array([c["doc_id"] == d for c in index.chunks])
        ax.scatter(Z[m, 0], Z[m, 1], s=14, color=cmap(n % 20), label=d if n < max_docs_in_legend else None)
    ax.set_title(f"t-SNE of {X.shape[0]} Chunk Embeddings (color = source document)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(ncol=2, fontsize=7, markerscale=1.5)
    return save_fig("Q1_chunk_embedding_tsne", fig)


# =============================================================================
# 4. Self-made QA dataset (LLM generated from the corpus, evidence is verified verbatim)
# =============================================================================
BANNED_Q = ["本文", "文中", "該段", "此片段", "上述", "根據以上", "這段", "片段"]
# questions that are not self-contained: they point at sentences / examples of the source passage
# (e.g. "在第一句和第二句中，代詞「它們」分別指的是什麼？") or start with a bare pronoun
_NOT_STANDALONE = re.compile(
    r"第[一二三四五六七八九十\d]+[句段個例]|前一句|後一句|此例|這個例子|例句|上文|下文|以上|如下|"
    r"^(它|它們|他們|其|該|此|這|那)")

QA_VERIFY_PROMPT = """請檢查下面這題問答資料的品質。
【問題】{q}
【標準答案】{a}
【證據（原文）】{ev}
判斷：
1. standalone：只看【問題】這一句（不看證據），一般讀者是否能明白在問什麼？若問題依賴原文的上下文（例如「第一句」「這個例子」「它」而沒說明指誰），則為 false。
2. supported：【證據】是否明確支持【標準答案】，且答案沒有與證據矛盾或加入證據沒有的資訊？
只輸出 JSON：{{"standalone": true/false, "supported": true/false}}"""


def verify_qa(llm, items, q_key="question", a_key="answer", ev_key="evidence"):
    """LLM self-check of generated QA items; returns a list of (keep: bool, reason: str)."""
    prompts = [[{"role": "user", "content": QA_VERIFY_PROMPT.format(q=it[q_key], a=it[a_key], ev=it[ev_key])}]
               for it in items]
    out = []
    for o in llm.generate_batch(prompts, max_new_tokens=40, batch_size=16):
        obj = extract_json(o)
        if not isinstance(obj, dict):
            out.append((True, "unparsed"))          # keep when the checker itself fails
            continue
        b = lambda v: v.strip().lower() in ("true", "yes", "是") if isinstance(v, str) else bool(v)
        st, sp = b(obj.get("standalone", True)), b(obj.get("supported", True))
        out.append((st and sp, "ok" if st and sp else ("not standalone" if not st else "not supported")))
    return out

QA_PROMPT = """你是出題老師。請根據下方【文章片段】出 1 個事實型問題。
規則：
1. 問題必須「獨立可理解」：明確寫出主題名稱，禁止使用「本文」「文中」「該片段」「它」等指代。
2. 答案只能從片段得到，用 1~2 句話回答。
3. evidence 必須是片段中「逐字複製」的連續文字（不超過 120 字），且足以支持答案。
4. 只輸出 JSON：{{"question": "...", "answer": "...", "evidence": "..."}}

【主題】{title}
【文章片段】
{passage}"""

FOLLOWUP_PROMPT = """下面是一段文章片段，以及已經發生的第一輪問答。請接著出「第二輪追問」。
規則：
1. followup 是使用者自然的追問，必須使用代名詞或省略主題（例如「它」「這個方法」「那它的缺點呢」），
   不可出現主題名稱，單看這句話無法知道在問什麼。
2. standalone 是把 followup 改寫成完整獨立的問題（要含主題名稱）。
3. answer 為標準答案（1~2 句）；evidence 為片段中「逐字複製」的連續文字（不超過 120 字）。
4. 只輸出 JSON：{{"followup": "...", "standalone": "...", "answer": "...", "evidence": "..."}}

【主題】{title}
【文章片段】
{passage}
【第一輪】
Q: {q1}
A: {a1}"""

# questions that the corpus can NOT answer -> used to measure hallucination / refusal behaviour
UNANSWERABLE = [
    "台北 101 大樓的高度是多少公尺？", "珍珠奶茶是哪一年發明的？", "2022 年世界盃足球賽的冠軍是哪一隊？",
    "法國大革命爆發於哪一年？", "水的沸點在一大氣壓下是幾度？", "莫札特的第四十號交響曲是在哪一年完成的？",
    "玉山的海拔高度是多少？", "光合作用的化學反應式是什麼？", "鯨魚是魚類還是哺乳類？",
    "世界上最長的河流是哪一條？", "第一次工業革命從哪個國家開始？", "比特幣的發明者是誰？",
]


def _sample_passages(corpus, n, seed=SEED, size=450):
    rng = random.Random(seed)
    by_doc = {}
    for d in corpus:
        cs = [c for c in build_chunks([d], size, 0) if len(c["text"]) >= 200]
        rng.shuffle(cs)
        by_doc[d["doc_id"]] = cs
    out = []
    while len(out) < n and any(by_doc.values()):  # round-robin => topic diversity
        for k in list(by_doc):
            if by_doc[k] and len(out) < n:
                out.append(by_doc[k].pop())
    return out


def _valid_evidence(ev, passage):
    return isinstance(ev, str) and 8 <= len(ev.strip()) <= 200 and ev.strip() in passage


def generate_qa_dataset(llm, corpus, n_target: int = 120, seed: int = SEED, force: bool = False):
    path = DATA_DIR / "q1_qa_dataset.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    doc_text = {d["doc_id"]: d["text"] for d in corpus}
    cands = _sample_passages(corpus, int(n_target * 2.2), seed)
    prompts = [[{"role": "user", "content": QA_PROMPT.format(title=c["title"], passage=c["text"])}]
               for c in cands]
    outs = llm.generate_batch(prompts, max_new_tokens=300, batch_size=8)
    pool, seen, drop = [], set(), {"format": 0, "pronoun / context-dependent": 0}
    for c, o in zip(cands, outs):
        obj = extract_json(o)
        if not isinstance(obj, dict):
            drop["format"] += 1
            continue
        q, a, ev = (str(obj.get(k, "")).strip() for k in ("question", "answer", "evidence"))
        q, a = to_tw(q), to_tw(a)                     # the evidence must stay verbatim (char positions)
        if not (6 <= len(q) <= 100 and len(a) >= 2 and _valid_evidence(ev, c["text"])) or q in seen:
            drop["format"] += 1
            continue
        if any(b in q for b in BANNED_Q) or _NOT_STANDALONE.search(q):
            drop["pronoun / context-dependent"] += 1
            continue
        s = doc_text[c["doc_id"]].find(ev, c["start"])
        if s < 0:
            drop["format"] += 1
            continue
        seen.add(q)
        pool.append(dict(question=q, answer=a, evidence=ev, doc_id=c["doc_id"],
                         title=c["title"], ev_start=s, ev_end=s + len(ev)))
    # LLM self-check: drop questions that are not standalone or whose answer is not supported by the evidence
    verdicts = verify_qa(llm, pool)
    rejected = [dict(**it, reason=r) for it, (k, r) in zip(pool, verdicts) if not k]
    for r in ("not standalone", "not supported"):
        drop[r] = sum(x["reason"] == r for x in rejected)
    qa = [dict(id=f"qa{i:03d}", **it) for i, it in
          enumerate([it for it, (k, _) in zip(pool, verdicts) if k][:n_target])]
    path.write_text(json.dumps(qa, ensure_ascii=False, indent=1), "utf-8")
    (DATA_DIR / "q1_qa_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=1), "utf-8")
    print(f"[QA] kept {len(qa)} / {len(cands)} candidates -> {path}")
    print(f"[QA] dropped: {drop}  (rejected items saved to Data/q1_qa_rejected.json for manual review)")
    return qa


_PRONOUN = re.compile(r"它|它們|其|這個|這些|那個|那些|該|此|他們|這種|那種")


def generate_multiturn_dataset(llm, corpus, qa, n_target: int = 40, seed: int = SEED, force: bool = False,
                               switch_ratio: float = 0.3):
    """2-turn conversations, two kinds:
    * kind="pronoun": turn-2 is a pronoun / ellipsis follow-up on the SAME topic (needs the history);
    * kind="switch" : turn-2 changes to ANOTHER document's topic (the history is only noise).
    Without "switch" items, history concatenation looks unrealistically good, because every follow-up
    comes from the same passage as turn 1."""
    path = DATA_DIR / "q1_multiturn_dataset.json"
    if path.exists() and not force:
        return json.loads(path.read_text("utf-8"))
    n_switch = int(round(n_target * switch_ratio))
    n_pron = n_target - n_switch
    doc_text = {d["doc_id"]: d["text"] for d in corpus}
    rng = random.Random(seed)
    base = rng.sample(qa, min(len(qa), int(n_pron * 2.0)))
    prompts, meta = [], []
    for it in base:
        s = max(0, it["ev_start"] - 150)
        passage = doc_text[it["doc_id"]][s:s + 520]
        prompts.append([{"role": "user", "content": FOLLOWUP_PROMPT.format(
            title=it["title"], passage=passage, q1=it["question"], a1=it["answer"])}])
        meta.append((it, passage, s))
    outs = llm.generate_batch(prompts, max_new_tokens=350, batch_size=8)
    mt = []
    for (it, passage, s0), o in zip(meta, outs):
        obj = extract_json(o)
        if not isinstance(obj, dict):
            continue
        f, sa, a, ev = (str(obj.get(k, "")).strip() for k in ("followup", "standalone", "answer", "evidence"))
        f, sa, a = to_tw(f), to_tw(sa), to_tw(a)
        if not (3 <= len(f) <= 60 and len(sa) >= 6 and a and _valid_evidence(ev, passage)):
            continue
        if to_tw(it["title"]) in f:  # the follow-up must NOT name the topic
            continue
        # the "oracle" standalone question must really be standalone: no leading 那/這, no bare pronoun
        if re.match(r"^[那這]", sa) or (_PRONOUN.search(sa) and to_tw(it["title"]) not in sa):
            continue
        st = doc_text[it["doc_id"]].find(ev, s0)
        if st < 0:
            continue
        mt.append(dict(id=f"mt{len(mt):03d}", kind="pronoun", doc_id=it["doc_id"], title=it["title"],
                       history=[{"role": "user", "content": it["question"]},
                                {"role": "assistant", "content": it["answer"]}],
                       followup=f, standalone=sa, answer=a, evidence=ev,
                       ev_start=st, ev_end=st + len(ev)))
        if len(mt) >= n_pron:
            break
    # topic-switch items: turn 1 from document A, turn 2 = a (standalone) question about document B
    used = {m["history"][0]["content"] for m in mt}
    pool = [q for q in qa if q["question"] not in used]
    rng.shuffle(pool)
    n_sw = 0
    for b in pool:
        if n_sw >= n_switch:
            break
        others = [q for q in qa if q["doc_id"] != b["doc_id"] and q["question"] != b["question"]]
        if not others:
            break
        a = rng.choice(others)
        mt.append(dict(id=f"mt{len(mt):03d}", kind="switch", doc_id=b["doc_id"], title=b["title"],
                       history=[{"role": "user", "content": a["question"]},
                                {"role": "assistant", "content": a["answer"]}],
                       followup=b["question"], standalone=b["question"], answer=b["answer"],
                       evidence=b["evidence"], ev_start=b["ev_start"], ev_end=b["ev_end"]))
        n_sw += 1
    path.write_text(json.dumps(mt, ensure_ascii=False, indent=1), "utf-8")
    print(f"[MT] kept {len(mt)} multi-turn items ({len(mt) - n_sw} pronoun follow-ups + {n_sw} topic switches)"
          f" -> {path}")
    return mt


def plot_qa_stats(qa, chunks):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist([len(q["question"]) for q in qa], bins=15, color="#54A24B")
    axes[0].set_title("Question Length")
    axes[0].set_xlabel("Characters")
    axes[1].hist([len(q["answer"]) for q in qa], bins=15, color="#E45756")
    axes[1].set_title("Answer Length")
    axes[1].set_xlabel("Characters")
    cnt = pd.Series([q["doc_id"] for q in qa]).value_counts().sort_index()
    axes[2].bar(cnt.index, cnt.values, color="#72B7B2")
    axes[2].set_title(f"QA per Document (total {len(qa)})")
    plt.setp(axes[2].get_xticklabels(), rotation=70, fontsize=7)
    fig.suptitle(f"Self-made QA Dataset Statistics  |  chunks (embeddings) = {len(chunks)}")
    return save_fig("Q1_qa_dataset_statistics", fig)


# =============================================================================
# 5. Prompting techniques
# =============================================================================
TECHNIQUES = ["zero_shot", "grounded_strict", "few_shot", "cot", "extract_then_answer"]
SYS_BASE = "你是一位嚴謹的知識問答助理，一律使用繁體中文回答。"
_FEWSHOT = [
    {"role": "user", "content": "參考資料：\n[1]（來源：地理）\n珠穆朗瑪峰位於中國與尼泊爾邊界，海拔 8,848.86 公尺。\n\n問題：世界最高峰的高度是多少？"},
    {"role": "assistant", "content": "世界最高峰珠穆朗瑪峰的海拔為 8,848.86 公尺 [1]。"},
    {"role": "user", "content": "參考資料：\n[1]（來源：動物）\n長頸鹿是現存最高的陸生動物。\n\n問題：長頸鹿的平均壽命是多久？"},
    {"role": "assistant", "content": "資料不足，無法回答。"},
]


def format_context(hits) -> str:
    return "\n\n".join(f"[{h['rank']}]（來源：{h['chunk']['title']}）\n{h['chunk']['text']}" for h in hits)


def build_messages(technique: str, question: str, hits, history=None):
    ctx = format_context(hits)
    hist = list(history or [])[-6:]
    sys_msg, shots = SYS_BASE, []
    if technique == "zero_shot":
        user = f"參考資料：\n{ctx}\n\n問題：{question}\n請根據參考資料回答。"
    elif technique == "grounded_strict":
        sys_msg = (SYS_BASE + "\n規則：\n1. 只能使用【參考資料】中的資訊，禁止使用外部知識。\n"
                   "2. 每個論點後面以 [編號] 標註引用的參考資料。\n"
                   "3. 若參考資料不足以回答，只回覆「資料不足，無法回答。」\n4. 回答簡潔，不超過 4 句。")
        user = f"參考資料：\n{ctx}\n\n問題：{question}\n（每個論點後請標註 [編號]；資料不足時請回覆「資料不足，無法回答。」）"
    elif technique == "few_shot":
        sys_msg = SYS_BASE + "\n只能根據參考資料回答並標註 [編號]；資料不足時回覆「資料不足，無法回答。」"
        shots = _FEWSHOT
        user = f"參考資料：\n{ctx}\n\n問題：{question}"
    elif technique == "cot":
        sys_msg = SYS_BASE + "\n只能根據參考資料回答；資料不足時最終答案請寫「資料不足，無法回答。」"
        user = (f"參考資料：\n{ctx}\n\n問題：{question}\n\n請先在 <think> 與 </think> 之間逐步分析："
                "哪些參考資料與問題相關、關鍵資訊是什麼；然後在 </think> 之後直接給出簡潔的最終答案，並以 [編號] 標註引用。")
    elif technique == "extract_then_answer":
        sys_msg = SYS_BASE + "\n只能根據參考資料回答。"
        user = (f"參考資料：\n{ctx}\n\n問題：{question}\n\n請依序輸出兩段：\n"
                "摘錄：逐字摘錄與問題最相關的句子（每句後標示 [編號]）；若沒有相關句子請寫「無」。\n"
                "答案：僅根據摘錄作答（簡潔，含 [編號]）；若摘錄為「無」，答案請寫「資料不足，無法回答。」")
    else:
        raise ValueError(technique)
    return [{"role": "system", "content": sys_msg}, *shots, *hist, {"role": "user", "content": user}]


def postprocess_answer(technique: str, text: str) -> str:
    t = text.strip()
    if technique == "cot":
        if "</think>" in t:
            t = t.split("</think>")[-1]
        elif "<think>" in t:
            t = re.split(r"最終答案[:：]", t)[-1] if "最終答案" in t else re.sub(r"<think>.*", "", t, flags=re.S)
        t = t.replace("<think>", "")
    elif technique == "extract_then_answer":
        if "答案：" in t or "答案:" in t:
            t = re.split(r"答案[:：]", t)[-1]
    return t.strip()


def answer_batch(llm, technique, questions, hits_list, histories=None, max_new_tokens=512):
    """Batched generation. Returns (clean_answers, raw_outputs)."""
    histories = histories or [None] * len(questions)
    prompts = [build_messages(technique, q, h, hist) for q, h, hist in zip(questions, hits_list, histories)]
    raws = llm.generate_batch(prompts, max_new_tokens=max_new_tokens, batch_size=6)
    return [postprocess_answer(technique, r) for r in raws], raws


# =============================================================================
# 6. Multi-turn RAG chat
# =============================================================================
class RAGChat:
    def __init__(self, llm, index: VectorIndex, top_k: int = 5, technique: str = "grounded_strict",
                 rewrite: bool = True, max_new_tokens: int = 512):
        self.llm, self.index, self.top_k, self.technique = llm, index, top_k, technique
        self.rewrite, self.max_new_tokens = rewrite, max_new_tokens
        self.history = []

    def reset(self):
        self.history = []

    def ask(self, question: str, top_k=None, technique=None) -> dict:
        k, tech = top_k or self.top_k, technique or self.technique
        t0 = time.perf_counter()
        standalone = rewrite_query(self.llm, self.history, question) if (self.rewrite and self.history) else question
        hits = self.index.search(standalone, k)
        raw = self.llm.generate(build_messages(tech, question, hits, self.history),
                                max_new_tokens=self.max_new_tokens)
        ans = postprocess_answer(tech, raw)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans) if 1 <= int(n) <= len(hits)})
        refs = [dict(h, cited=(h["rank"] in cited)) for h in hits]
        refs.sort(key=lambda r: (not r["cited"], r["rank"]))
        # keep stored history free from [n] markers (indices change every turn)
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": re.sub(r"\[\d+\]", "", ans)}]
        return dict(question=question, rewritten=standalone, answer=ans, refs=refs, hits=hits,
                    latency=time.perf_counter() - t0)


DEMO_TURNS = ["什麼是深度學習？", "它跟傳統機器學習有什麼不同？", "那它常用的神經網路架構有哪些？",
              "其中卷積神經網路主要用在什麼任務上？"]


def run_demo_conversation(chat: RAGChat, turns=DEMO_TURNS, show_chars: int = 160):
    chat.reset()
    results = []
    for i, q in enumerate(turns, 1):
        r = chat.ask(q)
        results.append(r)
        print(f"\n===== Turn {i} =====\n👤 {q}")
        if r["rewritten"] != q:
            print(f"🔁 standalone query: {r['rewritten']}")
        print(f"🤖 {r['answer']}\n📎 References:")
        for ref in r["refs"]:
            flag = "✔ cited" if ref["cited"] else "  retrieved"
            print(f"   [{ref['rank']}] {flag} | {ref['chunk']['doc_id']} {ref['chunk']['title']} | "
                  f"score={ref['score']:.3f} | {ref['chunk']['text'][:show_chars].replace(chr(10), ' ')}…")
    (OUT_DIR / "q1_demo_conversation.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "hits"} for r in results], ensure_ascii=False, indent=1,
        default=str), "utf-8")
    return results


def plot_demo_retrieval(results, name="Q1_demo_retrieval_scores"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharex=True)
    for ax, (i, r) in zip(np.atleast_1d(axes), enumerate(results, 1)):
        labels = [f"{h['chunk']['doc_id']}#{h['chunk']['chunk_id']}" for h in r["refs"]][::-1]
        scores = [h["score"] for h in r["refs"]][::-1]
        cols = ["#E45756" if h["cited"] else "#9ecae9" for h in r["refs"]][::-1]
        ax.barh(labels, scores, color=cols)
        ax.set_title(f"Turn {i}")
        ax.set_xlabel("Cosine similarity")
    fig.suptitle("Retrieved Chunks per Turn (red = cited in the answer)")
    return save_fig(name, fig)


# =============================================================================
# 7. Advanced A - chunk size / overlap / top-k search (stability-aware)
# =============================================================================
def run_chunk_grid(corpus, qa, embedder, chunk_sizes=(150, 300, 500, 800), overlaps=(0, 50, 100, 150),
                   max_k: int = 10) -> pd.DataFrame:
    """Per (chunk_size, overlap, question): rank of the first chunk hitting the gold evidence."""
    rows, questions = [], [q["question"] for q in qa]
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov > cs // 2:
                continue
            chunks = build_chunks(corpus, cs, ov)
            index = VectorIndex.build(chunks, embedder)
            res = index.search_many(questions, k=max_k)
            mean_len = float(np.mean([len(c["text"]) for c in chunks]))
            for q, hits in zip(qa, res):
                rank = next((h["rank"] for h in hits
                             if chunk_hits_evidence(h["chunk"], q["doc_id"], q["ev_start"], q["ev_end"])), 99)
                rows.append(dict(chunk_size=cs, overlap=ov, qid=q["id"], doc_id=q["doc_id"], rank=rank,
                                 n_chunks=len(chunks), mean_chunk_len=mean_len))
            print(f"[grid] chunk={cs:4d} overlap={ov:3d} -> {len(chunks)} chunks")
    return pd.DataFrame(rows)


def summarize_grid(df: pd.DataFrame, ks=(1, 3, 5, 8, 10), lam: float = 0.05, n_boot: int = 300,
                   seed: int = SEED) -> pd.DataFrame:
    """Stability-aware summary.
        hit_macro = mean over documents of per-document Hit@K   (topic-balanced)
        std_doc   = std over documents                          (consistency across topics)
        ci_low    = 5th percentile of bootstrap(question) hit   (worst-case estimate)
        score     = 0.5 * hit_macro + 0.5 * MRR - std_doc - lam * context_kchars
                    (lam: price of 1000 extra context chars). Hit@K alone saturates at large K (every config
                    reaches 1.0), so MRR - i.e. how high the gold chunk is ranked - is part of the score.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (cs, ov), g in df.groupby(["chunk_size", "overlap"]):
        for k in ks:
            hit = (g["rank"] <= k).astype(float).values
            rr = np.where(g["rank"].values <= k, 1.0 / g["rank"].values, 0.0)
            per_doc = pd.Series(hit).groupby(g["doc_id"].values).agg(["mean", "count"])
            per_doc = per_doc[per_doc["count"] >= 2] if (per_doc["count"] >= 2).sum() >= 3 else per_doc
            boots = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n_boot)]
            ctx = k * g["mean_chunk_len"].iloc[0]
            macro, sd = per_doc["mean"].mean(), per_doc["mean"].std(ddof=0)
            rows.append(dict(chunk_size=cs, overlap=ov, k=k, hit_micro=hit.mean(), hit_macro=macro,
                             std_doc=sd, ci_low=float(np.percentile(boots, 5)), mrr=rr.mean(),
                             ctx_chars=ctx, n_chunks=int(g["n_chunks"].iloc[0]),
                             hit1=float((g["rank"] <= 1).mean()),
                             score=0.5 * macro + 0.5 * rr.mean() - sd - lam * ctx / 1000))
    return pd.DataFrame(rows)


def top_configs(summary: pd.DataFrame, n: int = 3) -> list:
    best = summary.sort_values("score", ascending=False).head(n)
    return [dict(chunk_size=int(r.chunk_size), overlap=int(r.overlap), k=int(r.k)) for r in best.itertuples()]


def plot_grid_heatmaps(summary, ks=(1, 3, 5, 8)):
    fig, axes = plt.subplots(1, len(ks), figsize=(4.7 * len(ks), 4.3))
    vmin, vmax = summary["hit_macro"].min(), summary["hit_macro"].max()
    for ax, k in zip(np.atleast_1d(axes), ks):
        s = summary[summary.k == k]
        pm = s.pivot(index="chunk_size", columns="overlap", values="hit_macro")
        ps = s.pivot(index="chunk_size", columns="overlap", values="std_doc")
        im = ax.imshow(pm.values, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(pm.shape[1]))
        ax.set_xticklabels(pm.columns)
        ax.set_yticks(range(pm.shape[0]))
        ax.set_yticklabels(pm.index)
        ax.grid(False)
        for i in range(pm.shape[0]):
            for j in range(pm.shape[1]):
                v = pm.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}\n±{ps.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v < (vmin + vmax) / 2 else "black")
        ax.set_title(f"Hit@{k}  (mean ± std over docs)")
        ax.set_xlabel("Overlap (chars)")
        ax.set_ylabel("Chunk size (chars)" if k == ks[0] else "")
    fig.subplots_adjust(wspace=0.3)
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.8, label="Macro Hit@K")
    fig.suptitle("Retrieval Hit Rate by Chunk Size x Overlap", y=1.03)
    return save_fig("Q1_grid_hit_heatmaps", fig)


def plot_k_curves(summary):
    best_ov = summary.loc[summary.groupby(["chunk_size", "k"])["score"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cs, g in best_ov.groupby("chunk_size"):
        g = g.sort_values("k")
        axes[0].errorbar(g["k"], g["hit_macro"], yerr=g["std_doc"], marker="o", capsize=3, label=f"chunk={cs}")
        axes[1].plot(g["k"], g["mrr"], marker="s", label=f"chunk={cs}")
    axes[0].set_title("Hit@K vs Top-K (error bar = std across docs)")
    axes[1].set_title("MRR@K vs Top-K")
    for a in axes:
        a.set_xlabel("Top-K")
        a.legend()
    axes[0].set_ylabel("Macro Hit@K")
    axes[1].set_ylabel("MRR")
    return save_fig("Q1_grid_hit_vs_topk", fig)


def plot_stability_scatter(summary):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(summary["std_doc"], summary["hit_macro"], c=summary["k"], s=summary["ctx_chars"] / 8 + 15,
                    cmap="plasma", alpha=0.75, edgecolor="k", linewidth=0.3)
    b = summary.sort_values("score", ascending=False).iloc[0]
    ax.scatter([b.std_doc], [b.hit_macro], s=260, facecolor="none", edgecolor="red", linewidth=2)
    ax.annotate(f"best: chunk={int(b.chunk_size)}, ov={int(b.overlap)}, K={int(b.k)}", (b.std_doc, b.hit_macro),
                textcoords="offset points", xytext=(10, -18), color="red")
    ax.set_xlabel("Std of per-document Hit@K  (lower = more stable)")
    ax.set_ylabel("Macro Hit@K  (higher = better)")
    ax.set_title("Accuracy vs Stability (bubble size = context length, color = K)")
    fig.colorbar(sc, label="Top-K")
    return save_fig("Q1_grid_stability_scatter", fig)


def evaluate_end2end(llm, embedder, corpus, qa_eval, configs, technique="grounded_strict"):
    """Stage 2: answer quality of the most promising retrieval configs (LLM-judge + char-F1)."""
    questions = [q["question"] for q in qa_eval]
    golds = [q["answer"] for q in qa_eval]
    rows = []
    for cfg in configs:
        index = VectorIndex.build(build_chunks(corpus, cfg["chunk_size"], cfg["overlap"]), embedder)
        hits = index.search_many(questions, k=cfg["k"])
        answers, _ = answer_batch(llm, technique, questions, hits)
        scores = judge_correctness(llm, list(zip(questions, golds, answers)))
        for q, g, a, s, h in zip(qa_eval, golds, answers, scores, hits):
            rows.append(dict(config=f"c{cfg['chunk_size']}/o{cfg['overlap']}/k{cfg['k']}", qid=q["id"],
                             judge=s, char_f1=char_f1(a, g), doc_id=q["doc_id"],
                             hit=any(chunk_hits_evidence(x["chunk"], q["doc_id"], q["ev_start"], q["ev_end"]) for x in h)))
    return pd.DataFrame(rows)


def plot_end2end(df):
    g = df.groupby("config").agg(judge=("judge", "mean"), judge_sd=("judge", "std"),
                                 f1=("char_f1", "mean"), hit=("hit", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].bar(g["config"], g["judge"], yerr=g["judge_sd"], capsize=3, color="#4C78A8")
    axes[0].set_title("LLM-judge correctness (1-5)")
    axes[1].bar(g["config"], g["f1"], color="#F58518")
    axes[1].set_title("Character-level F1")
    axes[2].bar(g["config"], g["hit"], color="#54A24B")
    axes[2].set_title("Retrieval Hit@K")
    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=20)
    fig.suptitle("End-to-end Comparison of the Top Retrieval Configs")
    return save_fig("Q1_end2end_top_configs", fig)


def save_best_config(cfg: dict, technique: str = "grounded_strict"):
    out = dict(cfg, technique=technique)
    (OUT_DIR / "q1_best_config.json").write_text(json.dumps(out, indent=1), "utf-8")
    print("[saved] q1_best_config.json", out)


def load_best_config() -> dict:
    p = OUT_DIR / "q1_best_config.json"
    return json.loads(p.read_text()) if p.exists() else dict(chunk_size=300, overlap=50, k=5,
                                                              technique="grounded_strict")


# =============================================================================
# 8. Advanced B - prompting technique comparison
# =============================================================================
def evaluate_prompting(llm, index: VectorIndex, qa_eval, techniques=TECHNIQUES, k: int = 5,
                       unanswerable=UNANSWERABLE) -> pd.DataFrame:
    qs = [q["question"] for q in qa_eval]
    golds = [q["answer"] for q in qa_eval]
    hits = index.search_many(qs, k)
    un_hits = index.search_many(unanswerable, k)
    rows = []
    for tech in techniques:
        t0 = time.perf_counter()
        answers, _ = answer_batch(llm, tech, qs, hits)
        lat = (time.perf_counter() - t0) / len(qs)
        scores = judge_correctness(llm, list(zip(qs, golds, answers)))
        un_answers, _ = answer_batch(llm, tech, unanswerable, un_hits)
        for q, g, a, s in zip(qs, golds, answers, scores):
            rows.append(dict(technique=tech, kind="answerable", judge=s, char_f1=char_f1(a, g),
                             cited=bool(re.search(r"\[\d+\]", a)), refused=is_refusal(a), length=len(a),
                             sec_per_q=lat))
        for q, a in zip(unanswerable, un_answers):
            rows.append(dict(technique=tech, kind="unanswerable", judge=np.nan, char_f1=np.nan,
                             cited=bool(re.search(r"\[\d+\]", a)), refused=is_refusal(a), length=len(a),
                             sec_per_q=lat))
        print(f"[prompting] {tech} done")
    return pd.DataFrame(rows)


def plot_prompting(df):
    ans = df[df.kind == "answerable"].groupby("technique")
    un = df[df.kind == "unanswerable"].groupby("technique")
    order = [t for t in TECHNIQUES if t in df.technique.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], ans["judge"].mean().reindex(order), "LLM-judge correctness (1-5) ↑", "#4C78A8"),
        (axes[0, 1], ans["char_f1"].mean().reindex(order), "Character F1 ↑", "#F58518"),
        (axes[0, 2], ans["cited"].mean().reindex(order), "Citation rate (answerable) ↑", "#54A24B"),
        (axes[1, 0], un["refused"].mean().reindex(order), "Correct refusal on unanswerable ↑", "#B279A2"),
        (axes[1, 1], ans["length"].mean().reindex(order), "Avg answer length (chars)", "#9D755D"),
        (axes[1, 2], ans["sec_per_q"].mean().reindex(order), "Seconds per question ↓", "#BAB0AC"),
    ]
    for ax, ser, title, col in panels:
        ax.bar(range(len(ser)), ser.values, color=col)
        ax.set_xticks(range(len(ser)))
        ax.set_xticklabels(ser.index, rotation=20)
        ax.set_title(title)
        for i, v in enumerate(ser.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Prompting Technique Comparison")
    return save_fig("Q1_prompting_technique_comparison", fig)


def choose_technique(df) -> str:
    """Composite = judge/5 + refusal-on-unanswerable (equal weight) + 0.3 * citation rate."""
    a = df[df.kind == "answerable"].groupby("technique")
    u = df[df.kind == "unanswerable"].groupby("technique")
    comp = a["judge"].mean() / 5 + u["refused"].mean() + 0.3 * a["cited"].mean()
    return comp.idxmax()


# =============================================================================
# 9. User interface (Gradio)
# =============================================================================
def format_refs(refs) -> str:
    if not refs:
        return "_No references._"
    parts = []
    for r in refs:
        mark = "✅ cited" if r.get("cited") else "▫️ retrieved"
        c = r["chunk"]
        parts.append(f"<details {'open' if r.get('cited') else ''}><summary><b>[{r['rank']}] {c['title']}</b> "
                     f"({c['doc_id']}) · score {r['score']:.3f} · {mark}</summary>\n\n> "
                     f"{c['text'].replace(chr(10), ' ')}\n\n</details>")
    return "\n".join(parts)


def launch_ui(chat: RAGChat, share: bool = False, port: int = 7860, inline: bool = False):
    import gradio as gr
    with gr.Blocks(title="Q1 Multi-turn RAG Chatbot") as demo:
        gr.Markdown("# 📚 Multi-turn RAG Chatbot\nAI / Machine-Learning knowledge base (Wikipedia zh-TW) · "
                    "bge-m3 retrieval · Qwen2.5-7B-Instruct")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = make_chatbot(gr, height=520, label="Conversation")
                msg = gr.Textbox(placeholder="輸入問題後按 Enter…（可以直接追問「它」「那…呢」）", show_label=False)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear conversation")
                gr.Examples(DEMO_TURNS, inputs=msg, label="Try a multi-turn demo (ask them in order)")
            with gr.Column(scale=2):
                top_k = gr.Slider(1, 10, value=chat.top_k, step=1, label="Top-K")
                tech = gr.Dropdown(TECHNIQUES, value=chat.technique, label="Prompting technique")
                rw = gr.Checkbox(value=chat.rewrite, label="Query rewriting (multi-turn)")
                standalone = gr.Textbox(label="Standalone query used for retrieval", interactive=False)
                latency = gr.Textbox(label="Latency", interactive=False)
                refs_md = gr.Markdown("_References will appear here._")

        def respond(message, history, k, t, r):
            if not message.strip():
                return history, "", gr.update(), gr.update(), gr.update()
            chat.rewrite = bool(r)
            out = chat.ask(message, top_k=int(k), technique=t)
            history = (history or []) + [{"role": "user", "content": message},
                                         {"role": "assistant", "content": out["answer"]}]
            return history, "", out["rewritten"], f"{out['latency']:.1f} s", format_refs(out["refs"])

        def reset():
            chat.reset()
            return [], "", "", "", "_References will appear here._"

        outs = [chatbot, msg, standalone, latency, refs_md]
        msg.submit(respond, [msg, chatbot, top_k, tech, rw], outs)
        send.click(respond, [msg, chatbot, top_k, tech, rw], outs)
        clear.click(reset, None, outs)
    launch_blocks(demo, gr, share=share, server_port=port, inline=inline, prevent_thread_lock=True)
    return demo


if __name__ == "__main__":  # python Codes/Quiz1.py  -> start the UI from the saved index
    cfg = load_best_config()
    emb = Embedder()
    idx = VectorIndex.load(emb)
    llm = ChatLLM()
    chat = RAGChat(llm, idx, top_k=cfg["k"], technique=cfg["technique"])
    d = launch_ui(chat)
    input("UI running at http://127.0.0.1:7860 - press Enter to stop.")
