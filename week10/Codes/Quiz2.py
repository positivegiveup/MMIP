"""Quiz 2 - Multi-turn VideoRAG chatbot.

Adapted from VideoRAG (Jeong et al., 2025: retrieve videos with visual + textual signals, feed
frames + ASR text to a large video-language model, and generate auxiliary text for videos that have
no speech). Our adaptation:

  OFFLINE  video -> fixed-length clips (10 s) -> 4 frames / clip
           |- ASR   : faster-whisper large-v3 -> transcript per clip          (speech videos)
           |- Caption: Qwen2.5-VL describes frames (actions + on-screen text)  (auxiliary text, esp. silent videos)
           |- Text embedding  (bge-m3)           of  caption + transcript
           '- Visual embedding (Chinese-CLIP)    of  mean(frame embeddings)
  ONLINE   query rewriting with chat history  ->  dual retrieval (text + visual, z-score fusion, alpha)
           -> top-k clips -> Qwen2.5-VL answers from frames + caption + transcript, with [V#] citations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import re
import shutil
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from Codes.common import *  # noqa: F401,F403

set_style()

VIDEO_DIR = DATA_DIR / "q2_videos"
CACHE_DIR = DATA_DIR / "q2_cache"
FRAME_DIR = CACHE_DIR / "frames"
CLIP_DIR = CACHE_DIR / "clips"
for _d in (VIDEO_DIR, CACHE_DIR, FRAME_DIR, CLIP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CLIP_LEN = 10.0          # seconds per clip
FRAMES_PER_CLIP = 4      # frames used for caption / visual embedding
ALPHA = 0.6              # text weight in fusion (tuned in Quiz3)
CLIP_MODEL = "OFA-Sys/chinese-clip-vit-large-patch14"

RECORDING_GUIDE = f"""
No videos found in {VIDEO_DIR}.
Please record your own videos and put them there (mp4/mov/mkv/webm), e.g.
  * 4-6 SPEECH videos   : you explain something while showing a screen / whiteboard / object (30-120 s)
  * 4-6 SILENT videos   : screen-recording of an operation (install a tool, use a website, cook a recipe ...)
                          with NO narration (30-120 s)
Different topics per video make retrieval meaningful. Then re-run the cell.
"""


# =============================================================================
# 1. Video corpus: discovery, probing, clips, frames
# =============================================================================
def list_videos():
    vids = sorted(p for p in VIDEO_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        raise FileNotFoundError(RECORDING_GUIDE)
    return vids


def probe_videos(videos) -> pd.DataFrame:
    import cv2
    rows = []
    for i, p in enumerate(videos):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        rows.append(dict(vid_idx=i, video_id=p.stem, path=str(p), fps=fps, duration=n / fps,
                         width=int(w), height=int(h),
                         folder_label=p.parent.name if p.parent.name in ("voice", "no_voice") else ""))
    return pd.DataFrame(rows)


def make_clips(meta: pd.DataFrame, clip_len: float = CLIP_LEN) -> list:
    clips = []
    for r in meta.itertuples():
        starts = list(np.arange(0, r.duration, clip_len))
        if len(starts) > 1 and r.duration - starts[-1] < 0.4 * clip_len:   # merge a tiny tail
            starts.pop()
        for j, s in enumerate(starts):
            e = r.duration if j == len(starts) - 1 else s + clip_len
            tag = hashlib.md5(r.video_id.encode("utf-8")).hexdigest()[:6]   # cache key tied to the video name
            clips.append(dict(clip_id=f"v{r.vid_idx:02d}_{tag}_c{j:03d}", clip_idx=j, vid_idx=r.vid_idx,
                              video_id=r.video_id, video_path=r.path, start=float(s), end=float(e)))
    return clips


def get_clip_frames(clip: dict, n: int = FRAMES_PER_CLIP, max_side: int = 448) -> list:
    """Uniformly sample n frames inside the clip; cached as JPG. Returns list of file paths."""
    import cv2
    paths = [FRAME_DIR / f"{clip['clip_id']}_{i}.jpg" for i in range(n)]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]
    cap = cv2.VideoCapture(clip["video_path"])
    for i, p in enumerate(paths):
        t = clip["start"] + (i + 0.5) * (clip["end"] - clip["start"]) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:  # fall back to the previous good frame
            if i > 0 and paths[i - 1].exists():
                Image.open(paths[i - 1]).save(p)
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        s = max_side / max(img.size)
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        img.save(p, quality=90)
    cap.release()
    return [str(p) for p in paths if p.exists()]


def extract_all_frames(clips):
    for c in clips:
        c["frames"] = get_clip_frames(c)
    return clips


# =============================================================================
# 2. ASR (speech) - faster-whisper, with a transformers fallback
# =============================================================================
def transcribe_videos(videos, language=None, force=False) -> dict:
    """Returns {video_id: [segments]}. Silent / audio-less videos give [].
    ASR runs in a SUBPROCESS (Codes/asr_worker.py) so that all Whisper VRAM is returned before the VLM loads."""
    todo = [p for p in videos if force or not (CACHE_DIR / f"asr_{p.stem}.json").exists()]
    if todo:
        cmd = [sys.executable, "-u", "-m", "Codes.asr_worker", str(CACHE_DIR), *map(str, todo)]
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            print(f"[ASR] worker exited with code {proc.returncode}")
    return {p.stem: (json.loads((CACHE_DIR / f"asr_{p.stem}.json").read_text("utf-8"))
                     if (CACHE_DIR / f"asr_{p.stem}.json").exists() else []) for p in videos}


def attach_transcripts(clips, asr: dict, meta: pd.DataFrame, min_ratio: float = 0.15, min_chars: int = 10):
    """Adds clip['transcript'], clip['has_speech'] and video-level speech flags (returned as DataFrame)."""
    vinfo = []
    for r in meta.itertuples():
        segs = asr.get(r.video_id, [])
        sp = sum(s["end"] - s["start"] for s in segs)
        chars = sum(len(s["text"]) for s in segs)
        asr_flag = bool(sp / max(r.duration, 1e-6) >= min_ratio and chars >= min_chars)
        lab = getattr(r, "folder_label", "")
        # the folder (voice / no_voice) is the ground truth; ASR-based detection is kept for comparison
        final = (lab == "voice") if lab else asr_flag
        vinfo.append(dict(video_id=r.video_id, speech_ratio=sp / max(r.duration, 1e-6), chars=chars,
                          asr_has_speech=asr_flag, folder_label=lab, has_speech=final))
    vdf = pd.DataFrame(vinfo)
    vflag = dict(zip(vdf.video_id, vdf.has_speech))
    for c in clips:
        segs = asr.get(c["video_id"], []) if vflag[c["video_id"]] else []   # whisper hallucinates on silent videos
        c["transcript"] = " ".join(s["text"] for s in segs if c["start"] <= (s["start"] + s["end"]) / 2 < c["end"]).strip()
        c["video_has_speech"] = vflag[c["video_id"]]
        c["has_speech"] = len(c["transcript"]) >= 5
    return vdf


# =============================================================================
# 3. Auxiliary text (captions) with the VLM
# =============================================================================
CAPTION_PROMPT = ("以下是同一段影片（第 {s}–{e} 秒）依時間順序抽出的 {n} 張畫面。請用繁體中文簡潔描述：\n"
                  "(1) 場景與主要物件；(2) 人物、滑鼠或使用者正在進行的操作／動作（例如點擊哪個按鈕、輸入什麼）；"
                  "(3) 畫面上可見的重要文字（標題、按鈕、程式碼、網址）。\n限 120 字內，只描述看得到的內容，不要猜測。")


def caption_clips(vlm, clips, force=False):
    path = CACHE_DIR / "captions.json"
    cache = json.loads(path.read_text("utf-8")) if path.exists() and not force else {}
    for i, c in enumerate(clips):
        if c["clip_id"] in cache:
            c["caption"] = cache[c["clip_id"]]
            continue
        imgs = [Image.open(p).convert("RGB") for p in c["frames"]]
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": CAPTION_PROMPT.format(
            s=int(c["start"]), e=int(c["end"]), n=len(imgs))}]
        c["caption"] = to_tw(vlm.generate([{"role": "user", "content": content}], images=imgs, max_new_tokens=220))
        cache[c["clip_id"]] = c["caption"]
        if i % 10 == 0:
            path.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
            print(f"[caption] {i + 1}/{len(clips)}")
    path.write_text(json.dumps(cache, ensure_ascii=False), "utf-8")
    return clips


# =============================================================================
# 4. Visual encoder (Chinese-CLIP) and the dual index
# =============================================================================
class ChineseCLIP:
    def __init__(self, name: str = CLIP_MODEL):
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        self.torch = torch
        self.model = ChineseCLIPModel.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.proc = ChineseCLIPProcessor.from_pretrained(name)

    def encode_images(self, imgs, batch_size: int = 32) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(imgs), batch_size):
                px = self.proc(images=imgs[s:s + batch_size], return_tensors="pt")["pixel_values"].cuda().half()
                # transformers 5.x returns a ModelOutput from get_image_features(); compute
                # pooled vision output -> visual_projection directly (works on 4.x and 5.x).
                pooled = self.model.vision_model(pixel_values=px).pooler_output
                f = self.model.visual_projection(pooled)
                out.append(self.torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def encode_text(self, texts, batch_size: int = 64) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                enc = self.proc(text=list(texts[s:s + batch_size]), padding=True, truncation=True,
                                max_length=52, return_tensors="pt").to("cuda")
                # transformers 4.57 bug: text_model has no pooler (add_pooling_layer=False) but
                # get_text_features() reads pooler_output (= None). Chinese-CLIP's original definition uses
                # the [CLS] hidden state -> text_projection, so compute it directly.
                h = self.model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                          token_type_ids=enc.get("token_type_ids")).last_hidden_state
                f = self.model.text_projection(h[:, 0, :])
                out.append(self.torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)


class VideoRAGIndex:
    def __init__(self, clips, text_emb, vis_emb, embedder=None, clip_model=None):
        self.clips, self.text_emb, self.vis_emb = clips, text_emb, vis_emb
        self.embedder, self.clip_model = embedder, clip_model

    # ---- build / persist -------------------------------------------------
    @classmethod
    def build(cls, clips, embedder, clip_model):
        docs = [f"{c['caption']}\n{c['transcript']}".strip() for c in clips]
        text_emb = embedder.encode(docs)
        vis = []
        for c in clips:
            f = clip_model.encode_images([Image.open(p).convert("RGB") for p in c["frames"]])
            v = f.mean(0)
            vis.append(v / (np.linalg.norm(v) + 1e-8))
        return cls(clips, text_emb, np.stack(vis), embedder, clip_model)

    def save(self, name: str = "q2_videorag_index"):
        np.savez_compressed(OUT_DIR / f"{name}.npz", text_emb=self.text_emb, vis_emb=self.vis_emb)
        (OUT_DIR / f"{name}_clips.json").write_text(json.dumps(self.clips, ensure_ascii=False), "utf-8")
        print(f"[saved] {OUT_DIR / (name + '.npz')} + {name}_clips.json")

    @classmethod
    def load(cls, embedder, clip_model, name: str = "q2_videorag_index"):
        z = np.load(OUT_DIR / f"{name}.npz")
        clips = json.loads((OUT_DIR / f"{name}_clips.json").read_text("utf-8"))
        return cls(clips, z["text_emb"], z["vis_emb"], embedder, clip_model)

    # ---- retrieval -------------------------------------------------------
    def score_matrices(self, queries):
        """Raw cosine scores: (Q, N) text-side and (Q, N) visual-side."""
        st = self.embedder.encode(list(queries)) @ self.text_emb.T
        sv = self.clip_model.encode_text(list(queries)) @ self.vis_emb.T
        return st, sv

    @staticmethod
    def fuse(st, sv, mode: str = "fusion", alpha: float = ALPHA):
        if mode == "text":
            return zscore(st)
        if mode == "visual":
            return zscore(sv)
        return alpha * zscore(st) + (1 - alpha) * zscore(sv)

    def retrieve(self, query: str, k: int = 3, mode: str = "fusion", alpha: float = ALPHA, max_per_video=None):
        st, sv = self.score_matrices([query])
        s = self.fuse(st, sv, mode, alpha)[0]
        hits, per_video = [], {}
        for j in np.argsort(-s):
            c = self.clips[j]
            if max_per_video and per_video.get(c["video_id"], 0) >= max_per_video:
                continue
            per_video[c["video_id"]] = per_video.get(c["video_id"], 0) + 1
            hits.append(dict(rank=len(hits) + 1, score=float(s[j]), z_text=float(zscore(st)[0, j]),
                             z_vis=float(zscore(sv)[0, j]), idx=int(j), clip=c))
            if len(hits) >= k:
                break
        return hits


# =============================================================================
# 5. Generation (multi-turn VideoRAG chat)
# =============================================================================
VIDEO_SYS = ("你是影片知識庫助理。你會收到從影片庫檢索到的片段（含畫面、自動畫面描述與語音逐字稿），"
             "請只根據這些片段回答，使用繁體中文。")
VIDEO_RULES = ("規則：只根據上述片段作答；每個論點後面用 [V編號] 標註來源；可以引用時間點；"
               "若片段裡找不到答案，回答「影片中找不到相關資訊」。回答簡潔。")


def _pick(frames, n):
    if len(frames) <= n:
        return frames
    return [frames[int(round(i * (len(frames) - 1) / (n - 1)))] for i in range(n)] if n > 1 else [frames[len(frames) // 2]]


def build_video_messages(question, hits, history=None, gen_mode: str = "full", n_frames: int = 2):
    """gen_mode: full (frames + caption + transcript) | text_only (caption + transcript) | frames_only."""
    images = []
    content = [{"type": "text", "text": "以下是檢索到的影片片段（依相關度排序）：\n"}]
    for i, h in enumerate(hits, 1):
        c = h["clip"]
        parts = [f"[V{i}] 影片「{c['video_id']}」 {fmt_time(c['start'])}–{fmt_time(c['end'])}"]
        if gen_mode in ("full", "text_only"):
            parts.append(f"畫面描述（自動產生）：{c['caption']}")
            parts.append(f"語音逐字稿：{c['transcript'] or '（此片段沒有語音）'}")
        content.append({"type": "text", "text": "\n".join(parts) + "\n"})
        if gen_mode in ("full", "frames_only"):
            for p in _pick(c["frames"], n_frames):
                content.append({"type": "image"})
                images.append(Image.open(p).convert("RGB"))
    content.append({"type": "text", "text": f"\n問題：{question}\n{VIDEO_RULES}"})
    msgs = [{"role": "system", "content": VIDEO_SYS}, *(history or [])[-6:], {"role": "user", "content": content}]
    return msgs, images


def cut_clip(clip: dict):
    """Cut the clip into a small mp4 for the UI (needs ffmpeg). Returns path or None."""
    out = CLIP_DIR / f"{clip['clip_id']}.mp4"
    if out.exists():
        return str(out)
    if shutil.which("ffmpeg") is None:      # no ffmpeg: play the whole source video instead of nothing
        return clip["video_path"]
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(clip["start"]), "-to", str(clip["end"]),
                        "-i", clip["video_path"], "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)],
                       check=True, timeout=120)
        return str(out)
    except Exception:
        return clip["video_path"]


class VideoRAGChat:
    def __init__(self, vlm, index: VideoRAGIndex, top_k: int = 3, mode: str = "fusion", alpha: float = ALPHA,
                 gen_mode: str = "full", rewrite: bool = True, n_frames: int = 2, max_new_tokens: int = 400):
        self.vlm, self.index, self.top_k, self.mode, self.alpha = vlm, index, top_k, mode, alpha
        self.gen_mode, self.rewrite, self.n_frames, self.max_new_tokens = gen_mode, rewrite, n_frames, max_new_tokens
        self.history = []

    def reset(self):
        self.history = []

    def ask(self, question: str, top_k=None, mode=None, alpha=None) -> dict:
        t0 = time.perf_counter()
        standalone = rewrite_query(self.vlm, self.history, question) if (self.rewrite and self.history) else question
        hits = self.index.retrieve(standalone, top_k or self.top_k, mode or self.mode,
                                   self.alpha if alpha is None else alpha)
        msgs, imgs = build_video_messages(question, hits, self.history, self.gen_mode, self.n_frames)
        ans = self.vlm.generate(msgs, images=imgs or None, max_new_tokens=self.max_new_tokens)
        cited = sorted({int(n) for n in re.findall(r"\[V(\d+)\]", ans) if 1 <= int(n) <= len(hits)})
        refs = [dict(h, cited=(h["rank"] in cited)) for h in hits]
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": re.sub(r"\[V\d+\]", "", ans)}]
        return dict(question=question, rewritten=standalone, answer=ans, refs=refs, hits=hits,
                    latency=time.perf_counter() - t0)


DEMO_TURNS = ["有沒有影片是在教怎麼在 Excel 裡計算平均值或最大值？",
              "那個函數要怎麼輸入？",
              "有哪支影片在講解 RAG？它主要在解決什麼問題？",
              "另外有沒有影片是在示範螢幕錄影的操作？第一步要做什麼？"]


def run_demo_conversation(chat: VideoRAGChat, turns=DEMO_TURNS):
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
            c = ref["clip"]
            print(f"   [V{ref['rank']}] {'✔ cited' if ref['cited'] else '  retrieved'} | vid{c['vid_idx']} {c['video_id']} "
                  f"{fmt_time(c['start'])}-{fmt_time(c['end'])} | fused={ref['score']:.2f} "
                  f"(text z={ref['z_text']:.2f}, visual z={ref['z_vis']:.2f})")
    (OUT_DIR / "q2_demo_conversation.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "hits"} for r in results], ensure_ascii=False, indent=1, default=str), "utf-8")
    return results


# =============================================================================
# 6. Figures (English titles / ASCII labels only -> no CJK font problems)
# =============================================================================
def plot_corpus_overview(meta: pd.DataFrame, vdf: pd.DataFrame, clips):
    m = meta.merge(vdf, on="video_id")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    cols = ["#E45756" if s else "#4C78A8" for s in m["has_speech"]]
    axes[0].bar([f"vid{i}" for i in m["vid_idx"]], m["duration"], color=cols)
    axes[0].set_title("Video Duration (red = speech, blue = silent)")
    axes[0].set_ylabel("Seconds")
    axes[1].bar([f"vid{i}" for i in m["vid_idx"]], m["speech_ratio"], color=cols)
    axes[1].set_title("Speech Ratio per Video (ASR)")
    axes[1].set_ylim(0, 1)
    axes[2].hist([len(c["transcript"]) for c in clips], bins=20, color="#72B7B2")
    axes[2].set_title(f"Transcript Length per Clip (N={len(clips)} clips)")
    axes[2].set_xlabel("Characters")
    fig.suptitle("Q2 Video Corpus Overview")
    return save_fig("Q2_video_corpus_overview", fig)


def plot_frame_samples(clips, n_videos: int = 6, cols: int = 4):
    vids = sorted({c["vid_idx"] for c in clips})[:n_videos]
    fig, axes = plt.subplots(len(vids), cols, figsize=(3.3 * cols, 2.3 * len(vids)))
    axes = np.atleast_2d(axes)
    for r, v in enumerate(vids):
        cs = [c for c in clips if c["vid_idx"] == v]
        picks = [cs[int(i)] for i in np.linspace(0, len(cs) - 1, cols)]
        for ax, c in zip(axes[r], picks):
            ax.imshow(Image.open(c["frames"][len(c["frames"]) // 2]))
            ax.set_title(f"vid{v} | {fmt_time(c['start'])} | {'speech' if c['has_speech'] else 'silent'}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Sample Frames from the Video Corpus")
    return save_fig("Q2_frame_samples", fig)


def plot_index_space(index: VideoRAGIndex):
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vid = np.array([c["vid_idx"] for c in index.clips])
    sp = np.array([c["has_speech"] for c in index.clips])
    for ax, emb, ttl in ((axes[0], index.text_emb, "Text embedding (caption + transcript, bge-m3)"),
                         (axes[1], index.vis_emb, "Visual embedding (Chinese-CLIP frames)")):
        Z = PCA(n_components=2, random_state=SEED).fit_transform(emb)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=vid, cmap="tab20", s=np.where(sp, 55, 25), edgecolor="k", linewidth=0.3)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=axes, label="video index (large marker = clip with speech)")
    fig.suptitle("PCA of Clip Embeddings")
    return save_fig("Q2_clip_embedding_space", fig)


def plot_fusion_scores(results, name="Q2_demo_fusion_scores"):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), sharey=True)
    for ax, (i, r) in zip(np.atleast_1d(axes), enumerate(results, 1)):
        labels = [f"vid{h['clip']['vid_idx']}@{fmt_time(h['clip']['start'])}" for h in r["refs"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, [h["z_text"] for h in r["refs"]], 0.4, label="text z-score", color="#4C78A8")
        ax.bar(x + 0.2, [h["z_vis"] for h in r["refs"]], 0.4, label="visual z-score", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_title(f"Turn {i}")
        if i == 1:
            ax.legend(fontsize=8)
    fig.suptitle("Text vs Visual Evidence of the Retrieved Clips")
    return save_fig(name, fig)


def plot_reference_sheet(results, name="Q2_multiturn_reference_frames", max_refs: int = 3):
    n = len(results)
    fig, axes = plt.subplots(n, max_refs, figsize=(3.8 * max_refs, 2.5 * n))
    axes = np.atleast_2d(axes)
    for r, res in enumerate(results):
        for cidx in range(max_refs):
            ax = axes[r, cidx]
            ax.axis("off")
            if cidx < len(res["refs"]):
                h = res["refs"][cidx]
                c = h["clip"]
                ax.imshow(Image.open(c["frames"][len(c["frames"]) // 2]))
                ax.set_title(f"Turn {r + 1} | V{h['rank']} | vid{c['vid_idx']} {fmt_time(c['start'])}-{fmt_time(c['end'])}"
                             f"{' *' if h['cited'] else ''}", fontsize=8)
    fig.suptitle("Reference Clips per Turn (* = cited in the answer)")
    return save_fig(name, fig)


# =============================================================================
# 7. Gradio UI
# =============================================================================
def launch_ui(chat: VideoRAGChat, share: bool = False, port: int = 7861, inline: bool = False):
    import gradio as gr

    def refs_md(refs):
        if not refs:
            return "_No references._"
        out = []
        for h in refs:
            c = h["clip"]
            out.append(f"**[V{h['rank']}] {c['video_id']} · {fmt_time(c['start'])}–{fmt_time(c['end'])}** "
                       f"{'✅ cited' if h['cited'] else '▫️'}  \nfused {h['score']:.2f} "
                       f"(text z {h['z_text']:.2f} / visual z {h['z_vis']:.2f})  \n"
                       f"🖼 {c['caption']}  \n🎙 {c['transcript'] or '（no speech）'}")
        return "\n\n---\n".join(out)

    with gr.Blocks(title="Q2 Multi-turn VideoRAG") as demo:
        gr.Markdown("# 🎬 Multi-turn VideoRAG Chatbot\nText (caption + ASR) × Visual (Chinese-CLIP) dual retrieval · Qwen2.5-VL")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = make_chatbot(gr, height=430)
                msg = gr.Textbox(placeholder="問關於影片庫的問題，可以直接追問…", show_label=False)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                gr.Examples(DEMO_TURNS, inputs=msg)
            with gr.Column(scale=2):
                top_k = gr.Slider(1, 6, value=chat.top_k, step=1, label="Top-K clips")
                mode = gr.Radio(["fusion", "text", "visual"], value=chat.mode, label="Retrieval mode")
                alpha = gr.Slider(0, 1, value=chat.alpha, step=0.05, label="Fusion alpha (text weight)")
                standalone = gr.Textbox(label="Standalone query", interactive=False)
                video = gr.Video(label="Top reference clip", height=220)
                gallery = gr.Gallery(label="Reference frames", columns=3, height=170)
                refs = gr.Markdown()

        def respond(message, history, k, m, a):
            if not message.strip():
                return history, "", gr.update(), gr.update(), gr.update(), gr.update()
            out = chat.ask(message, int(k), m, float(a))
            history = (history or []) + [{"role": "user", "content": message},
                                         {"role": "assistant", "content": out["answer"]}]
            first = next((h for h in out["refs"] if h["cited"]), out["refs"][0])
            frames = [(h["clip"]["frames"][len(h["clip"]["frames"]) // 2],
                       f"V{h['rank']} {fmt_time(h['clip']['start'])}") for h in out["refs"]]
            return history, "", out["rewritten"], cut_clip(first["clip"]), frames, refs_md(out["refs"])

        def reset():
            chat.reset()
            return [], "", "", None, [], ""

        outs = [chatbot, msg, standalone, video, gallery, refs]
        msg.submit(respond, [msg, chatbot, top_k, mode, alpha], outs)
        send.click(respond, [msg, chatbot, top_k, mode, alpha], outs)
        clear.click(reset, None, outs)
    launch_blocks(demo, gr, share=share, server_port=port, inline=inline, prevent_thread_lock=True)
    return demo
