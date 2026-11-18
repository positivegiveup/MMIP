"""Quiz 1 - downstream task definition, structured-output spec, system architecture diagrams."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from common import *

SLA = {"refund_return": "1 business day", "repair_warranty": "1 business day", "logistics_tracking": "4 hours",
       "technical_support": "4 hours", "billing_invoice": "1 business day", "product_inquiry": "same day",
       "complaint_escalation": "2 hours", "account_security": "15 min", "human_agent_request": "live queue"}
TEAM = {"returns_refunds": "Returns & Refunds", "repair_service": "Repair & Warranty", "logistics": "Logistics",
        "tech_support": "Tech Support", "billing": "Billing & Invoice", "sales": "Sales / Product Q&A",
        "customer_relations": "Customer Relations", "account_security": "Account Security", "human_agent": "Human Agent"}


def spec_rows():
    rows = []
    for it in IN_DOMAIN:
        rows.append([it, TEAM[INTENT_DEPT[it]], ", ".join(REQUIRED[it]) or "-",
                     "/".join(sorted({k for k in IMAGE_PREF[it] if k != "none"})) or "-",
                     {"urgent": "urgent", "high": "high", "medium": "medium", "low": "low"}[priority_for(it, ISSUES.get(it, [""])[0])],
                     SLA[it]])
    rows.append(["out_of_scope", "(none) polite redirect", "-", "-", "low", "-"])
    rows.append(["malicious", "(none) refuse + strike", "-", "-", "low", "-"])
    return rows


def show_spec_table():
    """Figure 1: downstream tasks + routing spec."""
    cols = ["intent", "downstream team", "required slots", "typical image evidence", "default priority", "SLA"]
    rows = spec_rows()
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.axis("off")
    ax.set_title("Q1 - Downstream tasks and routing spec (structured output targets)", fontsize=15, fontweight="bold", pad=12)
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colWidths=[.15, .2, .2, .22, .11, .12])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2F5D8A")
            cell.set_text_props(color="white", fontweight="bold")
        elif r >= len(rows) - 1:
            cell.set_facecolor("#FBE9E7")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F6FA")
    return save_fig(fig, "Q1_downstream_spec_table.png")


def example_payloads():
    """Print the JSON Schema and a few example payloads used as the structured-output contract."""
    print("=== JSON Schema (strict, additionalProperties=false) ===")
    print(json.dumps(SCHEMA, ensure_ascii=False, indent=2))
    ex = [make_label("repair_warranty", "TW20481733", "phone", "螢幕摔裂", "damage_photo", "medium"),
          make_label("refund_return", "", "headphones", "耳機有刮痕要退款", "damage_photo", "medium"),
          make_label("malicious")]
    for e in ex:
        print("\n--- example ---")
        print(json.dumps(e, ensure_ascii=False, indent=2))
    return ex


# --------------------------------------------------------------------------- drawing helpers
def _box(ax, x, y, w, h, text, fc="#E8F1FB", ec="#2F5D8A", fs=8.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def _arrow(ax, p1, p2, text="", color="#444", rad=0.0, fs=7.5, ls="-", tpos=0.5, dy=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(p1[0] + (p2[0] - p1[0]) * tpos, p1[1] + (p2[1] - p1[1]) * tpos + dy, text, fontsize=fs, ha="center",
                color=color, bbox=dict(fc="white", ec="none", pad=0.6, alpha=.9))


def draw_architecture():
    fig, ax = plt.subplots(figsize=(23, 13.5))
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 132)
    ax.axis("off")
    ax.set_title("Q1 - Multimodal multi-turn customer-service bot with triage: technical architecture",
                 fontsize=18, fontweight="bold")
    groups = [(1, 40, 34, 88, "1  Client", "#FFF8E1"), (38, 40, 46, 88, "2  Edge & Safety", "#FDECEA"),
              (87, 44, 52, 84, "3  Understanding & Dialog", "#E8F5E9"), (142, 44, 81, 84, "4  Validation & Dispatch", "#EDE7F6")]
    for x, y, w, h, t, c in groups:
        ax.add_patch(Rectangle((x, y), w, h, fc=c, ec="#999", lw=1, alpha=.55, zorder=0))
        ax.text(x + 1.5, y + h - 3, t, fontsize=11, fontweight="bold", color="#333")
    # client
    _box(ax, 5, 98, 27, 18, "Chat UI\ntext + image upload\nquick-reply chips\ntyping indicator", "#FFF3C4", "#B58900")
    _box(ax, 5, 72, 27, 18, "Session & consent\nsession id • locale\nPII notice", "#FFF3C4", "#B58900")
    # edge
    _box(ax, 43, 100, 38, 16, "API Gateway\nauth • rate limit • flood control\nsize / MIME limits", "#FAD7D3", "#B03A2E")
    _box(ax, 43, 60, 38, 33, "Input Guard\nL0 normalize (NFKC, zero-width)\n+ image validate/re-encode\nL1 regex rules (multi-lingual)\nL2 ML attack detector\nL3 OCR text-in-image scan\nstrike counter → cooldown",
         "#FAD7D3", "#B03A2E", fs=8.2)
    # dialog
    _box(ax, 91, 104, 44, 15, "Dialog Manager (stateful)\nslots • intent stack • parked topics\noff-topic counter • history summary", "#CDEBD0", "#2E7D32")
    _box(ax, 91, 79, 44, 19, "System 1  (fast path, CPU)\nTF-IDF/LR intent+priority • regex slots\nOOD reject • image-kind head\nconfidence score", "#CDEBD0", "#2E7D32")
    _box(ax, 91, 50, 44, 23, "System 2  (slow path, GPU)\nVLM (Qwen2.5-VL) + schema-constrained\ndecoding or SFT-LoRA\nused for: image OCR, ambiguous intent,\nmulti-intent, long context", "#CDEBD0", "#2E7D32")
    # validation / dispatch
    _box(ax, 146, 104, 33, 15, "Schema Validator\nJSON Schema • repair ≤ 2\nelse fallback → System 1", "#DCD3F2", "#5B3FA0")
    _box(ax, 146, 79, 33, 19, "Policy Checks\norder-ID lookup (Order API)\nrefund limit • dept allow-list\nPII masking", "#DCD3F2", "#5B3FA0")
    _box(ax, 146, 52, 33, 20, "Complete?\nmissing_slots == [ ]\nconfidence ≥ τ\nuser confirmed", "#FFE0B2", "#E65100", bold=True)
    _box(ax, 185, 107, 34, 12, "Dispatcher\nqueue • retry • idempotency key", "#DCD3F2", "#5B3FA0")
    teams = list(TEAM.values())
    for i, t in enumerate(teams):
        y = 98 - i * 6.0
        _box(ax, 189, y, 26, 4.8, t, "#F3EFFB", "#5B3FA0", fs=8)
    ax.text(202, 46, "downstream teams", fontsize=8.5, ha="center", style="italic", color="#5B3FA0")
    # bottom row
    _box(ax, 5, 6, 78, 26, "Response Composer\n• one targeted clarifying question at a time (from missing_slots)\n• reading-time-aware pacing: split bubbles, delay ≈ read time\n"
         "• topic bridging: ticket summary + next-step chips + parked topics\n• off-topic: short redirect, keep state • safe refusal (never echo prompt)\n• reply in user's language",
         "#E1F5FE", "#0277BD", fs=8.6)
    _box(ax, 88, 6, 62, 26, "Fallback & reliability\n• System 2 timeout / OOM → System 1 only → human handoff\n• invalid JSON → constrained decode / repair → fallback\n"
         "• Order API down → queue ticket, tell user\n• idle user → save draft + resume token", "#E1F5FE", "#0277BD", fs=8.6)
    _box(ax, 155, 6, 64, 26, "Observability & learning loop\n• trace id, latency, route (S1/S2), cost per turn\n• Benchmark regression (structure acc / latency / $)\n"
         "• human corrections + hard negatives → SFT data\n• attack log → detector retraining", "#E1F5FE", "#0277BD", fs=8.6)
    # arrows
    _arrow(ax, (32, 107), (43, 108), "message + image")
    _arrow(ax, (18.5, 98), (18.5, 90), dy=0)
    _arrow(ax, (62, 100), (62, 93))
    _arrow(ax, (81, 82), (91, 110), "clean text + sanitized image", rad=-0.15, tpos=.45)
    _arrow(ax, (113, 104), (113, 98))
    _arrow(ax, (113, 79), (113, 73), "low conf / needs image / OCR", tpos=.55, dy=0)
    _arrow(ax, (135, 90), (146, 109), "high conf", rad=.1)
    _arrow(ax, (135, 62), (146, 105), "VLM JSON", rad=-.25, tpos=.55)
    _arrow(ax, (162, 104), (162, 98))
    _arrow(ax, (162, 79), (162, 72))
    _arrow(ax, (179, 62), (190, 107), "yes → dispatch", rad=-.15, tpos=.4)
    _arrow(ax, (202, 107), (202, 103.5), dy=0)
    _arrow(ax, (146, 56), (83, 30), "no → clarify", rad=.15, tpos=.5)
    _arrow(ax, (146, 66), (135, 111), "update slots", rad=.25, color="#2E7D32", ls="--", tpos=.5)
    _arrow(ax, (46, 60), (40, 32), "blocked → refuse", color="#B03A2E", tpos=.55)
    _arrow(ax, (3, 32), (3, 98), "reply", color="#0277BD", rad=-.08, tpos=.5, dy=0)
    _arrow(ax, (113, 50), (113, 32), "timeout", color="#0277BD", ls="--", dy=0)
    _arrow(ax, (162, 52), (162, 32), "audit log", color="#0277BD", ls="--", dy=0)
    return save_fig(fig, "Q1_architecture.png")


def draw_state_machine():
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Q1 - Dialog state machine (multi-turn triage, detours and guards)", fontsize=16, fontweight="bold")
    S = {"IDLE\n(greeting)": (10, 36), "COLLECT\nintent known,\nslots missing": (38, 36), "CONFIRM\nall slots ok": (66, 36),
         "DISPATCHED\nticket created": (90, 36), "BRIDGE\nnext topic /\nparked topic": (90, 12),
         "OFF-TOPIC\ndetour": (38, 10), "BLOCKED\ncooldown": (10, 10), "HUMAN\nHANDOFF": (64, 10)}
    col = {"IDLE\n(greeting)": "#FFF3C4", "COLLECT\nintent known,\nslots missing": "#CDEBD0", "CONFIRM\nall slots ok": "#CDEBD0",
           "DISPATCHED\nticket created": "#DCD3F2", "BRIDGE\nnext topic /\nparked topic": "#E1F5FE",
           "OFF-TOPIC\ndetour": "#FFE0B2", "BLOCKED\ncooldown": "#FAD7D3", "HUMAN\nHANDOFF": "#F8BBD0"}
    for n, (x, y) in S.items():
        ax.add_patch(Circle((x, y), 6.2, fc=col[n], ec="#444", lw=1.5))
        ax.text(x, y, n, ha="center", va="center", fontsize=8.5, fontweight="bold")

    def tr(a, b, txt, rad=0.0, dy=1.5, dx=0, ls="-", color="#444"):
        (x1, y1), (x2, y2) = S[a], S[b]
        import numpy as np
        v = np.array([x2 - x1, y2 - y1], float)
        v /= np.linalg.norm(v)
        p1, p2 = (x1 + v[0] * 6.4, y1 + v[1] * 6.4), (x2 - v[0] * 6.4, y2 - v[1] * 6.4)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, txt, fontsize=8, ha="center", color=color,
                bbox=dict(fc="white", ec="none", pad=.6, alpha=.9))
    k = list(S)
    tr(k[0], k[1], "in-domain intent\n(conf ≥ τ)", dy=2)
    tr(k[1], k[2], "missing_slots = [ ]", dy=1.8)
    tr(k[2], k[3], "validator + policy OK", dy=1.8)
    tr(k[3], k[4], "show summary\n+ chips", dy=0, dx=7)
    ax.text(90, 2.5, "user picks a topic → IDLE / COLLECT", ha="center", fontsize=8, style="italic")
    tr(k[1], k[5], "off-topic / low conf", rad=.0, dx=-7, dy=0)
    tr(k[5], k[1], "re-ask pending slot", rad=.35, dx=8, dy=0, color="#2E7D32")
    tr(k[0], k[6], "attack / abuse\n(strike ≥ 1)", dy=0, dx=-8, color="#B03A2E")
    tr(k[1], k[6], "attack mid-flow", rad=-.2, dy=-2, color="#B03A2E")
    tr(k[6], k[0], "cooldown over", rad=.45, dy=-3, color="#2E7D32")
    tr(k[1], k[7], "user asks human /\n3 off-topic /\nfrustration", rad=-.25, dy=0, dx=6)
    tr(k[7], k[3], "agent gets transcript\n+ images", rad=.1, dy=-1, dx=5)
    ax.text(38, 46.2, "slot-fill turns stay here\n(order id / photo / product)", ha="center", fontsize=8, style="italic")
    ax.text(38, 28.2, "", fontsize=8)
    ax.text(2, 2, "τ: confidence threshold • strike ≥ 3 → cooldown • topic switch needs conf ≥ 0.8 (old topic is parked, not lost)",
            fontsize=9, style="italic")
    return save_fig(fig, "Q1_dialog_state_machine.png")


CORNER_CASES = [
    ("Missing required slot", "schema: missing_slots != []", "Ask ONE targeted question; keep partial slots"),
    ("Info only inside the image (receipt order ID)", "System 1: image=receipt & no regex ID", "Route to System 2 for OCR; merge into state"),
    ("Multi-intent in one message", "top-2 intent probs both ≥ 0.35", "Split into 2 tickets or ask which first; park the other"),
    ("Topic switch mid-flow", "new intent conf ≥ 0.8 and ≠ current", "Park old topic, offer to resume after dispatch"),
    ("Off-topic / no-context question", "OOD class or conf < reject τ", "Short redirect + menu; keep state; 3 strikes → offer human"),
    ("Prompt injection / jailbreak / prompt leak", "L0 normalize → L1 regex → L2 ML", "Refuse politely, never echo prompt, strike++"),
    ("Abusive / hostile user", "abuse lexicon + ML", "De-escalate, offer human agent, strike++"),
    ("Typographic attack inside image", "L3: OCR text → same text guard", "Ignore image text, ask for another photo"),
    ("Corrupt / huge / non-image file", "magic bytes, size, pixel cap, decode", "Reject + re-encode accepted images (strip EXIF)"),
    ("Blurry or irrelevant photo", "image_evidence = other / low conf", "Ask for clearer photo with an example"),
    ("Invalid / truncated JSON from model", "JSON Schema validator", "Constrained decoding → repair ≤ 2 → System 1 fallback"),
    ("Model timeout / GPU OOM", "watchdog + health check", "Degrade to System 1 → human handoff"),
    ("Hallucinated / wrong order ID", "Order API lookup (policy check)", "Never guess; ask user to re-enter"),
    ("PII in text or images", "regex masking (phone, card, ID)", "Mask in logs; minimal retention"),
    ("Flooding / spam / repeated messages", "rate limit + duplicate detector", "Throttle, cooldown, drop before model"),
    ("Language switch / code-mixed input", "language-id + multilingual patterns", "Reply in user's language; guard stays multilingual"),
    ("User goes silent mid-flow", "idle timer", "Save draft + resume token, one gentle nudge"),
    ("Frustration / explicit human request", "intent=human_agent_request / sentiment", "Warm handoff with transcript + images"),
]


def draw_corner_cases():
    fig, ax = plt.subplots(figsize=(17, 11.5))
    ax.axis("off")
    ax.set_title("Q1 - Corner-case matrix: detection and handling", fontsize=16, fontweight="bold", pad=10)
    tb = ax.table(cellText=[list(r) for r in CORNER_CASES], colLabels=["corner case", "how it is detected", "how the system responds"],
                  loc="center", cellLoc="left", colWidths=[.27, .30, .43])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1, 1.95)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2F5D8A")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F6FA")
    return save_fig(fig, "Q1_corner_case_matrix.png")
