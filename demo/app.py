"""Drift-Sense chat demo: upload a zoomed-in pattern + a zoomed-out image,
the system finds the pattern (closest to centre if several) and answers in
English with the marked image.

Run: uv run python demo/app.py
"""
import os
import random
import re
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from drift_sense.dataset import load_manifest
from drift_sense.predict import predict
from drift_sense.refine import refine_subpixel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = {r["pair_id"]: r for r in load_manifest(os.path.join(ROOT, "dataset_drift_sense/manifest.csv"))}
TMP = tempfile.mkdtemp(prefix="drift_demo_")

_VLM = {}

def narrate(facts):
    """CNN measures, the VLM speaks: turn the numbers into English."""
    try:
        if "model" not in _VLM:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit")
            _VLM.update(model=model, processor=processor, generate=generate, tpl=apply_chat_template)
        prompt = _VLM["tpl"](
            _VLM["processor"], _VLM["model"].config,
            "You are the voice of a wafer alignment system. Report these "
            "measurement facts in one or two short, clear English sentences. "
            f"No extra commentary. Facts: {facts}",
            num_images=0,
        )
        out = _VLM["generate"](_VLM["model"], _VLM["processor"], prompt, max_tokens=80, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        return text.strip()
    except Exception:
        return ""

def locate(ref_path, search_path):
    """CNN + sub-pixel first; classical NCC + phase correlation as fallback."""
    out = predict(ref_path, search_path, ckpt=os.path.join(ROOT, "checkpoints/best.pt"))
    out["engine"] = "CNN + sub-pixel refine"
    if not out["found"]:
        ref_full = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        search_full = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ref_small = cv2.resize(ref_full, (100, 100), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_full, ref_small, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, (mx, my) = cv2.minMaxLoc(res)
        if maxv >= 0.5:
            x, y = refine_subpixel(ref_small, search_full, mx + 50.0, my + 50.0)
            out = {"x": x, "y": y, "confidence": float(maxv), "found": True,
                   "engine": "classical fallback (NCC + phase correlation)"}
    return out

def _save_gray(img, name):
    path = os.path.join(TMP, name)
    cv2.imwrite(path, img)
    return path

def _normalize_upload(path, name):
    """Any upload -> grayscale 1000x1000 PNG the pipeline understands."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return _save_gray(cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA), name)

def annotate(search_path, pred_xy, truth_xy=None):
    img = cv2.cvtColor(cv2.imread(search_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    if truth_xy is not None:
        cv2.circle(img, (round(truth_xy[0]), round(truth_xy[1])), 14, (0, 200, 0), 2)
    x, y = round(pred_xy[0]), round(pred_xy[1])
    cv2.drawMarker(img, (x, y), (40, 40, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (x, y), 8, (40, 40, 255), 2)
    path = os.path.join(TMP, "annotated.png")
    cv2.imwrite(path, img)
    return path

def _result_text(out, truth_xy):
    if not out["found"]:
        return (f"**No pattern found.** Best candidate scored only "
                f"{out['confidence']:.0%} — below the 50% threshold."), \
               f"pattern_found=False, best_confidence={out['confidence']:.0%}"
    lines = [f"**Pattern found at ({out['x']:.0f}, {out['y']:.0f})** in the zoomed-out "
             f"image, confidence {out['confidence']:.0%}.",
             f"Engine: {out['engine']}"]
    facts = (f"pattern_found=True, position=({out['x']:.1f}, {out['y']:.1f}) px, "
             f"confidence={out['confidence']:.0%}")
    if truth_xy is not None:
        err = ((out["x"] - truth_xy[0]) ** 2 + (out["y"] - truth_xy[1]) ** 2) ** 0.5
        lines.append(f"Error vs truth: **{err:.2f} px = {err*10:.0f} nm**")
        facts += f", error_vs_expected={err:.2f} px ({err*10:.0f} nm)"
    return "\n\n".join(lines), facts

def _paths_from_content(content):
    """Pull file paths out of a history message, whatever shape Gradio used."""
    if isinstance(content, str):
        return []
    if isinstance(content, dict):
        p = content.get("path")
        return [p] if p else []
    if isinstance(content, (tuple, list)):
        out = []
        for c in content:
            out += [c] if isinstance(c, str) and os.path.exists(c) else _paths_from_content(c)
        return out
    p = getattr(content, "path", None)
    return [p] if p else []

def respond(message, history):
    text = (message.get("text") or "").strip()
    files = [f for f in (message.get("files") or [])]

    # one image now + one in the previous message = a pair
    # (unless the text asks for patch mode: coordinates or cut/patch keywords)
    wants_patch = bool(re.search(r"\(?\s*\d{2,3}\s*[, ]\s*\d{2,3}\s*\)?|cut|patch", text, re.I))
    if len(files) == 1 and not wants_patch:
        prev = []
        for h in reversed(history or []):
            if isinstance(h, dict) and h.get("role") == "user":
                prev = _paths_from_content(h.get("content"))
                if prev:
                    break
        if prev:
            files = [prev[-1]] + files

    # "example" -> load a held-out test pair the model never saw
    m = re.search(r"example\s*(\d+)?", text, re.I)
    if not files and m:
        pid = int(m.group(1)) if m.group(1) else random.randint(451, 500)
        pid = min(max(pid, 1), 500)
        r = MANIFEST[pid]
        ref_path = os.path.join(ROOT, r["ref_path"])
        search_path = os.path.join(ROOT, r["search_path"])
        truth = (r["gt_x"], r["gt_y"])
        header = f"Test example {pid}: "
    elif len(files) >= 2:
        yield "🔎 Looking at both images…"
        a = _normalize_upload(files[0], "up_a.png")
        b = _normalize_upload(files[1], "up_b.png")
        if a is None or b is None:
            yield "I couldn't read one of those files — please attach images."
            return
        # order-free: try both ways, keep the confident one
        out_ab, out_ba = locate(a, b), locate(b, a)
        if out_ba["confidence"] > out_ab["confidence"]:
            ref_path, search_path, out = b, a, out_ba
        else:
            ref_path, search_path, out = a, b, out_ab
        truth, header = None, ""
        ann = annotate(search_path, (out["x"], out["y"]), None)
        md, facts = _result_text(out, None)
        yield {"text": header + md, "files": [ann] if out["found"] else []}
        said = narrate(facts)
        if said:
            yield {"text": f"{header}{md}\n\n🗣️ {said}", "files": [ann] if out["found"] else []}
        return
    elif len(files) == 1:
        # one image: cut a patch (at given coords or the centre), zoom it, find it back
        search_path = _normalize_upload(files[0], "up_search.png")
        if search_path is None:
            yield "I couldn't read that file — please attach an image."
            return
        cm = re.search(r"\(?\s*(\d{2,3})\s*[, ]\s*(\d{2,3})\s*\)?", text)
        cx, cy = (int(cm.group(1)), int(cm.group(2))) if cm else (500, 500)
        cx, cy = min(max(cx, 50), 950), min(max(cy, 50), 950)
        search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        patch = search[cy - 50 : cy + 50, cx - 50 : cx + 50]
        ref_path = _save_gray(cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC), "up_ref.png")
        truth = (cx, cy)
        header = f"I cut a patch at ({cx}, {cy}), zoomed it 10x, and searched for it: "
    else:
        yield ("Attach the **zoomed-in pattern** and the **zoomed-out image** "
               "(any order) and I'll find the pattern — closest to the centre if "
               "it repeats.\n\nOr try:\n- one image alone: I cut a patch from it "
               "(add coordinates like `(600, 400)` to pick the spot) and find it back\n"
               "- `example` — run a held-out test pair from the dataset")
        return

    yield "🔎 Locating the pattern…"
    out = locate(ref_path, search_path)
    ann = annotate(search_path, (out["x"], out["y"]), truth) if out["found"] else None
    md, facts = _result_text(out, truth if out["found"] else None)
    yield {"text": header + md, "files": [ann] if ann else []}
    said = narrate(facts)
    if said:
        yield {"text": f"{header}{md}\n\n🗣️ {said}", "files": [ann] if ann else []}

demo = gr.ChatInterface(
    respond,
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_count="multiple",
        file_types=["image"],
        placeholder="Attach image(s), type nothing, press send…",
    ),
    title="Drift-Sense",
    description="Give me a zoomed-in pattern and a zoomed-out image — I find the pattern "
                "(closest to the centre if it repeats) to ~0.5 px and answer in English. "
                "Red cross = my answer, green circle = the true spot (when known). "
                "Type `example` for a held-out test pair.",
)

if __name__ == "__main__":
    threading.Thread(target=lambda: narrate("warmup"), daemon=True).start()
    demo.launch()
