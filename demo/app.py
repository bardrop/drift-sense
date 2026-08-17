"""Gradio demo: CNN + sub-pixel refine finds the pattern, Qwen3.5-4B says it in English.

Run: uv run python demo/app.py
"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from drift_sense.dataset import load_manifest
from drift_sense.predict import predict
from drift_sense.refine import refine_subpixel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "demo", "web_images")
MANIFEST = {r["pair_id"]: r for r in load_manifest(os.path.join(ROOT, "dataset_drift_sense/manifest.csv"))}
TMP = tempfile.mkdtemp(prefix="drift_demo_")

_VLM = {}

def narrate(facts):
    """CNN measures, the VLM speaks: turn the numbers into one English sentence."""
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
    except Exception as e:
        return f"(VLM narration unavailable: {e})"

def locate(ref_path, search_path):
    """CNN first; on rejection fall back to classical NCC + phase correlation.

    The CNN's drift prior is trained for 'near the nominal spot', so patches
    cut elsewhere (or non-SEM textures) get rejected. The fallback is generic.
    """
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
            out = {
                "x": x, "y": y, "confidence": float(maxv), "found": True,
                "engine": "classical fallback (NCC + phase correlation)",
                "message": f"Pattern found at ({x:.0f}, {y:.0f}) in the search image. "
                           f"Confidence: {maxv:.0%}.",
            }
    return out

def annotate(search_path, pred_xy, truth_xy=None):
    img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if truth_xy is not None:
        cv2.circle(img, (round(truth_xy[0]), round(truth_xy[1])), 14, (0, 220, 0), 2)
    x, y = round(pred_xy[0]), round(pred_xy[1])
    cv2.drawMarker(img, (x, y), (255, 40, 40), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (x, y), 8, (255, 40, 40), 2)
    return img

def report(out, truth_xy, use_vlm):
    lines = [f"**{out['message']}**", f"Engine: {out.get('engine', 'CNN + sub-pixel refine')}"]
    facts = (f"pattern_found={out['found']}, position=({out['x']:.1f}, {out['y']:.1f}) px, "
             f"confidence={out['confidence']:.0%}")
    if truth_xy is not None:
        err = ((out["x"] - truth_xy[0]) ** 2 + (out["y"] - truth_xy[1]) ** 2) ** 0.5
        lines.append(f"Error vs truth: **{err:.2f} px = {err*10:.0f} nm** (1 px = 10 nm)")
        facts += f", error_vs_expected={err:.2f} px ({err*10:.0f} nm)"
    if use_vlm:
        lines.append(f"\n🗣️ **VLM says:** {narrate(facts)}")
    return "\n\n".join(lines)

# --- Tab 1: dataset pairs -------------------------------------------------
def run_pair(pair_id, use_vlm):
    r = MANIFEST[int(pair_id)]
    ref_path = os.path.join(ROOT, r["ref_path"])
    search_path = os.path.join(ROOT, r["search_path"])
    out = locate(ref_path, search_path)
    truth = (r["gt_x"], r["gt_y"])
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    return ref_img, annotate(search_path, (out["x"], out["y"]), truth), report(out, truth, use_vlm)

# --- Tab 2: make a pair from any image ------------------------------------
def run_make_pair(image, cx, cy, use_vlm):
    if image is None:
        return None, None, "Upload or pick an image first."
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    search = cv2.resize(gray, (1000, 1000), interpolation=cv2.INTER_AREA)
    cx = int(min(max(cx, 50), 950))
    cy = int(min(max(cy, 50), 950))
    patch = search[cy - 50 : cy + 50, cx - 50 : cx + 50]
    ref = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_CUBIC)  # fake 10x zoom-in
    ref_path = os.path.join(TMP, "ref.png")
    search_path = os.path.join(TMP, "search.png")
    cv2.imwrite(ref_path, ref)
    cv2.imwrite(search_path, search)
    out = locate(ref_path, search_path)
    return ref, annotate(search_path, (out["x"], out["y"]), (cx, cy)), report(out, (cx, cy), use_vlm)

# --- Tab 3: no-match ------------------------------------------------------
def run_no_match(pair_id, image, use_vlm):
    if image is None:
        return None, "Pick a web image first."
    r = MANIFEST[int(pair_id)]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    search = cv2.resize(gray, (1000, 1000), interpolation=cv2.INTER_AREA)
    search_path = os.path.join(TMP, "search_nm.png")
    cv2.imwrite(search_path, search)
    out = locate(os.path.join(ROOT, r["ref_path"]), search_path)
    return annotate(search_path, (out["x"], out["y"])), report(out, None, use_vlm)

def _web_examples():
    return [[p] for p in sorted(glob.glob(os.path.join(WEB_DIR, "*.jpg")))]

with gr.Blocks(title="Drift-Sense Demo") as demo:
    gr.Markdown("# Drift-Sense: SEM Pattern Localization\n"
                "CNN + sub-pixel phase correlation does the measuring (~0.5 px = 5 nm). "
                "A fine-tuned Qwen3.5-4B turns the numbers into English.")
    use_vlm = gr.Checkbox(value=True, label="VLM narration (first call loads the 4B model, ~20 s)")

    with gr.Tab("Dataset pair"):
        gr.Markdown("Pairs **451–500 are held-out test data** the model never saw. "
                    "Green circle = ground truth, red cross = prediction.")
        pid = gr.Slider(1, 500, value=500, step=1, label="Pair id")
        btn1 = gr.Button("Locate", variant="primary")
        with gr.Row():
            ref_out1 = gr.Image(label="Reference (100x zoom-in)")
            search_out1 = gr.Image(label="Search (10x) with prediction")
        md1 = gr.Markdown()
        btn1.click(run_pair, [pid, use_vlm], [ref_out1, search_out1, md1])

    with gr.Tab("Any image → make a pair"):
        gr.Markdown("Pick any image. We cut a 100×100 patch at your chosen spot, zoom it "
                    "10x to fake the reference, and the model must find it. Note: the model's "
                    "drift prior was trained around (250, 250), so spots near there are its home turf.")
        img2 = gr.Image(label="Search image", type="numpy")
        gr.Examples(examples=_web_examples(), inputs=img2, label="Web images (Wikimedia Commons)")
        with gr.Row():
            cx = gr.Slider(50, 950, value=250, step=1, label="Patch centre x")
            cy = gr.Slider(50, 950, value=250, step=1, label="Patch centre y")
        btn2 = gr.Button("Cut patch and locate", variant="primary")
        with gr.Row():
            ref_out2 = gr.Image(label="Generated reference (10x patch)")
            search_out2 = gr.Image(label="Prediction (green = where we cut)")
        md2 = gr.Markdown()
        btn2.click(run_make_pair, [img2, cx, cy, use_vlm], [ref_out2, search_out2, md2])

    with gr.Tab("No match (honesty check)"):
        gr.Markdown("A dataset reference vs an unrelated web image: the model should say "
                    "**no pattern found** (confidence below 0.5).")
        pid3 = gr.Slider(1, 500, value=451, step=1, label="Reference pair id")
        img3 = gr.Image(label="Unrelated search image", type="numpy")
        gr.Examples(examples=_web_examples(), inputs=img3, label="Web images")
        btn3 = gr.Button("Try to locate", variant="primary")
        search_out3 = gr.Image(label="Best (rejected) candidate")
        md3 = gr.Markdown()
        btn3.click(run_no_match, [pid3, img3, use_vlm], [search_out3, md3])

if __name__ == "__main__":
    demo.launch()
