import cv2
import numpy as np

WIN = 100          # window size in search px; matches the 10x-downscaled reference
MAX_SHIFT = 8.0    # coarse CNN is within a few px; larger shifts are distrusted

def refine_subpixel(ref_small, search_img, x, y):
    """Sub-pixel correction of a coarse match location via phase correlation.

    ref_small: (100,100) float32 reference at search scale (10 nm/px).
    search_img: full-resolution float32 search image.
    x, y: coarse location in search px. Returns refined (x, y).
    """
    h, w = search_img.shape
    half = WIN // 2
    cx = min(max(int(round(x)), half), w - half)
    cy = min(max(int(round(y)), half), h - half)
    crop = search_img[cy - half : cy + half, cx - half : cx + half]

    win = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(
        ref_small.astype(np.float32), crop.astype(np.float32), win
    )
    # phaseCorrelate(ref, crop) reports the ref->crop displacement with the
    # sign such that crop center + (dx, dy) is the pattern's true center.
    rx, ry = cx + dx, cy + dy
    if abs(rx - x) > MAX_SHIFT or abs(ry - y) > MAX_SHIFT:
        return x, y
    return rx, ry
