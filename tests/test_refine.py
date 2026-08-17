import cv2
import numpy as np
from drift_sense.refine import refine_subpixel

def test_recovers_known_subpixel_shift():
    rng = np.random.default_rng(7)
    big = cv2.GaussianBlur(rng.random((300, 300)).astype(np.float32), (5, 5), 1.0)
    true_x, true_y = 150.7, 149.3
    # cut the reference window exactly at the true (sub-pixel) center
    m = np.float32([[1, 0, -(true_x - 50)], [0, 1, -(true_y - 50)]])
    ref_small = cv2.warpAffine(big, m, (100, 100), flags=cv2.INTER_LINEAR)
    # coarse guess a few px off, refinement should land within 0.3 px
    rx, ry = refine_subpixel(ref_small, big, 153.0, 146.0)
    assert abs(rx - true_x) < 0.3
    assert abs(ry - true_y) < 0.3

def test_wild_shift_falls_back_to_coarse():
    rng = np.random.default_rng(8)
    ref = rng.random((100, 100)).astype(np.float32)
    search = rng.random((1000, 1000)).astype(np.float32)  # unrelated noise
    rx, ry = refine_subpixel(ref, search, 250.0, 250.0)
    # unrelated images: refinement must not fling the answer far away
    assert abs(rx - 250.0) <= 8.0
    assert abs(ry - 250.0) <= 8.0
