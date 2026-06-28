"""Shared logo-bbox detection for A8/A9/A10."""

from pathlib import Path

import cv2
import numpy as np


def detect_logo_bbox(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Heuristic: largest sensible contour in the top 30% of the page."""
    h, w = img.shape[:2]
    top = img[: int(h * 0.3), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    area_top = w * int(h * 0.3)
    min_a, max_a = area_top * 0.001, area_top * 0.5
    cands = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = bw * bh
        if min_a < a < max_a and bw >= 10 and bh >= 10:
            cands.append((x, y, bw, bh, a))
    if cands:
        cands.sort(key=lambda c: c[4], reverse=True)
        x, y, bw, bh, _ = cands[0]
        return (x, y, bw, bh)

    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    if bw < 10 or bh < 10:
        return None
    return (x, y, bw, bh)
