"""A10: Font Substitution. Re-render logo text in a different font family.

v1 DejaVu Sans (user example)
v2 Liberation Sans
v3 Ubuntu
v4 Noto Sans
v5 FreeSans
v6 DejaVu Serif (sans vs serif contrast)

Uses OCR if pytesseract is installed; otherwise falls back to a generic brand-like word.
"""

import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._base import AttackResult, BaseAttack, copy_features_except
from ._logo_bbox import detect_logo_bbox

# Font preference per variant — try several filenames; first that exists wins.
_VARIANT_FONT_CANDIDATES = {
    "v1": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "v2": [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    "v3": [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
    ],
    "v4": [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ],
    "v5": [
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ],
    "v6": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ],
}

_FALLBACK_TEXTS = ["BRAND", "LOGO", "SIGN IN", "LOGIN", "ACCOUNT"]


def _resolve_font(variant: str) -> str | None:
    for path in _VARIANT_FONT_CANDIDATES.get(variant, []):
        if Path(path).is_file():
            return path
    # Fallback: glob any TTF from that family
    for path in _VARIANT_FONT_CANDIDATES.get(variant, []):
        parent = str(Path(path).parent) + "/"
        matches = sorted(glob.glob(parent + "*.ttf"))
        if matches:
            return matches[0]
    # Final fallback: any system font
    any_ttf = sorted(glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True))
    return any_ttf[0] if any_ttf else None


def _ocr_text(crop: np.ndarray) -> str | None:
    try:
        import pytesseract
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        txt = pytesseract.image_to_string(pil, config="--psm 7").strip()
        if 2 <= len(txt) <= 40:
            return txt
    except Exception:
        pass
    return None


def _dominant_colors(crop: np.ndarray):
    h, w = crop.shape[:2]
    corners = np.concatenate([
        crop[:2, :2].reshape(-1, 3), crop[:2, -2:].reshape(-1, 3),
        crop[-2:, :2].reshape(-1, 3), crop[-2:, -2:].reshape(-1, 3),
    ])
    bg = tuple(int(c) for c in np.median(corners, axis=0))
    diff = np.abs(crop.astype(float) - np.array(bg)).mean(axis=2)
    fg_mask = diff > 30
    if fg_mask.sum() > 0:
        fg = tuple(int(c) for c in np.median(crop[fg_mask], axis=0))
    else:
        fg = (0, 0, 0)
    return fg, bg


def _render(text: str, font_path: str, size: tuple[int, int], fg, bg) -> np.ndarray:
    w, h = size
    fg_rgb, bg_rgb = (fg[2], fg[1], fg[0]), (bg[2], bg[1], bg[0])
    img = Image.new("RGB", (w, h), bg_rgb)
    draw = ImageDraw.Draw(img)
    fs = h
    font = None
    while fs > 8:
        try:
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            font = ImageFont.load_default()
            break
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= w * 0.95 and th <= h * 0.9:
            break
        fs -= 1
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=fg_rgb, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class A10LogoFontSubst(BaseAttack):
    name = "A10_logo_font_subst"
    surface = "logo"
    target_file = "screenshot.png"

    def get_variants(self) -> list[str]:
        return list(_VARIANT_FONT_CANDIDATES.keys())

    def apply(self, feature_dir: Path, output_dir: Path, url: str, variant: str) -> AttackResult:
        src = feature_dir / "screenshot.png"
        if not src.is_file():
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "screenshot missing"},
            )
        img = cv2.imread(str(src))
        if img is None:
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "cv2 imread failed"},
            )
        bbox = detect_logo_bbox(img)
        if bbox is None:
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "no logo bbox detected"},
            )
        font_path = _resolve_font(variant)
        if not font_path:
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "no font found"},
            )

        x, y, bw, bh = bbox
        crop = img[y:y + bh, x:x + bw].copy()
        txt = _ocr_text(crop)
        if txt is None:
            rng = np.random.default_rng(abs(hash(url)) & 0xFFFFFFFF)
            txt = _FALLBACK_TEXTS[int(rng.integers(len(_FALLBACK_TEXTS)))]
        fg, bg = _dominant_colors(crop)
        rendered = _render(txt, font_path, (bw, bh), fg, bg)
        adv = img.copy()
        adv[y:y + bh, x:x + bw] = rendered

        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "screenshot.png"), adv)
        copy_features_except(feature_dir, output_dir, skip_names={"screenshot.png"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "screenshot.png"),
            success=True,
            metadata={"bbox": bbox, "font": Path(font_path).stem, "text": txt},
        )
