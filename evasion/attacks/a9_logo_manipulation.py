"""A9: Logo Manipulation. Geometric/color transforms on the logo region.

v1 rotate 15   (user example)
v2 rotate 30
v3 flip horizontal
v4 hue shift +90
v5 grayscale
v6 aspect stretch 1.5x (horizontal)
"""

from pathlib import Path

import cv2
import numpy as np

from ._base import AttackResult, BaseAttack, copy_features_except
from ._logo_bbox import detect_logo_bbox


def _bg_color(logo: np.ndarray) -> tuple[int, int, int]:
    vals = np.median(logo[:3, :, :].reshape(-1, 3), axis=0).astype(np.uint8)
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _rotate(logo, deg):
    h, w = logo.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(logo, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=_bg_color(logo))


def _v1(logo): return _rotate(logo, 15)
def _v2(logo): return _rotate(logo, 30)
def _v3(logo): return cv2.flip(logo, 1)
def _v4(logo):
    hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 90) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def _v5(logo):
    g = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def _v6(logo):
    h, w = logo.shape[:2]
    s = cv2.resize(logo, (int(w * 1.5), h), interpolation=cv2.INTER_LINEAR)
    off = (s.shape[1] - w) // 2
    return s[:, off:off + w]


_VARIANTS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4, "v5": _v5, "v6": _v6}


class A9LogoManipulation(BaseAttack):
    name = "A9_logo_manipulation"
    surface = "logo"
    target_file = "screenshot.png"

    def get_variants(self) -> list[str]:
        return list(_VARIANTS.keys())

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

        x, y, bw, bh = bbox
        crop = img[y:y + bh, x:x + bw].copy()
        if crop.size == 0:
            return AttackResult(
                url=url, attack_name=self.name, variant=variant,
                original_path=str(src), adversarial_path="", success=False,
                metadata={"error": "empty crop"},
            )
        mod = _VARIANTS[variant](crop)
        if mod.shape != crop.shape:
            mod = cv2.resize(mod, (bw, bh), interpolation=cv2.INTER_LINEAR)
        adv = img.copy()
        adv[y:y + bh, x:x + bw] = mod

        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "screenshot.png"), adv)
        copy_features_except(feature_dir, output_dir, skip_names={"screenshot.png"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "screenshot.png"),
            success=True, metadata={"bbox": bbox, "transform": variant},
        )
