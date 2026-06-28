"""A8: Logo Masking. White rectangle over logo region.

v1 25%  (w/2, h/2)
v2 33%  (w/3, h)
v3 50%  (w/2, h)       <- the reference masking in the user's table
v4 50%  (w,   h/2)
v5 66%  (2w/3, h)
v6 75%  (w,   3h/4)
"""

from pathlib import Path

import cv2
import numpy as np

from ._base import AttackResult, BaseAttack, copy_features_except
from ._logo_bbox import detect_logo_bbox

_VARIANTS = {
    "v1": (0.5, 0.5),
    "v2": (1 / 3, 1.0),
    "v3": (0.5, 1.0),
    "v4": (1.0, 0.5),
    "v5": (2 / 3, 1.0),
    "v6": (1.0, 0.75),
}


class A8LogoMasking(BaseAttack):
    name = "A8_logo_masking"
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

        wf, hf = _VARIANTS[variant]
        x, y, bw, bh = bbox
        mw, mh = max(1, int(bw * wf)), max(1, int(bh * hf))
        rng = np.random.default_rng(42)
        mx = x + int(rng.integers(0, max(bw - mw, 1)))
        my = y + int(rng.integers(0, max(bh - mh, 1)))
        adv = img.copy()
        cv2.rectangle(adv, (mx, my), (mx + mw, my + mh), (255, 255, 255), -1)

        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "screenshot.png"), adv)
        copy_features_except(feature_dir, output_dir, skip_names={"screenshot.png"})

        return AttackResult(
            url=url, attack_name=self.name, variant=variant,
            original_path=str(src), adversarial_path=str(output_dir / "screenshot.png"),
            success=True, metadata={"bbox": bbox, "mask_fraction": wf * hf},
        )
