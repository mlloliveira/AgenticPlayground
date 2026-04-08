# core/vision.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image, ImageDraw  # make sure Pillow is in requirements.txt

from .config import AppConfig

# Local project root (same pattern as conversations.py / state.py)
PROJECT_APP_DIR = Path(__file__).resolve().parents[1]

# Where user-uploaded images live; adapt if you already chose a different path.
UPLOADS_DIR = PROJECT_APP_DIR / "uploads"
VIT_PREVIEW_DIR = UPLOADS_DIR / "vit_previews"


def ensure_upload_dirs() -> None:
    """Create uploads + preview dirs if missing."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    VIT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def purge_uploads_dir() -> None:
    """
    Delete everything inside uploads/ (including previews),
    then recreate the folders.

    Call this ONCE per Streamlit session so old runs don't clutter the folder.
    """
    if UPLOADS_DIR.exists():
        for p in UPLOADS_DIR.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                # Best-effort cleanup; ignore failures
                pass

    ensure_upload_dirs()


def _safe_resample_mode():
    # Pillow 10+ uses Image.Resampling, older versions still expose Image.LANCZOS
    return getattr(Image, "Resampling", Image).LANCZOS


def make_vit_patch_preview(
    image_path: str | Path,
    cfg: AppConfig,
) -> Optional[Dict[str, Any]]:
    """
    Conceptual ViT-style visualization:
      - Downsample to preview_size × preview_size (default 224×224).
      - Draw vertical/horizontal lines every patch_size pixels
        (default 16 px → 14×14 patches → 13 lines each dimension).
      - Save result under uploads/vit_previews.

    Returns a small dict with metadata + preview_path, or None on error.

    NOTE: This is *conceptual* — real models may use different resolutions/patch sizes.
    """
    if not cfg.vision.get("show_patch_grid", True):
        return None

    preview_size = int(cfg.vision.get("preview_size", 224))
    patch_size = int(cfg.vision.get("patch_size", 16))

    if preview_size <= 0 or patch_size <= 0:
        return None

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    ensure_upload_dirs()

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            # Square conceptual canvas (ViT-like); keep aspect by letterboxing
            resample = _safe_resample_mode()
            img_resized = img.resize((preview_size, preview_size), resample=resample)

            draw = ImageDraw.Draw(img_resized)

            # How many patches per side and internal grid lines?
            num_cells = preview_size // patch_size  # e.g. 224//16 = 14 patches
            if num_cells < 2:
                # Patches would be meaningless
                return None

            # Draw vertical lines between patches (no extra border line)
            for i in range(1, num_cells):
                x = i * patch_size
                draw.line([(x, 0), (x, preview_size - 1)], fill=(255, 255, 255), width=1)

            # Draw horizontal lines between patches
            for j in range(1, num_cells):
                y = j * patch_size
                draw.line([(0, y), (preview_size - 1, y)], fill=(255, 255, 255), width=1)

            # Save preview next to uploads
            stem = image_path.stem
            preview_path = VIT_PREVIEW_DIR / f"{stem}_vit_grid_{preview_size}_{patch_size}.png"

            # Avoid recomputing if we already created this exact preview
            if not preview_path.exists():
                img_resized.save(preview_path, format="PNG")

            return {
                "input_path": str(image_path),
                "preview_path": str(preview_path),
                "input_size": (orig_w, orig_h),
                "preview_size": preview_size,
                "patch_size": patch_size,
                "num_patches_per_side": num_cells,      # e.g. 14
                "num_patches_total": num_cells * num_cells,  # e.g. 196
            }
    except Exception:
        # Keep the playground robust; failure just means "no preview".
        return None
