from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def draw_label(image: Image.Image, label: str, font_size: int = 22) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(font_size)
    pad = 6
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle((0, 0, w + pad * 2, h + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def draw_candidate_marker(image: Image.Image, label: str, font_size: int = 42) -> Image.Image:
    """Draw a high-contrast candidate marker in the center of a panorama tile."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(font_size)
    w, h = canvas.size
    marker_r = max(26, min(w, h) // 7)
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r), fill=(255, 230, 0), outline=(0, 0, 0), width=4)
    draw.text((cx, cy), label, anchor="mm", fill=(0, 0, 0), font=font)
    return canvas


def stop_tile(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    title_font = _font(30)
    body_font = _font(16)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(0, 0, 0), width=3)
    draw.text((size // 2, size // 3), "STOP", anchor="mm", fill=(0, 0, 0), font=title_font)
    draw.text((size // 2, size // 2), "Choose only if\ngoal is reached", anchor="mm", fill=(40, 40, 40), font=body_font, align="center")
    return img


def make_grid(items: List[Image.Image], cols: int, tile_size: int, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), bg)
    for idx, item in enumerate(items):
        x = (idx % cols) * tile_size
        y = (idx // cols) * tile_size
        canvas.paste(resize_square(item, tile_size), (x, y))
    return canvas


def stitch_views_to_grid(
    views: List[Image.Image],
    output_path: str,
    *,
    cols: int = 12,
    tile_size: int = 224,
    label_each: bool = True,
) -> str:
    """Resize each view to a square tile and paste into one row-major grid image."""
    if not views:
        raise ValueError("views must be non-empty")
    tiles = []
    for idx, im in enumerate(views):
        tile = resize_square(im, tile_size)
        if label_each:
            tile = draw_label(tile, "view_{:02d}".format(idx))
        tiles.append(tile)
    grid = make_grid(tiles, cols=cols, tile_size=tile_size)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=92)
    return str(out)


def build_composite_image(
    panorama_views: List[Image.Image],
    candidate_images: Dict[str, Optional[Image.Image]],
    output_path: str,
    view_indices: Optional[List[int]] = None,
    tile_size: int = 224,
) -> str:
    """Create one composite image for choice-style VLN.

    The top grid shows the current panorama; the bottom grid shows labeled candidates.
    """
    if view_indices is None:
        view_indices = list(range(0, 36, 3))[:12]
    pano_tiles = [draw_label(panorama_views[i], "view_{:02d}".format(i)) for i in view_indices]
    pano_grid = make_grid(pano_tiles, cols=4, tile_size=tile_size)

    cand_tiles = []
    for label in sorted(candidate_images.keys(), key=lambda x: (x == "STOP", x)):
        img = stop_tile(tile_size) if label == "STOP" or candidate_images[label] is None else candidate_images[label]
        cand_tiles.append(draw_label(img, label, font_size=34))
    cand_grid = make_grid(cand_tiles, cols=4, tile_size=tile_size)

    title_h = 42
    canvas = Image.new("RGB", (max(pano_grid.width, cand_grid.width), title_h * 2 + pano_grid.height + cand_grid.height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(24)
    draw.text((8, 8), "Current Panorama", fill=(0, 0, 0), font=font)
    canvas.paste(pano_grid, (0, title_h))
    offset_y = title_h + pano_grid.height
    draw.text((8, offset_y + 8), "Candidate Views", fill=(0, 0, 0), font=font)
    canvas.paste(cand_grid, (0, offset_y + title_h))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    return str(out)


def build_marked_panorama_image(
    panorama_views: List[Image.Image],
    candidate_point_ids: Dict[str, int],
    output_path: str,
    view_indices: Optional[List[int]] = None,
    tile_size: int = 224,
) -> str:
    """Backward-compatible wrapper: full panorama grid (default 12 columns)."""
    del candidate_point_ids, view_indices
    return stitch_views_to_grid(panorama_views, output_path, cols=12, tile_size=tile_size, label_each=True)