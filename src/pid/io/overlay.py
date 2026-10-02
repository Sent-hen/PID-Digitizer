"""Draw a `Drawing` (ground truth or prediction) over its source image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from pid.io.drawing import Drawing, LineType

Image.MAX_IMAGE_PIXELS = None  # dataset images are ~33 MP

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "general": (230, 75, 30),
    "tank": (240, 160, 30),
    "valve": (200, 190, 0),
    "instrumentation": (60, 170, 60),
    "pump": (0, 200, 220),
    "arrow": (60, 90, 120),
    "inlet_outlet": (30, 90, 220),
    "crossing": (150, 40, 220),
    "ankle": (230, 60, 200),
}
EDGE_COLORS = {LineType.SOLID: (20, 120, 255), LineType.NON_SOLID: (255, 40, 40)}
DEFAULT_COLOR = (128, 128, 128)


def render_overlay(drawing: Drawing, image: Image.Image | None = None,
                   max_side: int = 3000) -> Image.Image:
    """Return an RGB overlay downscaled so its longest side is at most `max_side`."""
    if image is None:
        if drawing.image_path is None:
            raise ValueError("drawing has no image_path and no image was given")
        image = Image.open(drawing.image_path)
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS)
    # Fade the drawing so annotations stand out.
    image = Image.blend(image, Image.new("RGB", image.size, (255, 255, 255)), 0.45)
    draw = ImageDraw.Draw(image, "RGBA")
    lw = max(1, round(3 * scale * 2))

    for r in drawing.ignore_regions:
        draw.rectangle([r.xmin * scale, r.ymin * scale, r.xmax * scale, r.ymax * scale],
                       fill=(0, 0, 0, 25), outline=(0, 0, 0, 90), width=1)

    by_id = drawing.symbol_by_id()
    for e in drawing.edges:
        (x0, y0), (x1, y1) = by_id[e.source].bbox.center, by_id[e.target].bbox.center
        draw.line([x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                  fill=EDGE_COLORS[e.line_type], width=lw)

    for s in drawing.symbols:
        color = CLASS_COLORS.get(s.cls, DEFAULT_COLOR)
        b = s.bbox
        if s.cls in ("crossing", "ankle"):
            cx, cy = b.center
            rad = max(2.0, 6 * scale * 2)
            draw.ellipse([cx * scale - rad, cy * scale - rad, cx * scale + rad, cy * scale + rad],
                         fill=color)
        else:
            draw.rectangle([b.xmin * scale, b.ymin * scale, b.xmax * scale, b.ymax * scale],
                           outline=color, width=lw)

    # Legend
    y = 8
    for name, color in [*CLASS_COLORS.items(), *((f"{k.value} line", c)
                                                  for k, c in EDGE_COLORS.items())]:
        draw.rectangle([8, y, 26, y + 14], fill=color)
        draw.text((32, y), name, fill=(0, 0, 0))
        y += 18
    return image


def save_overlay(drawing: Drawing, out: Path, max_side: int = 3000) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(drawing, max_side=max_side).save(out)
    return out
