import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import encoder

RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
MAGMA_LUT = Path(__file__).resolve().parent / "lut_magma.npy"
FIGURE_DIR = ROOT / "report" / "figures"
OUTPUT_DIR = ROOT / "outputs"
METHODS = ("full_search", "three_step_search", "three_step_rdo")
SCALE = 2
ARROW_GAIN = 1.0
ERROR_LIMIT = 32
CROP = (176, 136, 128)
CROP_SCALE = 3
SAME_COLOR = (80, 220, 120)
DIFFERENT_COLOR = (255, 70, 70)
GRID_COLOR = (255, 60, 60)


def upscale(gray, scale=SCALE, brightness=1.0):
    image = Image.fromarray(np.clip(gray.astype(np.float64) * brightness, 0, 255).astype(np.uint8))
    return image.resize((gray.shape[1] * scale, gray.shape[0] * scale), Image.NEAREST).convert(
        "RGB"
    )


def draw_arrow(draw, start, end, color, width=2, head=5):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    if start == end:
        draw.ellipse([start[0] - 1, start[1] - 1, start[0] + 1, start[1] + 1], fill=color)
        return

    for side in (-0.5, 0.5):
        tip = (end[0] - head * math.cos(angle + side), end[1] - head * math.sin(angle + side))
        draw.line([end, tip], fill=color, width=width)


def motion_field_image(frame, macroblocks, colors, scale=SCALE, gain=ARROW_GAIN, size=16):
    image = upscale(frame, scale, brightness=0.6)
    draw = ImageDraw.Draw(image)

    for macroblock, color in zip(macroblocks, colors):
        dy, dx = macroblock["mvs"][0]
        center = (
            (macroblock["col"] * size + size / 2) * scale,
            (macroblock["row"] * size + size / 2) * scale,
        )
        end = (center[0] + dx * scale * gain, center[1] + dy * scale * gain)
        draw_arrow(draw, center, end, color)

    return image


def mode_map_image(decoded, macroblocks, scale=SCALE, size=16):
    image = upscale(decoded, scale)
    draw = ImageDraw.Draw(image)

    for macroblock in macroblocks:
        if macroblock["mode"] == 1:
            continue

        left = macroblock["col"] * size * scale
        top = macroblock["row"] * size * scale
        right = left + size * scale - 1
        bottom = top + size * scale - 1
        middle_x = left + size * scale // 2
        middle_y = top + size * scale // 2
        draw.rectangle([left, top, right, bottom], outline=GRID_COLOR, width=2)
        draw.line([(middle_x, top), (middle_x, bottom)], fill=GRID_COLOR, width=1)
        draw.line([(left, middle_y), (right, middle_y)], fill=GRID_COLOR, width=1)

    return image


def error_map_image(current, decoded, lut, scale=SCALE, limit=ERROR_LIMIT):
    error = np.abs(current.astype(np.int32) - decoded.astype(np.int32))
    index = np.rint(np.clip(error / limit, 0, 1) * 255).astype(np.int32)
    rgb = (lut[index] * 255).astype(np.uint8)

    return Image.fromarray(rgb).resize(
        (current.shape[1] * scale, current.shape[0] * scale), Image.NEAREST
    )


def crop_image(frame, crop=CROP, scale=CROP_SCALE):
    top, left, size = crop
    return upscale(frame[top : top + size, left : left + size], scale)


def main():
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    reference = encoder.load_gray_image(ROOT / "images" / "one_gray.png")
    current = encoder.load_gray_image(ROOT / "images" / "two_gray.png")
    decoded = {
        method: encoder.load_gray_image(OUTPUT_DIR / method / "decoded.png") for method in METHODS
    }
    decoded_rdo = decoded["three_step_rdo"]
    lut = np.load(MAGMA_LUT)

    full = results["methods"]["full_search"]["macroblocks"]
    tss = results["methods"]["three_step_search"]["macroblocks"]
    rdo = results["methods"]["three_step_rdo"]["macroblocks"]
    tss_colors = [
        SAME_COLOR if a["mvs"] == b["mvs"] else DIFFERENT_COLOR for a, b in zip(full, tss)
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    motion_field_image(current, full, [SAME_COLOR] * len(full)).save(
        FIGURE_DIR / "mv_full_search.png"
    )
    motion_field_image(current, tss, tss_colors).save(FIGURE_DIR / "mv_three_step_search.png")
    mode_map_image(decoded_rdo, rdo).save(FIGURE_DIR / "rdo_modes.png")
    upscale(reference).save(FIGURE_DIR / "reference.png")
    upscale(current).save(FIGURE_DIR / "current.png")
    crop_image(current).save(FIGURE_DIR / "crop_current.png")

    for method, frame in decoded.items():
        upscale(frame).save(FIGURE_DIR / f"decoded_{method}.png")
        error_map_image(current, frame, lut).save(FIGURE_DIR / f"error_{method}.png")
        crop_image(frame).save(FIGURE_DIR / f"crop_{method}.png")

    print("saved figures to", FIGURE_DIR)


if __name__ == "__main__":
    main()
