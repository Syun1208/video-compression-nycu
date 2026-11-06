from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "report" / "figures"
BLOCK_SIZE = 8
BLOCK_SCALE = 8
CORRUPTED_BLOCK = (64, 64)
RING_MARGIN = 10
ERROR_LIMIT = 32
INPUT_LIMIT = 96
BITS_LIMIT = 200
DAMAGE_LIMIT = 10
RING_COLOR = (235, 104, 52)
SEQUENTIAL_STEPS = [
    "#ffffff",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
MODE_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def hex_to_rgb(color):
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]


def sequential_image(values, limit, steps=SEQUENTIAL_STEPS):
    stops = np.array([hex_to_rgb(step) for step in steps], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, len(steps))
    scaled = np.clip(values.astype(np.float64) / limit, 0.0, 1.0)
    channels = [np.interp(scaled, positions, stops[:, channel]) for channel in range(3)]

    return Image.fromarray(np.rint(np.stack(channels, axis=-1)).astype(np.uint8))


def enlarge(image, scale):
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def mode_map(modes, colors=MODE_COLORS, scale=BLOCK_SCALE):
    palette = np.array([hex_to_rgb(color) for color in colors], dtype=np.uint8)
    return enlarge(Image.fromarray(palette[modes]), scale)


def damage_map(damage, block=CORRUPTED_BLOCK, margin=RING_MARGIN):
    image = sequential_image(damage, DAMAGE_LIMIT).convert("RGB")
    top, left = block
    box = [
        left - margin,
        top - margin,
        left + BLOCK_SIZE + margin - 1,
        top + BLOCK_SIZE + margin - 1,
    ]
    ImageDraw.Draw(image).rectangle(box, outline=RING_COLOR, width=2)

    return image


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUTPUT_DIR / "arrays_lena.npz")
    original = data["original"]
    recon_without = data["recon_without"]
    recon_with = data["recon_with"]

    images = {
        "original": Image.fromarray(original),
        "decoded_without": Image.fromarray(recon_without),
        "decoded_with": Image.fromarray(recon_with),
        "error_without": sequential_image(
            np.abs(original.astype(np.int32) - recon_without), ERROR_LIMIT
        ),
        "error_with": sequential_image(np.abs(original.astype(np.int32) - recon_with), ERROR_LIMIT),
        "input_without": sequential_image(np.abs(data["residual_without"]), INPUT_LIMIT),
        "input_with": sequential_image(np.abs(data["residual_with"]), INPUT_LIMIT),
        "bits_without": enlarge(sequential_image(data["bits_without"], BITS_LIMIT), BLOCK_SCALE),
        "bits_with": enlarge(sequential_image(data["bits_with"], BITS_LIMIT), BLOCK_SCALE),
        "modes": mode_map(data["modes"]),
        "damage_without": damage_map(data["damage_without"]),
        "damage_with": damage_map(data["damage_with"]),
    }

    for name, image in images.items():
        image.save(FIGURE_DIR / f"{name}.png")

    print("saved", len(images), "figures to", FIGURE_DIR)


if __name__ == "__main__":
    main()
