import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import decoder_HW3_314540080 as decoder
import encoder_HW3_314540080 as encoder

HW1_DIR = ROOT.parent / "VC_HW1"
IMAGES = {"lena": ROOT / "lena.png", "mandrill": HW1_DIR / "data" / "mandril_gray.png"}
OUTPUT_DIR = ROOT / "outputs"
RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
BLOCK_SIZE = 8
LEVEL_SHIFT = 128
TIMING_RUNS = 5
CORRUPTED_BLOCK = (64, 64)
CORRUPTION_DC_STEPS = 5


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def psnr(original, decoded):
    mse = float(np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def encode_image(image, use_intra):
    height, width = image.shape
    rows = height // BLOCK_SIZE
    cols = width // BLOCK_SIZE
    recon = np.zeros_like(image)
    residual_image = np.zeros(image.shape, dtype=np.int32)
    writer = encoder.BitWriter()
    flat_prediction = np.full((BLOCK_SIZE, BLOCK_SIZE), LEVEL_SHIFT, dtype=np.int32)

    modes = np.full((rows, cols), -1, dtype=np.int32)
    block_bits = np.zeros((rows, cols), dtype=np.int32)
    nonzero = np.zeros((rows, cols), dtype=np.int32)
    zigzags = np.zeros((rows * cols, 64), dtype=np.int32)
    parts = {"mode": 0, "dc": 0, "ac": 0, "eob": 0}
    zero_tokens = 0
    dc_magnitude = 0
    ac_magnitude = 0.0

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            row = y // BLOCK_SIZE
            col = x // BLOCK_SIZE
            block = image[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            start = writer.bit_count

            if use_intra:
                top, left, corner = encoder.get_reference_pixels(recon, y, x)
                mode, _, prediction = encoder.select_mode(block, top, left, corner, y > 0, x > 0)
                encoder.write_mode(writer, mode)
                modes[row, col] = mode
                parts["mode"] += encoder.MODE_BITS
            else:
                prediction = flat_prediction

            residual = block.astype(np.int32) - prediction
            _, _, quantized, zigzag = encoder.transform_block(residual)

            for position, (token, value) in enumerate(encoder.encode_block_symbols(zigzag)):
                length = len(encoder.write_symbol(writer, token, value))

                if token == "EOB":
                    parts["eob"] += length
                elif position == 0:
                    parts["dc"] += length
                else:
                    parts["ac"] += length
                    zero_tokens += int(value == 0)

            recon[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = encoder.reconstruct_block(
                prediction, quantized
            )
            residual_image[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = residual
            block_bits[row, col] = writer.bit_count - start
            nonzero[row, col] = np.count_nonzero(quantized)
            zigzags[row * cols + col] = zigzag
            dc_magnitude += abs(zigzag[0])
            ac_magnitude += float(np.abs(residual - residual.mean()).sum())

    blocks = rows * cols
    raw = writer.finish()
    file_bytes = encoder.HEADER.size + len(raw)

    return {
        "raw": encoder.HEADER.pack(writer.bit_count) + raw,
        "recon": recon,
        "residual": residual_image,
        "modes": modes,
        "block_bits": block_bits,
        "zigzags": zigzags,
        "metrics": {
            "bpp": 8 * file_bytes / image.size,
            "psnr": psnr(image, recon),
            "mse": float(np.mean((image.astype(np.float64) - recon.astype(np.float64)) ** 2)),
            "file_bytes": file_bytes,
            "payload_bits": writer.bit_count,
            "bits": parts,
            "bits_per_block": {key: value / blocks for key, value in parts.items()},
            "nonzero_per_block": float(nonzero.mean()),
            "zero_tokens_per_block": zero_tokens / blocks,
            "mean_abs_dct_input": float(np.abs(residual_image).mean()),
            "mean_abs_dct_input_without_block_mean": ac_magnitude / image.size,
            "mean_abs_quantized_dc": dc_magnitude / blocks,
        },
    }


def decode_image(zigzags, modes, shape, use_intra):
    height, width = shape
    cols = width // BLOCK_SIZE
    recon = np.zeros(shape, dtype=np.uint8)
    flat_prediction = np.full((BLOCK_SIZE, BLOCK_SIZE), LEVEL_SHIFT, dtype=np.int32)

    for y in range(0, height, BLOCK_SIZE):
        for x in range(0, width, BLOCK_SIZE):
            row = y // BLOCK_SIZE
            col = x // BLOCK_SIZE
            prediction = flat_prediction

            if use_intra:
                top, left, corner = decoder.get_reference_pixels(recon, y, x)
                prediction = decoder.predict_block(
                    int(modes[row, col]), top, left, corner, y > 0, x > 0
                )

            residual = decoder.inverse_transform_block(zigzags[row * cols + col])[-1]
            recon[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = decoder.reconstruct_block(
                prediction, residual
            )

    return recon


def corruption_study(image, clean, use_intra, block=CORRUPTED_BLOCK, dc_steps=CORRUPTION_DC_STEPS):
    height, width = image.shape
    cols = width // BLOCK_SIZE
    index = block[0] // BLOCK_SIZE * cols + block[1] // BLOCK_SIZE
    zigzags = clean["zigzags"].copy()
    zigzags[index, 0] += dc_steps
    corrupted = decode_image(zigzags, clean["modes"], image.shape, use_intra)
    difference = np.abs(corrupted.astype(np.int32) - clean["recon"].astype(np.int32))
    per_block = difference.reshape(height // BLOCK_SIZE, BLOCK_SIZE, cols, BLOCK_SIZE).max(
        axis=(1, 3)
    )

    return difference, {
        "blocks_changed": int((per_block > 0).sum()),
        "psnr_after": psnr(image, corrupted),
        "max_difference": int(difference.max()),
    }


def compare_with_hw1(image_path, baseline):
    with tempfile.TemporaryDirectory() as folder:
        binary = Path(folder) / "hw1.bin"
        decoded_png = Path(folder) / "hw1.png"
        subprocess.run(
            [sys.executable, "encoder.py", str(image_path), str(binary)],
            cwd=HW1_DIR,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "decoder.py", str(binary), str(decoded_png)],
            cwd=HW1_DIR,
            check=True,
            capture_output=True,
        )

        return {
            "bin_identical": binary.read_bytes() == baseline["raw"],
            "png_identical": bool(np.array_equal(load_gray(decoded_png), baseline["recon"])),
        }


def median_seconds(function, runs=TIMING_RUNS):
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "corrupted_block": {
            "y": CORRUPTED_BLOCK[0],
            "x": CORRUPTED_BLOCK[1],
            "dc_steps": CORRUPTION_DC_STEPS,
        },
        "images": {},
    }

    for name, path in IMAGES.items():
        image = load_gray(path)
        with_intra = encode_image(image, use_intra=True)
        without_intra = encode_image(image, use_intra=False)

        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "intra.bin"
            encoder.encode(str(path), str(binary))
            intra_matches_submission = binary.read_bytes() == with_intra["raw"]

        damage_with, corruption_with = corruption_study(image, with_intra, use_intra=True)
        damage_without, corruption_without = corruption_study(image, without_intra, use_intra=False)

        modes = with_intra["modes"]
        total_with = with_intra["block_bits"]
        total_without = without_intra["block_bits"]
        results["images"][name] = {
            "with_intra": with_intra["metrics"],
            "without_intra": without_intra["metrics"],
            "mode_counts": [int((modes == mode).sum()) for mode in range(5)],
            "blocks_cheaper_with_intra": int((total_with < total_without).sum()),
            "blocks_costlier_with_intra": int((total_with > total_without).sum()),
            "intra_bin_identical_to_submitted_encoder": intra_matches_submission,
            "baseline_vs_hw1_codec": compare_with_hw1(path, without_intra),
            "corruption": {"with_intra": corruption_with, "without_intra": corruption_without},
            "seconds": {
                "encode_with_intra": median_seconds(lambda: encode_image(image, True)),
                "encode_without_intra": median_seconds(lambda: encode_image(image, False)),
                "decode_with_intra": median_seconds(
                    lambda: decode_image(with_intra["zigzags"], modes, image.shape, True)
                ),
                "decode_without_intra": median_seconds(
                    lambda: decode_image(without_intra["zigzags"], modes, image.shape, False)
                ),
            },
        }
        np.savez_compressed(
            OUTPUT_DIR / f"arrays_{name}.npz",
            original=image,
            recon_with=with_intra["recon"],
            recon_without=without_intra["recon"],
            residual_with=with_intra["residual"],
            residual_without=without_intra["residual"],
            bits_with=total_with,
            bits_without=total_without,
            modes=modes,
            damage_with=damage_with,
            damage_without=damage_without,
        )
        print(name, json.dumps(results["images"][name]), flush=True)

    RESULTS_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("saved", RESULTS_JSON)


if __name__ == "__main__":
    main()
