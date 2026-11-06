import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import decoder_HW3_314540080 as decoder
import encoder_HW3_314540080 as encoder
import evaluate as ta_evaluate

ENCODER_PY = ROOT / "encoder_HW3_314540080.py"
DECODER_PY = ROOT / "decoder_HW3_314540080.py"
MANDRILL_PNG = ROOT.parent / "VC_HW1" / "data" / "mandril_gray.png"
HW1_LENA_PNG = ROOT.parent / "VC_HW1" / "data" / "lena_gray.png"
TIE_TOLERANCE = 1e-9
RANDOM_CASES = 20000

SPEC_VALUE_PREFIX = {
    0: "00",
    1: "010",
    2: "011",
    3: "100",
    4: "101",
    5: "110",
    6: "1110",
    7: "11110",
    8: "111110",
    9: "1111110",
    10: "11111110",
    11: "111111110",
}
SPEC_EOB = "111111111"


def spec_reference_pixels(recon, y, x):
    width = len(recon[0])

    if y == 0:
        top = [128] * 16
    else:
        top = [recon[y - 1][x + i] if x + i < width else recon[y - 1][x + 7] for i in range(16)]

    left = [128] * 8 if x == 0 else [recon[y + i][x - 1] for i in range(8)]
    corner = 128 if y == 0 or x == 0 else recon[y - 1][x - 1]

    return top, left, corner


def spec_prediction(mode, top, left, corner, has_top, has_left):
    def with_corner(values, index):
        return corner if index == -1 else values[index]

    horizontal = sum(k * (with_corner(top, 3 + k) - with_corner(top, 3 - k)) for k in range(1, 5))
    vertical = sum(k * (with_corner(left, 3 + k) - with_corner(left, 3 - k)) for k in range(1, 5))
    a = 16 * (top[7] + left[7])
    b = (17 * horizontal + 16) >> 5
    c = (17 * vertical + 16) >> 5

    prediction = [[0] * 8 for _ in range(8)]

    for row in range(8):
        for column in range(8):
            if mode == 0:
                value = top[column]
            elif mode == 1:
                value = left[row]
            elif mode == 2:
                if has_top and has_left:
                    value = (sum(top[:8]) + sum(left) + 8) >> 4
                elif has_top:
                    value = (sum(top[:8]) + 4) >> 3
                elif has_left:
                    value = (sum(left) + 4) >> 3
                else:
                    value = 128
            elif mode == 3:
                d = column + row

                if row == 7 and column == 7:
                    value = (top[14] + 3 * top[15] + 2) >> 2
                else:
                    value = (top[d] + 2 * top[d + 1] + top[d + 2] + 2) >> 2
            else:
                value = max(0, min(255, (a + b * (column - 3) + c * (row - 3) + 16) >> 5))

            prediction[row][column] = value

    return prediction


def spec_symbol_bits(value):
    category = abs(value).bit_length()

    if category == 0:
        return SPEC_VALUE_PREFIX[0]

    amplitude = value if value > 0 else value + 2**category - 1

    return SPEC_VALUE_PREFIX[category] + format(amplitude, f"0{category}b")


def spec_block_bits(values):
    last = max((i for i, value in enumerate(values) if value != 0), default=-1)
    bits = "".join(spec_symbol_bits(value) for value in values[: last + 1])

    return bits + (SPEC_EOB if last < 63 else "")


def spec_encode(image, order=encoder.ZIGZAG):
    height, width = image.shape
    pixels = image.tolist()
    recon = [[0] * width for _ in range(height)]
    payload = []
    modes = []

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = [row[x : x + 8] for row in pixels[y : y + 8]]
            top, left, corner = spec_reference_pixels(recon, y, x)
            best_error, best_mode, best_prediction = None, None, None

            for mode in range(5):
                prediction = spec_prediction(mode, top, left, corner, y > 0, x > 0)
                error = sum(
                    (block[i][j] - prediction[i][j]) ** 2 for i in range(8) for j in range(8)
                )

                if best_error is None or error < best_error:
                    best_error, best_mode, best_prediction = error, mode, prediction

            residual = np.array(block, dtype=np.int32) - np.array(best_prediction, dtype=np.int32)
            after_rows = residual @ encoder.DCT_BASIS.T
            coefficients = encoder.DCT_BASIS @ after_rows
            quantized = np.rint(coefficients / encoder.Q).astype(np.int32)
            flat = quantized.reshape(-1).tolist()
            payload.append(format(best_mode, "03b") + spec_block_bits([flat[i] for i in order]))

            spatial = (encoder.DCT_BASIS.T @ (quantized * encoder.Q)) @ encoder.DCT_BASIS
            rebuilt = (
                np.clip(np.rint(np.array(best_prediction) + spatial), 0, 255)
                .astype(np.uint8)
                .tolist()
            )

            for i in range(8):
                recon[y + i][x : x + 8] = rebuilt[i]

            modes.append(best_mode)

    bits = "".join(payload)
    padded = bits + "0" * ((-len(bits)) % 8)
    body = bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))

    return len(bits).to_bytes(4, "big") + body, np.array(recon, dtype=np.uint8), modes


def check_prediction_functions(cases=RANDOM_CASES, seed=7):
    generator = random.Random(seed)
    mismatches = 0
    wrong_type = 0
    plane_clipped = 0

    def sample():
        kind = generator.random()

        if kind < 0.2:
            return 0

        if kind < 0.4:
            return 255

        return generator.randint(0, 255)

    for _ in range(cases):
        top_row = [sample() for _ in range(16)]
        left_column = [sample() for _ in range(8)]
        corner = sample()
        has_top = generator.random() < 0.8
        has_left = generator.random() < 0.8

        for mode in range(5):
            expected = spec_prediction(mode, top_row, left_column, corner, has_top, has_left)

            for module in (encoder, decoder):
                result = module.predict_block(
                    mode,
                    np.array(top_row, dtype=np.int32),
                    np.array(left_column, dtype=np.int32),
                    corner,
                    has_top,
                    has_left,
                )

                if result.tolist() != expected:
                    mismatches += 1

                if result.shape != (8, 8) or result.dtype != np.int32:
                    wrong_type += 1

            if mode == 4 and (min(map(min, expected)) == 0 or max(map(max, expected)) == 255):
                plane_clipped += 1

    return {
        "cases": cases,
        "mismatches": mismatches,
        "wrong_type": wrong_type,
        "plane_cases_touching_clip": plane_clipped,
    }


def check_reference_functions(seed=11, size=64):
    recon = np.random.default_rng(seed).integers(0, 256, (size, size), dtype=np.uint8)
    lists = recon.tolist()
    mismatches = 0

    for module in (encoder, decoder):
        for y in range(0, size, 8):
            for x in range(0, size, 8):
                top, left, corner = module.get_reference_pixels(recon, y, x)
                expected_top, expected_left, expected_corner = spec_reference_pixels(lists, y, x)
                same = (
                    top.tolist() == expected_top
                    and left.tolist() == expected_left
                    and int(corner) == expected_corner
                )
                typed = (
                    top.dtype == np.int32
                    and left.dtype == np.int32
                    and top.size == 16
                    and left.size == 8
                )
                mismatches += int(not (same and typed))

    return mismatches


def synthetic_images(size=512):
    rows, columns = np.indices((size, size))
    noise = np.random.default_rng(3)

    return {
        "flat_128": np.full((size, size), 128, dtype=np.uint8),
        "flat_0": np.zeros((size, size), dtype=np.uint8),
        "flat_255": np.full((size, size), 255, dtype=np.uint8),
        "checkerboard_0_255": np.where((rows + columns) % 2, 255, 0).astype(np.uint8),
        "blocks_0_255": np.where((rows // 8 + columns // 8) % 2, 255, 0).astype(np.uint8),
        "gradient_horizontal": (columns // 2).astype(np.uint8),
        "gradient_vertical": (rows // 2).astype(np.uint8),
        "gradient_plane": ((rows + columns) // 4).astype(np.uint8),
        "noise": noise.integers(0, 256, (size, size), dtype=np.uint8),
    }


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def check_image(name, image, folder):
    png = folder / f"{name}.png"
    binary = folder / f"{name}.bin"
    decoded_png = folder / f"{name}_decoded.png"
    Image.fromarray(image).save(png)

    subprocess.run(
        [sys.executable, str(ENCODER_PY), str(png), str(binary)], check=True, capture_output=True
    )
    height, width = image.shape
    subprocess.run(
        [
            sys.executable,
            str(DECODER_PY),
            str(binary),
            str(decoded_png),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
        check=True,
        capture_output=True,
    )

    raw = binary.read_bytes()
    spec_raw, spec_recon, spec_modes = spec_encode(image)
    decoded = np.asarray(Image.open(decoded_png))
    ta_modes = ta_evaluate.read_modes(binary, width, height)

    return {
        "bytes": len(raw),
        "bin_identical_to_spec_encoder": raw == spec_raw,
        "decoded_identical_to_spec_recon": bool(np.array_equal(decoded, spec_recon)),
        "ta_evaluator_modes_match": ta_modes == spec_modes,
        "mode_counts": [spec_modes.count(mode) for mode in range(5)],
    }


def encode_with_forward(image, forward):
    height, width = image.shape
    recon = np.zeros_like(image)
    writer = encoder.BitWriter()
    modes = []
    predictions = []
    near_ties = 0

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = image[y : y + 8, x : x + 8]
            top, left, corner = encoder.get_reference_pixels(recon, y, x)
            mode, _, prediction = encoder.select_mode(block, top, left, corner, y > 0, x > 0)
            coefficients = forward(block.astype(np.int32) - prediction)
            ratio = coefficients / encoder.Q
            near_ties += int((np.abs(np.abs(ratio - np.floor(ratio)) - 0.5) < TIE_TOLERANCE).sum())
            quantized = np.rint(ratio).astype(np.int32)
            encoder.write_mode(writer, mode)
            flat = quantized.reshape(-1)

            for token, value in encoder.encode_block_symbols(
                [int(flat[i]) for i in encoder.ZIGZAG]
            ):
                encoder.write_symbol(writer, token, value)

            recon[y : y + 8, x : x + 8] = encoder.reconstruct_block(prediction, quantized)
            modes.append(mode)
            predictions.append(prediction)

    return writer.bit_count, modes, predictions, near_ties


def rounding_tie_study(image):
    basis = encoder.DCT_BASIS

    def template_order(residual):
        return basis @ (residual @ basis.T)

    def reversed_order(residual):
        return (basis @ residual) @ basis.T

    def sequential_order(residual):
        rows = residual.tolist()
        basis_rows = basis.tolist()
        after_rows = [
            [sum(w * v for w, v in zip(basis_row, row)) for basis_row in basis_rows] for row in rows
        ]
        columns = list(zip(*after_rows))

        return np.array(
            [
                [sum(w * v for w, v in zip(basis_row, column)) for basis_row in basis_rows]
                for column in columns
            ]
        ).T

    bits_a, modes_a, predictions_a, ties = encode_with_forward(image, template_order)
    study = {"exact_half_coefficients": ties, "template_order_bits": bits_a}

    for name, forward in (("reversed", reversed_order), ("sequential", sequential_order)):
        bits_b, modes_b, predictions_b, _ = encode_with_forward(image, forward)
        study[name] = {
            "bits": bits_b,
            "blocks_with_different_mode": sum(a != b for a, b in zip(modes_a, modes_b)),
            "blocks_with_different_prediction": sum(
                not np.array_equal(a, b) for a, b in zip(predictions_a, predictions_b)
            ),
        }

    return study


def main():
    report = {
        "numpy": np.__version__,
        "predict_block_random": check_prediction_functions(),
        "get_reference_pixels_mismatches": check_reference_functions(),
    }

    images = {"lena": load_gray(ROOT / "lena.png")}

    for name, path in (("hw1_lena", HW1_LENA_PNG), ("mandrill", MANDRILL_PNG)):
        if path.exists():
            images[name] = load_gray(path)

    natural = list(images)
    images.update(synthetic_images())

    with tempfile.TemporaryDirectory() as folder:
        report["images"] = {
            name: check_image(name, image, Path(folder)) for name, image in images.items()
        }

    report["rounding_tie_study"] = {name: rounding_tie_study(images[name]) for name in natural}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
