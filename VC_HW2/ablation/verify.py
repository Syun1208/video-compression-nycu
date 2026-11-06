import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import encoder
import motion
from predictor_mvd import motion_vector_difference, mv_predictor

METHODS = ("full_search", "three_step_search", "three_step_rdo")
REFERENCE_PNG = ROOT / "images" / "one_gray.png"
CURRENT_PNG = ROOT / "images" / "two_gray.png"
OUTPUT_DIR = ROOT / "outputs"
NEAR_TIE_TOLERANCE = 1e-9


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


TA_EVALUATOR = load_module("ta_evaluator", ROOT / "provided" / "evaluator.py")
TA_HW1_DECODER = load_module("ta_hw1_decoder", ROOT / "hw1_reference" / "decoder.py")


def ta_residual_block(reader):
    zigzag = np.zeros(64, dtype=np.int32)
    position = 0

    while position < 64:
        category = reader.read_value_or_eob()

        if category is None:
            break

        zigzag[position] = TA_HW1_DECODER.read_value(reader, category)
        position += 1

    flat = np.zeros(64, dtype=np.int32)
    flat[TA_HW1_DECODER.ZIGZAG] = zigzag
    dequantized = flat.reshape(8, 8) * TA_HW1_DECODER.Q
    after_first_pass = TA_HW1_DECODER.DCT_BASIS.T @ dequantized

    return after_first_pass @ TA_HW1_DECODER.DCT_BASIS


def independent_decode(reference, raw):
    reader = TA_EVALUATOR.BitReader(raw)
    height, width = reference.shape
    field = np.zeros((height // 8, width // 8, 2), dtype=np.int32)
    spatial = np.zeros((height, width), dtype=np.float64)

    def read_mv(row, col):
        predictor = mv_predictor(field, row, col)
        return (
            int(predictor[0]) + reader.read_signed_exp_golomb(),
            int(predictor[1]) + reader.read_signed_exp_golomb(),
        )

    def predict(y, x, mv, size):
        return reference[y + mv[0] : y + mv[0] + size, x + mv[1] : x + mv[1] + size].astype(
            np.float64
        )

    for y in range(0, height, 16):
        for x in range(0, width, 16):
            if reader.read_bits(1) == 1:
                mv = read_mv(y // 8, x // 8)
                prediction = predict(y, x, mv, 16)

                for oy, ox in encoder.SUB_BLOCK_OFFSETS:
                    spatial[y + oy : y + oy + 8, x + ox : x + ox + 8] = prediction[
                        oy : oy + 8, ox : ox + 8
                    ] + ta_residual_block(reader)

                field[y // 8 : y // 8 + 2, x // 8 : x // 8 + 2] = mv
            else:
                for oy, ox in encoder.SUB_BLOCK_OFFSETS:
                    sub_y, sub_x = y + oy, x + ox
                    mv = read_mv(sub_y // 8, sub_x // 8)
                    spatial[sub_y : sub_y + 8, sub_x : sub_x + 8] = predict(
                        sub_y, sub_x, mv, 8
                    ) + ta_residual_block(reader)
                    field[sub_y // 8, sub_x // 8] = mv

    if reader.pos != reader.valid_bits:
        raise ValueError("independent decoder left unused payload bits")

    return np.clip(np.rint(spatial), 0, 255).astype(np.uint8), spatial


def brute_force_mv(current, reference, y, x, size, search_range=7):
    block = current[y : y + size, x : x + size].astype(np.int64)
    best = None

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            if not (
                0 <= y + dy <= reference.shape[0] - size
                and 0 <= x + dx <= reference.shape[1] - size
            ):
                continue

            sad = int(
                np.abs(
                    block
                    - reference[y + dy : y + dy + size, x + dx : x + dx + size].astype(np.int64)
                ).sum()
            )
            key = (sad, abs(dy) + abs(dx), dy, dx)

            if best is None or key < best:
                best = key

    return best[2], best[3]


def reference_tss(current, reference, y, x, size):
    block = current[y : y + size, x : x + size].astype(np.int64)
    center = (0, 0)

    for step in (4, 2, 1):
        best = None

        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                dy, dx = center[0] + sy * step, center[1] + sx * step

                if abs(dy) > 7 or abs(dx) > 7:
                    continue

                if not (
                    0 <= y + dy <= reference.shape[0] - size
                    and 0 <= x + dx <= reference.shape[1] - size
                ):
                    continue

                sad = int(
                    np.abs(
                        block
                        - reference[y + dy : y + dy + size, x + dx : x + dx + size].astype(np.int64)
                    ).sum()
                )
                key = (sad, abs(dy) + abs(dx), dy, dx)

                if best is None or key < best:
                    best = key

        center = (best[2], best[3])

    return center


def check_motion_search(current, reference):
    mismatches = {"full_search_16": 0, "tss_16": 0, "tss_8": 0}

    for y in range(0, current.shape[0], 16):
        for x in range(0, current.shape[1], 16):
            if motion.full_search(current, reference, y, x, 16)[0] != brute_force_mv(
                current, reference, y, x, 16
            ):
                mismatches["full_search_16"] += 1

            if motion.three_step_search(current, reference, y, x, 16)[0] != reference_tss(
                current, reference, y, x, 16
            ):
                mismatches["tss_16"] += 1

    for y in range(0, current.shape[0], 8):
        for x in range(0, current.shape[1], 8):
            if motion.three_step_search(current, reference, y, x, 8)[0] != reference_tss(
                current, reference, y, x, 8
            ):
                mismatches["tss_8"] += 1

    return mismatches


def check_spec_mvd_example():
    field = np.zeros((4, 4, 2), dtype=np.int32)
    mvds = []

    for oy, ox in encoder.SUB_BLOCK_OFFSETS:
        row, col = oy // 8, ox // 8
        predictor = mv_predictor(field, row, col)
        mvds.append(motion_vector_difference((1, 0), predictor).tolist())
        field[row, col] = (1, 0)

    return mvds == [[1, 0], [1, 0], [0, 0], [0, 0]], mvds


def check_rdo_decisions(current, reference, encoded):
    wrong = 0
    field = np.zeros_like(encoded.mv_field)
    index = 0

    for y in range(0, current.shape[0], 16):
        for x in range(0, current.shape[1], 16):
            block = current[y : y + 16, x : x + 16]
            large = encoder.encode_large_mode(
                current, reference, field, y, x, motion.three_step_search
            )
            small = encoder.encode_small_mode(
                current, reference, field, y, x, motion.three_step_search
            )
            expected = 0 if encoder.rdo_cost(block, small) < encoder.rdo_cost(block, large) else 1
            chosen = encoded.macroblocks[index]

            if chosen.mode != expected:
                wrong += 1

            field = chosen.mv_field
            index += 1

    return wrong


def check_synthetic_shift(reference, shift=(2, -3)):
    dy, dx = shift
    current = np.roll(reference, shift=(-dy, -dx), axis=(0, 1))
    found = motion.full_search(current, reference, 160, 160, 16)[0]
    flat = np.full_like(reference, 128)
    flat_mv = motion.three_step_search(flat, flat, 160, 160, 16)[0]

    return found == shift and flat_mv == (0, 0), {"shift_found": found, "flat_tie_mv": flat_mv}


def main():
    reference = encoder.load_gray_image(REFERENCE_PNG)
    current = encoder.load_gray_image(CURRENT_PNG)
    report = {"numpy": np.__version__, "methods": {}}

    for method in METHODS:
        encoded = encoder.encode_frame(reference, current, method)
        raw = (OUTPUT_DIR / method / "encoded.bin").read_bytes()
        decoded = np.asarray(Image.open(OUTPUT_DIR / method / "decoded.png"))
        independent, spatial = independent_decode(reference, raw)
        distance_to_half = np.abs(spatial - np.floor(spatial) - 0.5)
        report["methods"][method] = {
            "bin_matches_encoder": raw == encoded.writer.to_bytes(),
            "encoder_reconstruction_matches_decoder": bool(
                np.array_equal(encoded.reconstruction, decoded)
            ),
            "independent_decoder_matches_decoder": bool(np.array_equal(independent, decoded)),
            "pixels_near_rounding_tie": int((distance_to_half < NEAR_TIE_TOLERANCE).sum()),
            "closest_distance_to_half": float(distance_to_half.min()),
        }

        if method == "three_step_rdo":
            report["methods"][method]["wrong_rdo_decisions"] = check_rdo_decisions(
                current, reference, encoded
            )

    report["motion_search_mismatches"] = check_motion_search(current, reference)
    report["spec_mvd_example"] = check_spec_mvd_example()
    report["synthetic_shift_and_tie"] = check_synthetic_shift(reference)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
