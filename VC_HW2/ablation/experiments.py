import importlib.util
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import decoder
import encoder
import motion
from bitstream import BitReader
from hw1_codec import read_residual_block

METHODS = ("full_search", "three_step_search", "three_step_rdo")
REFERENCE_PNG = ROOT / "images" / "one_gray.png"
CURRENT_PNG = ROOT / "images" / "two_gray.png"
RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
TIMING_RUNS = 5


def psnr(original, decoded):
    mse = float(np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def median_time(function, runs=TIMING_RUNS):
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def bit_breakdown(raw, shape):
    reader = BitReader(raw)
    height, width = shape
    counts = {"mode": 0, "mvd": 0, "residual": 0}

    def measure(key, function):
        start = reader.pos
        function()
        counts[key] += reader.pos - start

    for _ in range(0, height, 16):
        for _ in range(0, width, 16):
            mode = reader.read_bits(1)
            counts["mode"] += 1

            for _ in range(1 if mode == 1 else 4):
                measure(
                    "mvd",
                    lambda: (reader.read_signed_exp_golomb(), reader.read_signed_exp_golomb()),
                )

                if mode == 0:
                    measure("residual", lambda: read_residual_block(reader))

            if mode == 1:
                for _ in range(4):
                    measure("residual", lambda: read_residual_block(reader))

    return counts


def prediction_mad(current, reference, encoded):
    prediction = np.empty_like(current)
    index = 0

    for y in range(0, current.shape[0], 16):
        for x in range(0, current.shape[1], 16):
            candidate = encoded.macroblocks[index]

            if candidate.mode == 1:
                prediction[y : y + 16, x : x + 16] = motion.motion_compensate(
                    reference, y, x, candidate.motion_vectors[0], 16
                )
            else:
                for (oy, ox), mv in zip(encoder.SUB_BLOCK_OFFSETS, candidate.motion_vectors):
                    prediction[y + oy : y + oy + 8, x + ox : x + ox + 8] = motion.motion_compensate(
                        reference, y + oy, x + ox, mv, 8
                    )

            index += 1

    return float(np.abs(current.astype(np.int32) - prediction.astype(np.int32)).mean())


def search_points_per_macroblock(current, reference, method):
    total = 0

    for y in range(0, current.shape[0], 16):
        for x in range(0, current.shape[1], 16):
            if method == "full_search":
                total += motion.full_search(current, reference, y, x, 16)[1]
                continue

            total += motion.three_step_search(current, reference, y, x, 16)[1]

            if method == "three_step_rdo":
                for oy, ox in encoder.SUB_BLOCK_OFFSETS:
                    total += motion.three_step_search(current, reference, y + oy, x + ox, 8)[1]

    return total / ((current.shape[0] // 16) * (current.shape[1] // 16))


def zero_motion_search(current, reference, y, x, size):
    return (0, 0), 1


def zero_motion_baseline(current, reference):
    field = np.zeros((current.shape[0] // 8, current.shape[1] // 8, 2), dtype=np.int32)
    writer = encoder.BitWriter()
    reconstruction = np.empty_like(current)

    for y in range(0, current.shape[0], 16):
        for x in range(0, current.shape[1], 16):
            candidate = encoder.encode_large_mode(
                current, reference, field, y, x, zero_motion_search
            )
            writer.extend(candidate.writer)
            reconstruction[y : y + 16, x : x + 16] = candidate.reconstruction
            field = candidate.mv_field

    raw = writer.to_bytes()

    return {
        "bpp": 8 * len(raw) / current.size,
        "psnr": psnr(current, reconstruction),
        "file_bytes": len(raw),
        "payload_bits": writer.bit_count,
        "bits": bit_breakdown(raw, current.shape),
    }


def intra_baseline(current):
    spec = importlib.util.spec_from_file_location(
        "ta_hw1_encoder", ROOT / "hw1_reference" / "encoder.py"
    )
    ta_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta_encoder)

    with tempfile.TemporaryDirectory() as folder:
        binary = Path(folder) / "intra.bin"
        decoded_png = Path(folder) / "intra.png"
        ta_encoder.encode(str(CURRENT_PNG), str(binary))
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "hw1_reference" / "decoder.py"),
                str(binary),
                str(decoded_png),
                "--width",
                str(current.shape[1]),
                "--height",
                str(current.shape[0]),
            ],
            check=True,
            capture_output=True,
        )
        decoded = encoder.load_gray_image(decoded_png)
        size = binary.stat().st_size

    return {"bpp": 8 * size / current.size, "psnr": psnr(current, decoded), "file_bytes": size}


def search_range_study(current, reference, search_ranges=(7, 15, 32), size=16):
    study = {}

    for search_range in search_ranges:
        total_sad = 0
        at_edge = 0
        horizontal = {}

        for y in range(0, current.shape[0], size):
            for x in range(0, current.shape[1], size):
                mv, _ = motion.full_search(current, reference, y, x, size, search_range)
                total_sad += motion.sum_of_absolute_differences(
                    current[y : y + size, x : x + size], reference, y, x, mv
                )
                at_edge += int(max(abs(mv[0]), abs(mv[1])) == search_range)
                horizontal[mv[1]] = horizontal.get(mv[1], 0) + 1

        blocks = (current.shape[0] // size) * (current.shape[1] // size)
        study[str(search_range)] = {
            "mean_mad": total_sad / (blocks * size * size),
            "mvs_at_search_edge": at_edge,
            "most_common_dx": sorted(horizontal.items(), key=lambda item: -item[1])[:5],
        }

    return study


def mv_grid(encoded, rows, cols):
    grid = []

    for index, candidate in enumerate(encoded.macroblocks):
        grid.append(
            {
                "row": index // cols,
                "col": index % cols,
                "mode": candidate.mode,
                "mvs": [list(map(int, mv)) for mv in candidate.motion_vectors],
            }
        )

    return grid


def main():
    reference = encoder.load_gray_image(REFERENCE_PNG)
    current = encoder.load_gray_image(CURRENT_PNG)
    rows, cols = current.shape[0] // 16, current.shape[1] // 16
    results = {
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "frame": {
            "height": current.shape[0],
            "width": current.shape[1],
            "macroblocks": rows * cols,
        },
        "methods": {},
    }

    encoded_frames = {}

    for method in METHODS:
        encoded = encoder.encode_frame(reference, current, method)
        raw = encoded.writer.to_bytes()
        decoded = decoder.decode_frame(reference, raw, method)
        encoded_frames[method] = encoded
        modes = [candidate.mode for candidate in encoded.macroblocks]
        vectors = [mv for candidate in encoded.macroblocks for mv in candidate.motion_vectors]
        results["methods"][method] = {
            "psnr": psnr(current, decoded),
            "bpp": 8 * len(raw) / current.size,
            "file_bytes": len(raw),
            "payload_bits": encoded.writer.bit_count,
            "bits": bit_breakdown(raw, current.shape),
            "mode_16x16": modes.count(1),
            "mode_8x8": modes.count(0),
            "motion_vectors": len(vectors),
            "nonzero_motion_vectors": sum(1 for mv in vectors if mv != (0, 0)),
            "prediction_mad": prediction_mad(current, reference, encoded),
            "search_points_per_macroblock": search_points_per_macroblock(
                current, reference, method
            ),
            "encode_seconds": median_time(lambda: encoder.encode_frame(reference, current, method)),
            "decode_seconds": median_time(lambda: decoder.decode_frame(reference, raw, method)),
            "macroblocks": mv_grid(encoded, rows, cols),
        }
        print(
            method,
            {k: v for k, v in results["methods"][method].items() if k != "macroblocks"},
            flush=True,
        )

    full = encoded_frames["full_search"].macroblocks
    tss = encoded_frames["three_step_search"].macroblocks
    results["full_vs_tss_same_mv"] = sum(
        1 for a, b in zip(full, tss) if a.motion_vectors == b.motion_vectors
    )
    results["zero_motion"] = zero_motion_baseline(current, reference)
    results["intra_hw1"] = intra_baseline(current)
    results["full_search_range_study"] = search_range_study(current, reference)
    print(
        "same MV FS/TSS:",
        results["full_vs_tss_same_mv"],
        "zero motion:",
        results["zero_motion"],
        "intra:",
        results["intra_hw1"],
        "range study:",
        results["full_search_range_study"],
    )

    RESULTS_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("saved", RESULTS_JSON)


if __name__ == "__main__":
    main()
