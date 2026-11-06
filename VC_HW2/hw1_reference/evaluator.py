#!/usr/bin/env python3
"""Standalone evaluator for the simplified DCT + VALUE/EOB codec."""
import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# The course-defined header contains only valid payload length, in bits.
HEADER = struct.Struct(">I")  # 4 bytes / 32 bits


def evaluate(original_png, reconstructed_png, bitstream_bin):
    original = np.asarray(Image.open(original_png).convert("L"), dtype=np.float64)
    reconstructed = np.asarray(Image.open(reconstructed_png).convert("L"), dtype=np.float64)
    if original.shape != reconstructed.shape:
        raise ValueError("image dimensions differ")

    h, w = original.shape
    if h == 0 or w == 0 or h % 8 or w % 8:
        raise ValueError("image dimensions must be positive multiples of 8")

    data = Path(bitstream_bin).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file shorter than 4-byte header")
    (payload_bits,) = HEADER.unpack(data[:HEADER.size])
    expected_bytes = HEADER.size + (payload_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"file size mismatch: expected {expected_bytes} bytes, got {len(data)}"
        )

    padding_bits = (-payload_bits) % 8
    if padding_bits and data[-1] & ((1 << padding_bits) - 1):
        raise ValueError("nonzero final padding bits")

    blocks = w * h // 64
    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    total_bits = len(data) * 8
    raw_coefficient_bits = blocks * 64 * 16

    print(f"Original size: {w} x {h}")
    print(f"Blocks: {blocks}")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {'inf' if math.isinf(psnr) else f'{psnr:.6f} dB'}")
    print(f"Valid payload bits: {payload_bits}")
    print(f"Header bits: {HEADER.size * 8}")
    print(f"Final padding bits: {padding_bits}")
    print(f"Total bits: {total_bits}")
    print(f"File size: {len(data)} bytes")
    print(f"BPP: {total_bits / (w * h):.6f}")
    print(f"Raw quantized coefficient bits: {raw_coefficient_bits}")
    print(f"Payload / raw coefficient ratio: {payload_bits / raw_coefficient_bits:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_png")
    parser.add_argument("reconstructed_png")
    parser.add_argument("bitstream_bin")
    args = parser.parse_args()
    evaluate(args.original_png, args.reconstructed_png, args.bitstream_bin)
