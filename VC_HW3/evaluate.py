#!/usr/bin/env python3
"""HW3 evaluator: read the 3-bit mode of every block directly from the bitstream.

With --decoder, the bitstream is also decoded with the functions of that decoder file,
and the 8x8 prediction of each block listed in --blocks is printed.
"""
import argparse
import importlib.util
import struct
from pathlib import Path

import numpy as np

# The course-defined header contains only valid payload length, in bits.
HEADER = struct.Struct(">I")  # 4 bytes / 32 bits
MODE_NAMES = ["Vertical", "Horizontal", "DC", "Diagonal Down-Left", "Plane"]
MODE_BITS = 3
CODE_TO_CATEGORY = {
    "00": 0, "010": 1, "011": 2, "100": 3, "101": 4, "110": 5,
    "1110": 6, "11110": 7, "111110": 8, "1111110": 9,
    "11111110": 10, "111111110": 11,
}
EOB = "111111111"


class BitReader:
    def __init__(self, data, valid_bits):
        self.data = data
        self.valid_bits = valid_bits
        self.pos = 0

    def read_bits(self, count):
        if self.pos + count > self.valid_bits:
            raise ValueError("unexpected end of valid payload")
        value = 0
        for _ in range(count):
            value = (value << 1) | ((self.data[self.pos // 8] >> (7 - self.pos % 8)) & 1)
            self.pos += 1
        return value

    def skip_block(self):
        """Skip the VALUE / EOB tokens of one block."""
        pos = 0
        while pos < 64:
            bits = ""
            while True:
                bits += str(self.read_bits(1))
                if bits == EOB:
                    return
                if bits in CODE_TO_CATEGORY:
                    break
                if len(bits) >= 9:
                    raise ValueError("invalid Huffman prefix")
            self.read_bits(CODE_TO_CATEGORY[bits])
            pos += 1


def read_modes(bitstream_bin, width, height):
    data = Path(bitstream_bin).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("file shorter than 4-byte header")
    (payload_bits,) = HEADER.unpack(data[:HEADER.size])
    expected_bytes = HEADER.size + (payload_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(f"file size mismatch: expected {expected_bytes} bytes, got {len(data)}")
    padding_bits = (-payload_bits) % 8
    if padding_bits and data[-1] & ((1 << padding_bits) - 1):
        raise ValueError("nonzero final padding bits")

    reader = BitReader(data[HEADER.size:], payload_bits)
    modes = []
    for _ in range(width * height // 64):
        mode = reader.read_bits(MODE_BITS)
        if mode >= len(MODE_NAMES):
            raise ValueError(f"invalid mode {mode} at block {len(modes)}")
        modes.append(mode)
        reader.skip_block()
    if reader.pos != payload_bits:
        raise ValueError("unused valid payload bits remain")
    return modes


def decode_predictions(decoder_py, bitstream_bin, width, height, wanted):
    """Decode the bitstream with the decoder's own functions; return {block index: prediction}."""
    spec = importlib.util.spec_from_file_location("hw3_decoder", decoder_py)
    dec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dec)

    data = Path(bitstream_bin).read_bytes()
    (payload_bits,) = HEADER.unpack(data[:HEADER.size])
    reader = dec.BitReader(data[HEADER.size:], payload_bits)
    recon = np.zeros((height, width), dtype=np.uint8)
    predictions = {}
    index = 0
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            mode = dec.read_mode(reader)
            top, left, corner = dec.get_reference_pixels(recon, y, x)
            prediction = dec.predict_block(mode, top, left, corner, y > 0, x > 0)
            zigzagged, _ = dec.read_block(reader)
            residual = dec.inverse_transform_block(zigzagged)[-1]
            recon[y:y + 8, x:x + 8] = dec.reconstruct_block(prediction, residual)
            if index in wanted:
                predictions[index] = np.asarray(prediction)
            index += 1
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bitstream_bin")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--blocks", help="comma-separated block indices to print, e.g. 70,200")
    parser.add_argument("--decoder", help="your decoder .py file; also prints the prediction of each listed block")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        raise ValueError("width and height must be positive multiples of 8")

    modes = read_modes(args.bitstream_bin, args.width, args.height)
    cols = args.width // 8

    if args.blocks:
        indices = [int(b) for b in args.blocks.split(",")]
        for index in indices:
            if not 0 <= index < len(modes):
                raise ValueError(f"invalid block index: {index}")
        predictions = {}
        if args.decoder:
            predictions = decode_predictions(args.decoder, args.bitstream_bin,
                                             args.width, args.height, set(indices))
        for index in indices:
            y, x = index // cols * 8, index % cols * 8
            print(f"block={index} (y={y}, x={x}) mode={modes[index]} {MODE_NAMES[modes[index]]}")
            if index in predictions:
                print("prediction:")
                for row in predictions[index]:
                    print(" ".join(f"{int(v):3d}" for v in row))
                print()
        return

    print(f"Image size: {args.width} x {args.height}")
    print(f"Blocks: {len(modes)}")
    print("Mode counts:")
    for m, name in enumerate(MODE_NAMES):
        print(f"  mode={m} {name:<18} {modes.count(m)}")
    print("Mode map (one digit per block, raster order):")
    for r in range(0, len(modes), cols):
        print("".join(str(m) for m in modes[r:r + cols]))


if __name__ == "__main__":
    main()
