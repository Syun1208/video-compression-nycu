#!/usr/bin/env python3
"""HW3 decoder: VALUE/EOB decoding + IDCT + 8x8 intra prediction reconstruction."""
import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77], [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101], [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float64)

# Must be identical to the encoder's provided DCT basis vectors.
DCT_BASIS = np.array([
    [ 0.353553390593274,  0.353553390593274,  0.353553390593274,  0.353553390593274,
      0.353553390593274,  0.353553390593274,  0.353553390593274,  0.353553390593274],
    [ 0.490392640201615,  0.415734806151273,  0.277785116509801,  0.097545161008064,
     -0.097545161008064, -0.277785116509801, -0.415734806151273, -0.490392640201615],
    [ 0.461939766255643,  0.191341716182545, -0.191341716182545, -0.461939766255643,
     -0.461939766255643, -0.191341716182545,  0.191341716182545,  0.461939766255643],
    [ 0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801,
      0.277785116509801,  0.490392640201615,  0.097545161008064, -0.415734806151273],
    [ 0.353553390593274, -0.353553390593274, -0.353553390593274,  0.353553390593274,
      0.353553390593274, -0.353553390593274, -0.353553390593274,  0.353553390593274],
    [ 0.277785116509801, -0.490392640201615,  0.097545161008064,  0.415734806151273,
     -0.415734806151273, -0.097545161008064,  0.490392640201615, -0.277785116509801],
    [ 0.191341716182545, -0.461939766255643,  0.461939766255643, -0.191341716182545,
     -0.191341716182545,  0.461939766255643, -0.461939766255643,  0.191341716182545],
    [ 0.097545161008064, -0.277785116509801,  0.415734806151273, -0.490392640201615,
      0.490392640201615, -0.415734806151273,  0.277785116509801, -0.097545161008064],
], dtype=np.float64)
ZIGZAG = np.array([
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,
], dtype=np.int32)

CODE_TO_CATEGORY = {
    "00": 0, "010": 1, "011": 2, "100": 3, "101": 4, "110": 5,
    "1110": 6, "11110": 7, "111110": 8, "1111110": 9,
    "11111110": 10, "111111110": 11,
}
EOB = "111111111"

MODE_NAMES = ["Vertical", "Horizontal", "DC", "Diagonal Down-Left", "Plane"]
MODE_BITS = 3
UNAVAILABLE = 128


class BitReader:
    def __init__(self, data, valid_bits):
        if valid_bits > len(data) * 8:
            raise ValueError("valid_payload_bits exceeds file payload")
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

    def read_token(self):
        bits = ""
        for _ in range(9):
            bits += str(self.read_bits(1))
            if bits == EOB:
                return None
            if bits in CODE_TO_CATEGORY:
                return CODE_TO_CATEGORY[bits]
        raise ValueError("invalid Huffman prefix")


def read_mode(reader):
    """Read the 3-bit mode code (MSB-first) of the next block and return it as an int."""
    mode = reader.read_bits(MODE_BITS)
    if mode >= len(MODE_NAMES):
        raise ValueError(f"invalid mode: {mode}")
    return mode


# =========================================================================
# PART A - Functions from HW1 (copy your HW1 implementation)
#   Only change: inverse_transform_block (your HW1 reconstruct_block) stops
#   after the IDCT.
# =========================================================================

def read_value(reader, s):
    """Read s amplitude bits and return the signed coefficient."""

    # TODO: implement this function.
    if s == 0:
        return 0

    amplitude = reader.read_bits(s)

    if amplitude >= (1 << (s - 1)):
        return amplitude

    return amplitude - (1 << s) + 1


def read_block(reader):
    """Decode VALUE/EOB tokens into 64 zigzag coefficients.

    Return (coeffs, tokens). coeffs: int32 array of length 64.
    tokens: ("VALUE", value, start_bit, end_bit, s) or ("EOB", None, start_bit, end_bit).
    """
    # TODO: implement this function.
    coeffs = np.zeros(64, dtype=np.int32)
    tokens = []
    index = 0

    while index < 64:
        start_bit = reader.pos
        category = reader.read_token()

        if category is None:
            tokens.append(("EOB", None, start_bit, reader.pos))
            break

        value = read_value(reader, category)
        coeffs[index] = value
        tokens.append(("VALUE", value, start_bit, reader.pos, category))
        index += 1

    return coeffs, tokens


def inverse_transform_block(zigzagged, C=DCT_BASIS):
    """Inverse zigzag, dequantization and IDCT.

    Return (qf, dequantized, after_columns, residual):
        qf            = 8x8 int32 quantized coefficients (inverse zigzag)
        dequantized   = qf * Q
        after_columns = C.T @ dequantized   (1D IDCT on every column)
        residual      = after_columns @ C   (1D IDCT on every row), float
    """
    # TODO: implement this function.
    flat = np.zeros(64, dtype=np.int32)
    flat[ZIGZAG] = zigzagged
    qf = flat.reshape(8, 8)

    dequantized = qf * Q
    after_columns = C.T @ dequantized
    residual = after_columns @ C

    return qf, dequantized, after_columns, residual


# =========================================================================
# PART B - New in HW3: intra prediction
# =========================================================================

def get_reference_pixels(recon, y, x):
    """Same rules as the encoder: return (top[16], left[8], corner).

    block row y == 0 -> top = 128; block column x == 0 -> left = 128;
    y == 0 or x == 0 -> corner = 128; top-right not in image -> top[8..15] = top[7].
    """
    # TODO: implement this function.
    top = np.full(16, UNAVAILABLE, dtype=np.int32)
    left = np.full(8, UNAVAILABLE, dtype=np.int32)
    corner = UNAVAILABLE

    if y > 0:
        row_above = recon[y - 1, x:x + 16].astype(np.int32)
        top[:row_above.size] = row_above
        top[row_above.size:] = row_above[-1]

    if x > 0:
        left[:] = recon[y:y + 8, x - 1]

    if y > 0 and x > 0:
        corner = int(recon[y - 1, x - 1])

    return top, left, corner


def dc_prediction_value(top, left, has_top, has_left):
    top_sum = int(top[:8].sum())
    left_sum = int(left[:8].sum())

    if has_top and has_left:
        return (top_sum + left_sum + 8) >> 4

    if has_top:
        return (top_sum + 4) >> 3

    if has_left:
        return (left_sum + 4) >> 3

    return UNAVAILABLE


def diagonal_down_left_prediction(top):
    diagonal_values = np.empty(15, dtype=np.int32)
    diagonal_values[:14] = (top[:14] + 2 * top[1:15] + top[2:16] + 2) >> 2
    diagonal_values[14] = (top[14] + 3 * top[15] + 2) >> 2

    rows, columns = np.indices((8, 8))

    return diagonal_values[rows + columns]


def plane_gradient(samples, corner):
    samples_with_corner = [int(corner)] + [int(value) for value in samples[:8]]

    return sum(
        k * (samples_with_corner[4 + k] - samples_with_corner[4 - k])
        for k in range(1, 5)
    )


def plane_prediction(top, left, corner):
    a = 16 * (int(top[7]) + int(left[7]))
    b = (17 * plane_gradient(top, corner) + 16) >> 5
    c = (17 * plane_gradient(left, corner) + 16) >> 5

    rows, columns = np.indices((8, 8))
    surface = (a + b * (columns - 3) + c * (rows - 3) + 16) >> 5

    return np.clip(surface, 0, 255).astype(np.int32)


def predict_block(mode, top, left, corner, has_top, has_left):
    """Same as the encoder: return the 8x8 int32 prediction of `mode` (DC uses has_top / has_left)."""
    # TODO: implement this function.
    top = np.asarray(top, dtype=np.int32)
    left = np.asarray(left, dtype=np.int32)

    if mode == 0:
        return np.tile(top[:8], (8, 1))

    if mode == 1:
        return np.tile(left[:8].reshape(8, 1), (1, 8))

    if mode == 2:
        return np.full(
            (8, 8),
            dc_prediction_value(top, left, has_top, has_left),
            dtype=np.int32,
        )

    if mode == 3:
        return diagonal_down_left_prediction(top)

    if mode == 4:
        return plane_prediction(top, left, corner)

    raise ValueError(f"invalid mode: {mode}")


def reconstruct_block(prediction, residual):
    """Return clip(rint(prediction + residual), 0, 255) as a uint8 8x8 block."""
    # TODO: implement this function.
    return np.clip(np.rint(prediction + residual), 0, 255).astype(np.uint8)

# =========================================================================
# Do not modify.
# =========================================================================


def matrix(values, precision=None):
    kwargs = dict(separator=", ", max_line_width=140)
    if precision is not None:
        kwargs.update(precision=precision, floatmode="fixed")
    return np.array2string(np.asarray(values), **kwargs)


def trace_text(sel, valid_bits, payload_bytes, width, height, block_index):
    token_lines = [f"token[00]=MODE({sel['mode']})               bits={sel['mode_start']}:{sel['mode_end']} code={sel['mode']:0{MODE_BITS}b}"]
    for i, token in enumerate(sel["tokens"], start=1):
        if token[0] == "EOB":
            _, _, first, last = token
            token_lines.append(f"token[{i:02d}]=EOB                    bits={first}:{last} prefix=111111111")
        else:
            _, value, first, last, s = token
            token_lines.append(f"token[{i:02d}]=VALUE({value}) category={s} bits={first}:{last}")
    start, end = sel["mode_start"], sel["end"]
    sections = [
        ("IMAGE INFORMATION", f"width={width}\nheight={height}\nblock_count={width * height // 64}\nselected_block_index={block_index}\nselected_block_top_left=(y={sel['top_y']}, x={sel['left_x']})"),
        ("HEADER", f"valid_payload_bits={valid_bits}"),
        ("DECODED MODE", f"mode={sel['mode']}\nname={MODE_NAMES[sel['mode']]}\ncode={sel['mode']:0{MODE_BITS}b}"),
        ("REFERENCE PIXELS", f"top={matrix(sel['top'])}\nleft={matrix(sel['left'])}\ncorner={int(sel['corner'])}\ntop_available={int(sel['top_y'] > 0)}\nleft_available={int(sel['left_x'] > 0)}"),
        ("PREDICTION BLOCK", matrix(sel["prediction"])),
        ("DECODED MODE / VALUE / EOB TOKENS", "\n".join(token_lines)),
        ("DECODED ZIGZAG COEFFICIENTS", matrix(sel["zigzagged"])),
        ("INVERSE ZIGZAG QUANTIZED COEFFICIENTS", matrix(sel["qf"])),
        ("DEQUANTIZED DCT COEFFICIENTS", matrix(sel["dequantized"])),
        ("AFTER COLUMN-WISE 1D IDCT", matrix(sel["after_columns"], 4)),
        ("DECODED RESIDUAL", matrix(sel["residual"], 4)),
        ("OUTPUT BLOCK", matrix(sel["block"])),
        ("SELECTED BLOCK BITS", f"start_bit={start}\nend_bit={end}\nbit_count={end - start}"),
        ("BIT COUNTS", f"header_bits=32\nvalid_payload_bits={valid_bits}\nfinal_padding_bits={(-valid_bits) % 8}\ntotal_bits={(4 + payload_bytes) * 8}\nfile_size_bytes={4 + payload_bytes}"),
    ]
    return "\n\n".join(f"===== DECODER {name} =====\n{body}" for name, body in sections) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_bin")
    parser.add_argument("output_png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--trace", help="write a detailed trace for one block")
    parser.add_argument("--block-index", type=int, default=0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        raise ValueError("width and height must be positive multiples of 8")
    if not 0 <= args.block_index < args.width * args.height // 64:
        raise ValueError("invalid block index")

    raw = open(args.input_bin, "rb").read()
    if len(raw) < 4:
        raise ValueError("file is shorter than 4-byte header")
    valid_bits = struct.unpack(">I", raw[:4])[0]
    if len(raw) != 4 + (valid_bits + 7) // 8:
        raise ValueError("file size mismatch")
    padding_bits = (-valid_bits) % 8
    if padding_bits and raw[-1] & ((1 << padding_bits) - 1):
        raise ValueError("nonzero final padding bits")
    reader = BitReader(raw[4:], valid_bits)
    recon = np.zeros((args.height, args.width), dtype=np.uint8)
    sel = None
    index = 0

    for y in range(0, args.height, 8):
        for x in range(0, args.width, 8):
            mode_start = reader.pos
            mode = read_mode(reader)
            mode_end = reader.pos
            top, left, corner = get_reference_pixels(recon, y, x)
            has_top, has_left = y > 0, x > 0
            prediction = predict_block(mode, top, left, corner, has_top, has_left)
            zigzagged, tokens = read_block(reader)
            qf, dequantized, after_columns, residual = inverse_transform_block(zigzagged)
            block = reconstruct_block(prediction, residual)
            recon[y:y + 8, x:x + 8] = block
            if index == args.block_index:
                sel = dict(top_y=y, left_x=x, mode=mode, mode_start=mode_start, mode_end=mode_end,
                           top=top, left=left, corner=corner, prediction=prediction, tokens=tokens,
                           zigzagged=zigzagged, qf=qf, dequantized=dequantized,
                           after_columns=after_columns, residual=residual, block=block, end=reader.pos)
            index += 1

    if reader.pos != valid_bits:
        raise ValueError("unused valid payload bits remain")
    Image.fromarray(recon, mode="L").save(args.output_png)
    if args.trace:
        text = trace_text(sel, valid_bits, len(raw) - 4, args.width, args.height, args.block_index)
        Path(args.trace).write_text(text, encoding="utf-8")
        print(text, end="")


if __name__ == "__main__":
    main()
