#!/usr/bin/env python3
import argparse
import math
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


def idct_1d(coefficients):
    """Compute one length-8 inverse DCT from the provided basis vectors."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (8,):
        raise ValueError("idct_1d requires a length-8 vector")
    result = np.empty(8, dtype=np.float64)
    for x in range(8):
        total = 0.0
        for u in range(8):
            total += DCT_BASIS[u, x] * coefficients[u]
        result[x] = total
    return result


def idct_2d_separable(coefficients):
    """Apply inverse 1D DCT to all columns and then all rows."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (8, 8):
        raise ValueError("idct_2d_separable requires an 8x8 block")
    after_columns = np.empty((8, 8), dtype=np.float64)
    result = np.empty((8, 8), dtype=np.float64)
    for column in range(8):
        after_columns[:, column] = idct_1d(coefficients[:, column])
    for row in range(8):
        result[row, :] = idct_1d(after_columns[row, :])
    return after_columns, result


def read_value(reader, s):
    if s == 0:
        return 0
    amplitude = reader.read_bits(s)
    return amplitude if amplitude >= (1 << (s - 1)) else amplitude - ((1 << s) - 1)


def read_block(reader, capture_tokens=False):
    coeffs = np.zeros(64, dtype=np.int32)
    tokens = []
    pos = 0
    while pos < 64:
        token_start = reader.pos
        s = reader.read_token()
        if s is None:
            tokens.append(("EOB", None, token_start, reader.pos))
            return coeffs, tokens
        value = read_value(reader, s)
        coeffs[pos] = value
        tokens.append(("VALUE", value, token_start, reader.pos, s))
        pos += 1
    return coeffs, tokens


def matrix(values, precision=None):
    kwargs = dict(separator=", ", max_line_width=140)
    if precision is not None:
        kwargs.update(precision=precision, floatmode="fixed")
    return np.array2string(np.asarray(values), **kwargs)


def trace_text(selected, valid_bits, payload_bytes, width, height, block_index):
    top, left, zigzagged, qf, dequantized, after_columns, spatial, output, tokens, start, end = selected
    token_lines = []
    for i, token in enumerate(tokens):
        if token[0] == "EOB":
            _, _, first, last = token
            token_lines.append(f"token[{i:02d}]=EOB                    bits={first}:{last} prefix=111111111")
        else:
            _, value, first, last, s = token
            token_lines.append(f"token[{i:02d}]=VALUE({value}) category={s} bits={first}:{last}")
    sections = [
        ("IMAGE INFORMATION", f"width={width}\nheight={height}\nblock_count={width * height // 64}\nselected_block_index={block_index}\nselected_block_top_left=(y={top}, x={left})"),
        ("HEADER", f"valid_payload_bits={valid_bits}"),
        ("DECODED VALUE / EOB TOKENS", "\n".join(token_lines)),
        ("DECODED ZIGZAG COEFFICIENTS", matrix(zigzagged)),
        ("INVERSE ZIGZAG QUANTIZED COEFFICIENTS", matrix(qf)),
        ("DEQUANTIZED DCT COEFFICIENTS", matrix(dequantized)),
        ("AFTER COLUMN-WISE 1D IDCT", matrix(after_columns, 4)),
        ("INVERSE DCT + LEVEL SHIFT", matrix(spatial, 4)),
        ("OUTPUT BLOCK", matrix(output)),
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
    blocks = args.width * args.height // 64
    if not 0 <= args.block_index < blocks:
        raise ValueError("invalid block index")

    raw = open(args.input_bin, "rb").read()
    if len(raw) < 4:
        raise ValueError("file is shorter than 4-byte header")
    valid_bits = struct.unpack(">I", raw[:4])[0]
    expected_size = 4 + (valid_bits + 7) // 8
    if len(raw) != expected_size:
        raise ValueError("file size mismatch")
    padding_bits = (-valid_bits) % 8
    if padding_bits and raw[-1] & ((1 << padding_bits) - 1):
        raise ValueError("nonzero final padding bits")
    reader = BitReader(raw[4:], valid_bits)
    reconstruction = np.empty((args.height, args.width), dtype=np.uint8)
    selected = None
    index = 0

    for top in range(0, args.height, 8):
        for left in range(0, args.width, 8):
            start = reader.pos
            zigzagged, tokens = read_block(reader)

            qf_flat = np.zeros(64, dtype=np.int32)
            qf_flat[ZIGZAG] = zigzagged

            qf = qf_flat.reshape(8, 8)
            dequantized = qf * Q

            after_first_pass = DCT_BASIS.T @ dequantized
            inverse_dct = after_first_pass @ DCT_BASIS

            spatial = inverse_dct + 128.0
            block = np.clip(np.rint(spatial), 0, 255).astype(np.uint8)

            reconstruction[top:top + 8, left:left + 8] = block

            if index == args.block_index:
                selected = (
                    top,
                    left,
                    zigzagged,
                    qf,
                    dequantized,
                    after_first_pass,
                    spatial,
                    block,
                    tokens,
                    start,
                    reader.pos,
                )

            index += 1

    if reader.pos != valid_bits:
        raise ValueError("unused valid payload bits remain")
    Image.fromarray(reconstruction, mode="L").save(args.output_png)
    if args.trace:
        text = trace_text(selected, valid_bits, len(raw) - 4, args.width, args.height, args.block_index)
        Path(args.trace).write_text(text, encoding="utf-8")
        print(text, end="")


if __name__ == "__main__":
    main()
