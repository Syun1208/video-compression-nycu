#!/usr/bin/env python3
import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

HEADER = struct.Struct(">I")  # 4-byte unsigned big-endian valid_payload_bits
Q = np.array([
    [16,11,10,16,24,40,51,61], [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56], [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77], [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101], [72,92,95,98,112,100,103,99],
], dtype=np.float64)

# Provided 8-point orthonormal DCT-II basis vectors. Do not modify.
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
ZIGZAG = [0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,
          41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,
          23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63]

# Provided by the instructor. Do not modify.
VALUE_CODE = {
    0:(0b00,2), 1:(0b010,3), 2:(0b011,3), 3:(0b100,3), 4:(0b101,3),
    5:(0b110,3), 6:(0b1110,4), 7:(0b11110,5), 8:(0b111110,6),
    9:(0b1111110,7), 10:(0b11111110,8), 11:(0b111111110,9),
}
EOB_CODE = (0b111111111, 9)


class BitWriter:
    def __init__(self): self.data=bytearray(); self.byte=0; self.used=0; self.bit_count=0
    def write_bits(self, value, count):
        if count < 0 or value < 0 or value >= (1 << count): raise ValueError("invalid bits")
        for shift in range(count - 1, -1, -1):
            self.byte=(self.byte << 1) | ((value >> shift) & 1); self.used += 1; self.bit_count += 1
            if self.used == 8: self.data.append(self.byte); self.byte=0; self.used=0
    def finish(self):
        out=bytearray(self.data)
        if self.used: out.append(self.byte << (8-self.used))
        return bytes(out)


def matrix(value, precision=None):
    if value is None: return "[not available]"
    kwargs=dict(separator=", ", max_line_width=140)
    if precision is not None: kwargs.update(precision=precision, floatmode="fixed")
    return np.array2string(np.asarray(value), **kwargs)


class Trace:
    """Stores only values that have actually been reached during execution."""
    def __init__(self, path, block_index):
        self.path=path; self.block_index=block_index; self.data={"index":block_index}; self.error=None
    def update(self, **items): self.data.update(items)
    def fail(self, exc): self.error=f"{type(exc).__name__}: {exc}"
    def token_lines(self):
        tokens=self.data.get("tokens")
        if tokens is None: return "[not available]"
        lines=[]
        for i, (token, arg) in enumerate(tokens):
            name=token if arg is None else f"{token}({arg})"
            try: bits=token_bits_for_trace(token, arg)
            except Exception as exc: bits=f"[not available: {type(exc).__name__}]"
            lines.append(f"token[{i:02d}]={name:<20} bits={bits} length={len(bits) if bits[:1] != '[' else '?'}")
        return "\n".join(lines)
    def write(self):
        d=self.data
        info=(f'width={d.get("w", "[not available]")}\nheight={d.get("h", "[not available]")}\n'
              f'block_count={d.get("blocks", "[not available]")}\nselected_block_index={self.block_index}\n'
              f'selected_block_top_left={d.get("location", "[not available]")}')
        start=d.get("start"); end=d.get("end")
        bit_text="[not available]" if start is None or end is None else f"start_bit={start}\nend_bit={end}\nbit_count={end-start}"
        sections=[
            ("TRACE STATUS", "completed" if self.error is None else f"stopped: {self.error}"),
            ("IMAGE INFORMATION", info), ("INPUT BLOCK", matrix(d.get("block"))),
            ("LEVEL-SHIFTED BLOCK", matrix(d.get("shifted"), 1)),
            ("AFTER ROW-WISE 1D DCT", matrix(d.get("after_rows"), 4)),
            ("DCT COEFFICIENTS", matrix(d.get("F"), 4)),
            ("QUANTIZED COEFFICIENTS", matrix(d.get("QF"))),
            ("ZIGZAG COEFFICIENTS", matrix(d.get("zigzag"))),
            ("VALUE / EOB SYMBOLS", self.token_lines()),
            ("SELECTED BLOCK BITS", bit_text),
            ("HEADER", f'valid_payload_bits={d.get("payload_bits", "[not available]")}'),
            ("BIT COUNTS", f'header_bits=32\nvalid_payload_bits={d.get("payload_bits", "[not available]")}\n'
             f'final_padding_bits={d.get("padding_bits", "[not available]")}\n'
             f'total_bits={d.get("total_bits", "[not available]")}\nfile_size_bytes={d.get("file_size", "[not available]")}'),
        ]
        text="\n\n".join(f"===== ENCODER {name} =====\n{body}" for name, body in sections)+"\n"
        Path(self.path).write_text(text, encoding="utf-8"); print(text, end="")


def dct_1d(values):
    """Compute one length-8 DCT from the provided basis vectors."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (8,):
        raise ValueError("dct_1d requires a length-8 vector")
    result = np.empty(8, dtype=np.float64)
    for u in range(8):
        total = 0.0
        for x in range(8):
            total += DCT_BASIS[u, x] * values[x]
        result[u] = total
    return result


def dct_2d_separable(block):
    """Apply the same 1D DCT to all rows and then all columns."""
    block = np.asarray(block, dtype=np.float64)
    if block.shape != (8, 8):
        raise ValueError("dct_2d_separable requires an 8x8 block")
    after_rows = np.empty((8, 8), dtype=np.float64)
    result = np.empty((8, 8), dtype=np.float64)
    for row in range(8):
        after_rows[row, :] = dct_1d(block[row, :])
    for column in range(8):
        result[:, column] = dct_1d(after_rows[:, column])
    return after_rows, result


def validate_image_dimensions(image):
    h, w = image.shape
    if h <= 0 or w <= 0 or h % 8 != 0 or w % 8 != 0:
        raise ValueError("image dimensions must be positive multiples of 8")


def transform_block(block, C=DCT_BASIS):
    shifted = block.astype(np.float64) - 128.0

    after_rows = C @ shifted
    F = after_rows @ C.T

    QF = np.rint(F / Q).astype(np.int32)

    if np.any((QF < -2047) | (QF > 2047)):
        raise ValueError("quantized coefficient out of range")

    flat = QF.reshape(-1)
    zigzag = [int(flat[i]) for i in ZIGZAG]

    return shifted, after_rows, F, QF, zigzag


def _value_code_parts(v):
    """Compute (code, code_len, category, amplitude_or_None) for a VALUE symbol."""
    if v == 0:
        s = 0
    else:
        s = abs(v).bit_length()
    if s not in VALUE_CODE:
        raise ValueError(f"coefficient value out of range: {v}")
    code, length = VALUE_CODE[s]
    if s == 0:
        return code, length, s, None
    if v > 0:
        amplitude = v
    else:
        amplitude = v + (2 ** s - 1)
    return code, length, s, amplitude


def encode_block_symbols(zigzag):
    for v in zigzag:
        if not (-2047 <= v <= 2047):
            raise ValueError(f"quantized coefficient out of range: {v}")
    last_idx = -1
    for i in range(63, -1, -1):
        if zigzag[i] != 0:
            last_idx = i
            break
    if last_idx == -1:
        return [("EOB", None)]
    symbols = [("VALUE", int(zigzag[i])) for i in range(0, last_idx + 1)]
    if last_idx < 63:
        symbols.append(("EOB", None))
    return symbols


def write_symbol(writer, token, arg):
    if token == "VALUE":
        code, length, s, amplitude = _value_code_parts(arg)
        writer.write_bits(code, length)
        if s > 0:
            writer.write_bits(amplitude, s)
    elif token == "EOB":
        writer.write_bits(EOB_CODE[0], EOB_CODE[1])
    else:
        raise ValueError(f"unknown token: {token}")


def token_bits_for_trace(token, arg):
    if token == "VALUE":
        code, length, s, amplitude = _value_code_parts(arg)
        bits = format(code, f"0{length}b")
        if s > 0:
            bits += format(amplitude, f"0{s}b")
        return bits
    elif token == "EOB":
        return format(EOB_CODE[0], f"0{EOB_CODE[1]}b")
    else:
        raise ValueError(f"unknown token: {token}")


def encode(input_png, output_bin, trace=None, block_index=0):
    image=np.asarray(Image.open(input_png).convert("L"), dtype=np.uint8)
    h, w=image.shape
    if trace: trace.update(w=w, h=h, blocks=w*h//64)
    validate_image_dimensions(image)
    if not 0 <= block_index < w*h//64: raise ValueError("invalid block index")
    writer=BitWriter(); index=0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block=image[y:y+8, x:x+8]
            if trace and index == block_index: trace.update(location=f"(y={y}, x={x})", block=block)
            shifted, after_rows, F, QF, zigzag=transform_block(block)
            if trace and index == block_index:
                trace.update(shifted=shifted, after_rows=after_rows, F=F, QF=QF, zigzag=zigzag)
            symbols=encode_block_symbols(zigzag); start=writer.bit_count
            for token, arg in symbols: write_symbol(writer, token, arg)
            if trace and index == block_index: trace.update(tokens=symbols, start=start, end=writer.bit_count)
            index += 1
    payload=writer.finish(); payload_bits=writer.bit_count; total_bits=(HEADER.size+len(payload))*8
    Path(output_bin).write_bytes(HEADER.pack(payload_bits)+payload)
    if trace: trace.update(payload_bits=payload_bits, padding_bits=(-payload_bits)%8, total_bits=total_bits, file_size=total_bits//8)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("input_png"); p.add_argument("output_bin")
    p.add_argument("--trace"); p.add_argument("--block-index", type=int, default=0); args=p.parse_args()
    trace=Trace(args.trace, args.block_index) if args.trace else None
    try: encode(args.input_png, args.output_bin, trace, args.block_index)
    except Exception as exc:
        if trace: trace.fail(exc)
        raise
    finally:
        if trace: trace.write()
