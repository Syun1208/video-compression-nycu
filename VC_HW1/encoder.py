#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from codec_parameters import DCT_BASIS, EOB_CODE, HEADER, Q, VALUE_CODE, ZIGZAG


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
        for i, (token, arg, bits) in enumerate(tokens):
            name=token if arg is None else f"{token}({arg})"
            lines.append(f"token[{i:02d}]={name:<20} bits={bits} length={len(bits)}")
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


def validate_image_dimensions(image):
    h, w = image.shape
    if h <= 0 or w <= 0 or h % 8 != 0 or w % 8 != 0:
        raise ValueError("image dimensions must be positive multiples of 8")


def transform_block(block, C=DCT_BASIS):
    shifted = block.astype(np.float64) - 128.0
    # TODO (DCT / quantization / zigzag stages):
    # Standard separable version (used for the required trace):
    #   after_rows = shifted @ C.T      # row-wise 1D DCT
    #   F = C @ after_rows              # then column-wise 1D DCT
    # Quantize, check QF is in [-2047, 2047], and apply
    # ZIGZAG to create the 64-value sequence.
    # When complete, return:
    # return shifted, after_rows, F, QF, zigzag
    after_rows = shifted @ C.T
    F = C @ after_rows
    QF = np.rint(F / Q).astype(np.int32)
    if np.any(np.abs(QF) > 2047):
        raise ValueError("quantized coefficient outside [-2047, 2047]")
    zigzag = QF.reshape(64)[ZIGZAG]
    return shifted, after_rows, F, QF, zigzag


def encode_block_symbols(zigzag):
    # TODO: encode internal zeros as VALUE(0).  Return a list of
    # ("VALUE", v) and ("EOB", None) pairs.  An all-zero block uses only EOB;
    # append EOB only when the last nonzero value is before index 63.
    # When complete, return: symbols
    nonzero = np.flatnonzero(zigzag)
    last = int(nonzero[-1]) if nonzero.size else -1
    symbols = [("VALUE", int(v)) for v in zigzag[:last + 1]]
    if last < 63:
        symbols.append(("EOB", None))
    return symbols


def write_symbol(writer, token, arg):
    # TODO: write one symbol MSB-first through BitWriter, then return the
    # exact bit string written for the trace.  For VALUE(v): compute category
    # s, write VALUE_CODE[s], then write no amplitude bits for s=0, v itself
    # for v>0, or v + (2**s - 1) for v<0.  For EOB, write EOB_CODE.
    # When complete, return: bits
    if token == "EOB":
        code, length = EOB_CODE
        writer.write_bits(code, length)
        return f"{code:0{length}b}"
    if token != "VALUE":
        raise ValueError(f"unknown token: {token}")
    v = int(arg)
    s = abs(v).bit_length()  # = floor(log2|v|) + 1 for v != 0, 0 for v == 0
    code, length = VALUE_CODE[s]
    writer.write_bits(code, length)
    bits = f"{code:0{length}b}"
    if s > 0:
        amplitude = v if v > 0 else v + (1 << s) - 1
        writer.write_bits(amplitude, s)
        bits += f"{amplitude:0{s}b}"
    return bits


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
            encoded_tokens=[]
            for token, arg in symbols:
                bits=write_symbol(writer, token, arg)
                encoded_tokens.append((token, arg, bits))
            if trace and index == block_index: trace.update(tokens=encoded_tokens, start=start, end=writer.bit_count)
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
