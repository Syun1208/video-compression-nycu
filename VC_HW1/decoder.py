#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from codec_parameters import CODE_TO_CATEGORY, DCT_BASIS, EOB, HEADER, Q, ZIGZAG


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


def matrix(values, precision=None):
    if values is None: return "[not available]"
    kwargs = dict(separator=", ", max_line_width=140)
    if precision is not None: kwargs.update(precision=precision, floatmode="fixed")
    return np.array2string(np.asarray(values), **kwargs)


class Trace:
    """Save reached results, just like the encoder trace."""
    def __init__(self, path, block_index):
        self.path=path; self.block_index=block_index; self.data={}; self.error=None
    def update(self, **items): self.data.update(items)
    def fail(self, exc): self.error=f"{type(exc).__name__}: {exc}"
    def token_lines(self):
        tokens=self.data.get("tokens")
        if tokens is None: return "[not available]"
        lines=[]
        for i, token in enumerate(tokens):
            if token[0] == "EOB":
                _, _, first, last = token
                lines.append(f"token[{i:02d}]=EOB                    bits={first}:{last} prefix=111111111")
            else:
                _, value, first, last, s = token
                lines.append(f"token[{i:02d}]=VALUE({value}) category={s} bits={first}:{last}")
        return "\n".join(lines)
    def write(self):
        d=self.data; missing="[not available]"
        info=(f'width={d.get("w", missing)}\nheight={d.get("h", missing)}\n'
              f'block_count={d.get("blocks", missing)}\nselected_block_index={self.block_index}\n'
              f'selected_block_top_left={d.get("location", missing)}')
        start=d.get("start"); end=d.get("end")
        bit_text=missing if start is None or end is None else f"start_bit={start}\nend_bit={end}\nbit_count={end-start}"
        sections=[
            ("IMAGE INFORMATION", info),
            ("HEADER", f'valid_payload_bits={d.get("valid_bits", missing)}'),
            ("DECODED VALUE / EOB TOKENS", self.token_lines()),
            ("DECODED ZIGZAG COEFFICIENTS", matrix(d.get("zigzagged"))),
            ("INVERSE ZIGZAG QUANTIZED COEFFICIENTS", matrix(d.get("qf"))),
            ("DEQUANTIZED DCT COEFFICIENTS", matrix(d.get("dequantized"))),
            ("AFTER COLUMN-WISE 1D IDCT", matrix(d.get("after_columns"), 4)),
            ("INVERSE DCT + LEVEL SHIFT", matrix(d.get("spatial"), 4)),
            ("OUTPUT BLOCK", matrix(d.get("output"))),
            ("SELECTED BLOCK BITS", bit_text),
            ("BIT COUNTS", f'header_bits=32\nvalid_payload_bits={d.get("valid_bits", missing)}\n'
             f'final_padding_bits={d.get("padding_bits", missing)}\ntotal_bits={d.get("total_bits", missing)}\n'
             f'file_size_bytes={d.get("file_size", missing)}'),
        ]
        # Successful output keeps the original reference format.
        if self.error: sections.insert(0, ("TRACE STATUS", f"stopped: {self.error}"))
        text="\n\n".join(f"===== DECODER {name} =====\n{body}" for name, body in sections)+"\n"
        Path(self.path).write_text(text, encoding="utf-8"); print(text, end="")


def read_value(reader, s):
    # TODO: read s amplitude bits. For a negative value, return
    # amplitude - (2**s - 1). VALUE(0) has no amplitude bits.
    # When complete, return: value
    if s == 0:
        return 0
    amplitude = reader.read_bits(s)
    if amplitude >= (1 << (s - 1)):
        return amplitude
    return amplitude - (1 << s) + 1


def read_block(reader):
    # TODO: initialize 64 zero-filled int32 coefficients, then decode until
    # 64 VALUEs or EOB. Return coeffs plus token records for the trace.
    # EOB: ("EOB", None, start_bit, end_bit)
    # VALUE: ("VALUE", value, start_bit, end_bit, category)
    # When complete, return: coeffs, tokens
    coeffs = np.zeros(64, dtype=np.int32)
    tokens = []
    index = 0
    while index < 64:
        start_bit = reader.pos
        category = reader.read_token()
        if category is None:  # EOB: remaining coefficients stay zero
            tokens.append(("EOB", None, start_bit, reader.pos))
            break
        value = read_value(reader, category)
        coeffs[index] = value
        # end_bit is taken after the amplitude bits, so the range covers prefix + amplitude
        tokens.append(("VALUE", value, start_bit, reader.pos, category))
        index += 1
    return coeffs, tokens


def reconstruct_block(zigzagged, C=DCT_BASIS):
    """Reconstruct one 8x8 block and return the values needed by the trace."""
    # TODO: inverse zigzag -> dequantization -> two explicit 1D IDCT stages.
    # Restore row-major positions with ZIGZAG and reshape to 8x8 qf.
    # Dequantize using Q, then:
    #   after_columns = C.T @ dequantized
    #   spatial = after_columns @ C + 128.0
    #   block = np.clip(np.rint(spatial), 0, 255).astype(np.uint8)
    # When complete, return: qf, dequantized, after_columns, spatial, block
    flat = np.zeros(64, dtype=np.int32)
    flat[ZIGZAG] = zigzagged
    qf = flat.reshape(8, 8)
    dequantized = qf * Q
    after_columns = C.T @ dequantized
    spatial = after_columns @ C + 128.0
    block = np.clip(np.rint(spatial), 0, 255).astype(np.uint8)
    return qf, dequantized, after_columns, spatial, block


def decode(input_bin, output_png, width=512, height=512, trace=None, block_index=0):
    if trace: trace.update(w=width, h=height)
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError("width and height must be positive multiples of 8")
    blocks=width*height//64
    if trace: trace.update(blocks=blocks)
    if not 0 <= block_index < blocks: raise ValueError("invalid block index")

    raw=Path(input_bin).read_bytes()
    if trace: trace.update(file_size=len(raw), total_bits=len(raw)*8)
    if len(raw) < HEADER.size: raise ValueError("file is shorter than 4-byte header")
    valid_bits=HEADER.unpack(raw[:HEADER.size])[0]; padding_bits=(-valid_bits)%8
    if trace: trace.update(valid_bits=valid_bits, padding_bits=padding_bits)
    if len(raw) != HEADER.size+(valid_bits+7)//8: raise ValueError("file size mismatch")
    if padding_bits and raw[-1] & ((1 << padding_bits)-1): raise ValueError("nonzero final padding bits")
    reader=BitReader(raw[HEADER.size:], valid_bits)
    reconstruction=np.empty((height, width), dtype=np.uint8); index=0
    for top in range(0, height, 8):
        for left in range(0, width, 8):
            start=reader.pos
            if trace and index == block_index: trace.update(location=f"(y={top}, x={left})", start=start)
            zigzagged, tokens=read_block(reader)
            if trace and index == block_index:
                trace.update(zigzagged=zigzagged, tokens=tokens, end=reader.pos)
            qf, dequantized, after_columns, spatial, block=reconstruct_block(zigzagged)
            if trace and index == block_index:
                trace.update(qf=qf, dequantized=dequantized, after_columns=after_columns, spatial=spatial, output=block)
            reconstruction[top:top+8, left:left+8]=block
            index += 1
    if reader.pos != valid_bits: raise ValueError("unused valid payload bits remain")
    Image.fromarray(reconstruction).save(output_png)


def main():
    p=argparse.ArgumentParser(); p.add_argument("input_bin"); p.add_argument("output_png")
    p.add_argument("--width", type=int, default=512); p.add_argument("--height", type=int, default=512)
    p.add_argument("--trace"); p.add_argument("--block-index", type=int, default=0); args=p.parse_args()
    trace=Trace(args.trace, args.block_index) if args.trace else None
    try: decode(args.input_bin, args.output_png, args.width, args.height, trace, args.block_index)
    except Exception as exc:
        if trace: trace.fail(exc)
        raise
    finally:
        if trace: trace.write()


if __name__ == "__main__":
    main()
