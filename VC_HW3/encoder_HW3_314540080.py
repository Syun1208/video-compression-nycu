#!/usr/bin/env python3
"""HW3 encoder: 8x8 intra prediction + DCT + quantization + VALUE/EOB coding."""
import argparse
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

# Intra prediction modes. The mode index is written as a 3-bit code before each block.
MODE_NAMES = ["Vertical", "Horizontal", "DC", "Diagonal Down-Left", "Plane"]
MODE_BITS = 3
UNAVAILABLE = 128  # value used for reference pixels outside the image


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


def write_mode(writer, mode):
    """Write the mode index as a 3-bit unsigned integer (MSB-first)."""
    writer.write_bits(int(mode), MODE_BITS)


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
    def reference_text(self):
        top=self.data.get("top"); left=self.data.get("left"); corner=self.data.get("corner")
        if top is None or left is None or corner is None: return "[not available]"
        return f"top={matrix(top)}\nleft={matrix(left)}\ncorner={int(corner)}\ntop_available={int(self.data.get('has_top'))}\nleft_available={int(self.data.get('has_left'))}"
    def cost_text(self):
        costs=self.data.get("costs")
        if costs is None: return "[not available]"
        return "\n".join(f"mode={m} {MODE_NAMES[m]:<18} mse={float(c):.6f}" for m, c in enumerate(costs))
    def mode_text(self):
        mode=self.data.get("mode")
        if mode is None: return "[not available]"
        return f"mode={mode}\nname={MODE_NAMES[mode]}\ncode={mode:0{MODE_BITS}b}"
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
            ("REFERENCE PIXELS", self.reference_text()),
            ("MODE COSTS", self.cost_text()),
            ("SELECTED MODE", self.mode_text()),
            ("PREDICTION BLOCK", matrix(d.get("prediction"))),
            ("RESIDUAL BLOCK", matrix(d.get("residual"))),
            ("AFTER ROW-WISE 1D DCT", matrix(d.get("after_rows"), 4)),
            ("DCT COEFFICIENTS", matrix(d.get("F"), 4)),
            ("QUANTIZED COEFFICIENTS", matrix(d.get("QF"))),
            ("ZIGZAG COEFFICIENTS", matrix(d.get("zigzag"))),
            ("MODE / VALUE / EOB SYMBOLS", self.token_lines()),
            ("RECONSTRUCTED BLOCK", matrix(d.get("recon"))),
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


# =========================================================================
# PART A - Functions from HW1 (copy your HW1 implementation)
#   Only change: transform_block now receives the prediction residual,
#   so do NOT subtract 128 (no level shift) and do NOT return shifted.
# =========================================================================

def transform_block(residual, C=DCT_BASIS):
    """DCT, quantization, coefficient-range validation and zigzag scan.

    residual = block - prediction (already centered, no level shift).
    Return (after_rows, F, QF, zigzag):
        after_rows = residual @ C.T      (1D DCT on every row)
        F          = C @ after_rows      (1D DCT on every column)
        QF         = rint(F / Q) as int32, must be within [-2047, 2047]
        zigzag     = list of 64 Python ints in ZIGZAG order
    """
    # TODO: implement this function.
    after_rows = residual @ C.T
    F = C @ after_rows

    QF = np.rint(F / Q).astype(np.int32)

    if np.any(np.abs(QF) > 2047):
        raise ValueError("quantized coefficient outside [-2047, 2047]")

    flat = QF.reshape(-1)
    zigzag = [int(flat[i]) for i in ZIGZAG]

    return after_rows, F, QF, zigzag


def encode_block_symbols(zigzag):
    """Return the VALUE/EOB symbols of one block, e.g. [("VALUE", 5), ("VALUE", 0), ("EOB", None)].

    Emit VALUE for every coefficient up to the last nonzero one (internal zeros included),
    then EOB unless that was position 63. An all-zero block is [("EOB", None)].
    """
    # TODO: implement this function.
    nonzero = np.flatnonzero(zigzag)
    last = int(nonzero[-1]) if nonzero.size else -1

    symbols = [("VALUE", int(v)) for v in zigzag[:last + 1]]

    if last < 63:
        symbols.append(("EOB", None))

    return symbols


def write_symbol(writer, token, arg):
    """Write one VALUE or EOB symbol to the bitstream (MSB-first) and return the bits written.

    VALUE(v): VALUE_CODE prefix of category s, then s amplitude bits
              (v for v > 0, v + 2^s - 1 for v < 0, none for v == 0).
    EOB     : EOB_CODE.
    Return the bit string written, e.g. "011" + "10" for VALUE(2) -> "01110".
    """

    # TODO: implement this function.
    if token == "EOB":
        code, length = EOB_CODE
        writer.write_bits(code, length)

        return f"{code:0{length}b}"

    if token != "VALUE":
        raise ValueError(f"unknown token: {token}")

    v = int(arg)
    s = abs(v).bit_length()
    code, length = VALUE_CODE[s]
    writer.write_bits(code, length)
    bits = f"{code:0{length}b}"

    if s > 0:
        amplitude = v if v > 0 else v + (1 << s) - 1
        writer.write_bits(amplitude, s)
        bits += f"{amplitude:0{s}b}"

    return bits


# =========================================================================
# PART B - New in HW3: intra prediction
# =========================================================================

def get_reference_pixels(recon, y, x):
    """Return (top, left, corner) reference pixels of the block whose top-left corner is (y, x).

    top    : int32 array of length 16, the reconstructed row above the block (T[0..15]).
    left   : int32 array of length 8, the reconstructed column left of the block (L[0..7]).
    corner : int, the reconstructed above-left pixel C (only used by Plane).
    Rules: block row y == 0             -> top[0..15] = 128
           block column x == 0          -> left[0..7] = 128
           y == 0 or x == 0             -> corner = 128
           top-right not in image       -> top[8..15] = top[7]
    Only use `recon` (the reconstructed image), never the original image.
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
    """Return the 8x8 int32 prediction P for the given mode (P[y, x], y = row, x = column).

    has_top / has_left: whether the block above / to the left exists (y > 0 / x > 0).

    mode 0 Vertical          : P(x, y) = T[x]
    mode 1 Horizontal        : P(x, y) = L[y]
    mode 2 DC                : top and left available -> (sum(T[0..7]) + sum(L[0..7]) + 8) >> 4
                               only top available     -> (sum(T[0..7]) + 4) >> 3
                               only left available    -> (sum(L[0..7]) + 4) >> 3
                               neither available      -> 128
    mode 3 Diagonal Down-Left: d = x + y
                               P(x, y) = (T[d] + 2*T[d+1] + T[d+2] + 2) >> 2
                               P(7, 7) = (T[14] + 3*T[15] + 2) >> 2
    mode 4 Plane             : T[-1] = L[-1] = C (corner)
                               H = sum_{k=1..4} k * (T[3+k] - T[3-k])
                               V = sum_{k=1..4} k * (L[3+k] - L[3-k])
                               a = 16 * (T[7] + L[7])
                               b = (17*H + 16) >> 5,  c = (17*V + 16) >> 5
                               P(x, y) = clip((a + b*(x-3) + c*(y-3) + 16) >> 5, 0, 255)
    """
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


def select_mode(block, top, left, corner, has_top, has_left):
    """Try all 5 modes and return (best_mode, costs, prediction).

    costs[m] = mean((block - P_m) ** 2) over the 64 pixels (float).
    best_mode is the mode with the smallest cost; ties go to the smallest mode index.
    """
    # TODO: implement this function.
    block_values = block.astype(np.int32)

    predictions = [
        predict_block(mode, top, left, corner, has_top, has_left)
        for mode in range(len(MODE_NAMES))
    ]
    costs = [
        float(np.mean((block_values - prediction) ** 2))
        for prediction in predictions
    ]

    best_mode = int(np.argmin(costs))

    return best_mode, costs, predictions[best_mode]


def reconstruct_block(prediction, QF, C=DCT_BASIS):
    """Rebuild the block exactly as the decoder will, so it can be used as reference.

    dequantized = QF * Q
    residual    = (C.T @ dequantized) @ C
    recon       = clip(rint(prediction + residual), 0, 255) as uint8
    """
    # TODO: implement this function.
    dequantized = QF * Q
    residual = (C.T @ dequantized) @ C

    return np.clip(np.rint(prediction + residual), 0, 255).astype(np.uint8)

# =========================================================================
# Do not modify.
# =========================================================================


def encode(input_png, output_bin, trace=None, block_index=0):
    image=np.asarray(Image.open(input_png).convert("L"), dtype=np.uint8)
    h, w=image.shape
    if trace: trace.update(w=w, h=h, blocks=w*h//64)
    validate_image_dimensions(image)
    if not 0 <= block_index < w*h//64: raise ValueError("invalid block index")
    recon=np.zeros_like(image)
    writer=BitWriter(); index=0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            selected = trace is not None and index == block_index
            block=image[y:y+8, x:x+8]
            if selected: trace.update(location=f"(y={y}, x={x})", block=block)
            top, left, corner=get_reference_pixels(recon, y, x)
            has_top, has_left=y > 0, x > 0
            if selected: trace.update(top=top, left=left, corner=corner, has_top=has_top, has_left=has_left)
            mode, costs, prediction=select_mode(block, top, left, corner, has_top, has_left)
            residual=block.astype(np.int32)-prediction
            if selected: trace.update(costs=costs, mode=mode, prediction=prediction, residual=residual)
            after_rows, F, QF, zigzag=transform_block(residual)
            if selected: trace.update(after_rows=after_rows, F=F, QF=QF, zigzag=zigzag)
            start=writer.bit_count
            write_mode(writer, mode)
            symbols=encode_block_symbols(zigzag)
            encoded_tokens=[("MODE", mode, format(mode, f"0{MODE_BITS}b"))]
            for token, arg in symbols:
                bits=write_symbol(writer, token, arg)
                encoded_tokens.append((token, arg, bits))
            if selected: trace.update(tokens=encoded_tokens, start=start, end=writer.bit_count)
            recon[y:y+8, x:x+8]=reconstruct_block(prediction, QF)
            if selected: trace.update(recon=recon[y:y+8, x:x+8].copy())
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
