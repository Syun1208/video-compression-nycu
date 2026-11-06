#!/usr/bin/env python3
"""Student-visible HW2 Advanced bitstream SYNTAX validator and metrics.

Parses each 16x16 macroblock's mode, MVD signed Exp-Golomb codes, and four
HW1 VALUE/EOB residual coefficient sequences. It checks syntax, references,
MV ranges, complete payload consumption, header and zero padding.

This is NOT an encoder/decoder solution. PSNR is computed from the student-
provided decoded.png, NOT from reconstructed pixels decoded from the .bin.
The TA's independent canonical decoder performs that mandatory verification.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from predictor_mvd import mv_predictor

HEADER = struct.Struct('>I')
MACRO = 16
BLOCK = 8
SUBS = ((0, 0), (0, 8), (8, 0), (8, 8))
METHOD_RANGES = {'full_search': 7, 'three_step_search': 7, 'three_step_rdo': 7}
VALUE_CODE = {
    0: (0b00, 2), 1: (0b010, 3), 2: (0b011, 3),
    3: (0b100, 3), 4: (0b101, 3), 5: (0b110, 3),
    6: (0b1110, 4), 7: (0b11110, 5), 8: (0b111110, 6),
    9: (0b1111110, 7), 10: (0b11111110, 8),
    11: (0b111111110, 9),
}
CATEGORIES = {format(v, f'0{length}b'): k for k, (v, length) in VALUE_CODE.items()}
EOB = '111111111'


class BitReader:
    def __init__(self, raw: bytes):
        if len(raw) < HEADER.size:
            raise ValueError('missing 4-byte valid_payload_bits header')
        self.valid_bits = HEADER.unpack(raw[:4])[0]
        expected = HEADER.size + (self.valid_bits + 7) // 8
        if len(raw) != expected:
            raise ValueError(f'header/payload size mismatch: expected {expected} bytes, got {len(raw)}')
        self.payload = raw[4:]
        self.pos = 0
        self.padding_bits = (-self.valid_bits) % 8
        if self.padding_bits and self.payload[-1] & ((1 << self.padding_bits) - 1):
            raise ValueError('final zero-padding bits are not all zero')

    def read_bits(self, count: int) -> int:
        if count < 0 or self.pos + count > self.valid_bits:
            raise ValueError(f'out of valid payload at bit {self.pos} (need {count})')
        value = 0
        for _ in range(count):
            value = (value << 1) | (
                (self.payload[self.pos // 8] >> (7 - self.pos % 8)) & 1
            )
            self.pos += 1
        return value

    def read_signed_exp_golomb(self) -> int:
        zeros = 0
        while self.read_bits(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError('Signed Exp-Golomb unary prefix exceeds 31 zeros')
        payload = self.read_bits(zeros) if zeros else 0
        code_num = ((1 << zeros) | payload) - 1
        return (code_num + 1) // 2 if code_num & 1 else -(code_num // 2)

    def read_value_or_eob(self) -> int | None:
        prefix = ''
        for _ in range(9):
            prefix += str(self.read_bits(1))
            if prefix == EOB:
                return None
            if prefix in CATEGORIES:
                return CATEGORIES[prefix]
        raise ValueError(f'invalid HW1 VALUE/EOB token at payload bit {self.pos}')

    def consume_residual(self) -> int:
        """Parse one 8x8 quantized residual block without reconstructing pixels."""
        nvalues = 0
        while nvalues < 64:
            category = self.read_value_or_eob()
            if category is None:
                break
            if category:
                self.read_bits(category)  # Signed amplitude as fixed HW1 coding.
            nvalues += 1
        return nvalues


def _load_gray(path: str | Path, label: str) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != 'L':
            raise ValueError(f'{label} must be 8-bit grayscale (mode L), got {im.mode}')
        return np.asarray(im, dtype=np.uint8).copy()


def validate_hw2_advanced_payload(reference_png: str | Path,
                                  bitstream_bin: str | Path, method: str) -> dict:
    if method not in METHOD_RANGES:
        raise ValueError(f'unknown method: {method}')
    ref = _load_gray(reference_png, 'reference')
    h, w = ref.shape
    if h < 16 or w < 16 or h % 16 or w % 16:
        raise ValueError('reference dimensions must be positive multiples of 16')
    reader = BitReader(Path(bitstream_bin).read_bytes())
    mv_field = np.zeros((h // 8, w // 8, 2), dtype=np.int32)
    max_mv = METHOD_RANGES[method]
    large = small = mvs = nonzero_mvs = residual_blocks = residual_values = 0

    def read_mv(y: int, x: int, block_size: int) -> tuple[int, int]:
        nonlocal mvs, nonzero_mvs
        br, bc = y // BLOCK, x // BLOCK
        predictor = mv_predictor(mv_field, br, bc)
        # Keep arithmetic in Python integers to avoid np.int32 overflow on invalid payloads.
        dy = int(predictor[0]) + reader.read_signed_exp_golomb()
        dx = int(predictor[1]) + reader.read_signed_exp_golomb()
        if abs(dy) > max_mv or abs(dx) > max_mv:
            raise ValueError(f'MV {[dy, dx]} exceeds +/-{max_mv} for {method}')
        if not (0 <= y + dy <= h - block_size and 0 <= x + dx <= w - block_size):
            raise ValueError(f'MV {[dy, dx]} at {(y, x)} points outside reference')
        mvs += 1
        if dy or dx:
            nonzero_mvs += 1
        return dy, dx

    for y in range(0, h, MACRO):
        for x in range(0, w, MACRO):
            try:
                mode = reader.read_bits(1)
                if method != 'three_step_rdo' and mode != 1:
                    raise ValueError(f'{method} must use fixed 16x16 mode=1')
                if mode == 1:
                    large += 1
                    dy, dx = read_mv(y, x, 16)
                    mv_field[y // 8:y // 8 + 2, x // 8:x // 8 + 2] = (dy, dx)
                    for _ in SUBS:
                        residual_values += reader.consume_residual()
                        residual_blocks += 1
                else:
                    small += 1
                    for oy, ox in SUBS:
                        yy, xx = y + oy, x + ox
                        dy, dx = read_mv(yy, xx, 8)
                        mv_field[yy // 8, xx // 8] = (dy, dx)
                        residual_values += reader.consume_residual()
                        residual_blocks += 1
            except Exception as e:
                raise ValueError(
                    f'macroblock top-left={(y, x)}, payload_bit={reader.pos}: {e}'
                ) from e
    if reader.pos != reader.valid_bits:
        raise ValueError(
            f'{reader.valid_bits - reader.pos} unused valid payload bit(s) remain; '
            'extra fields/per-block padding are not allowed'
        )
    total_bits = 8 * Path(bitstream_bin).stat().st_size
    return {
        'macroblocks': (h // 16) * (w // 16),
        'mode_large': large, 'mode_small': small,
        'motion_vectors': mvs, 'nonzero_motion_vectors': nonzero_mvs,
        'residual_blocks': residual_blocks, 'residual_values': residual_values,
        'valid_payload_bits': reader.valid_bits,
        'final_padding_bits': reader.padding_bits,
        'total_stored_bits': total_bits, 'bpp': total_bits / (w * h),
        'width': w, 'height': h,
    }


def evaluate(reference_png: str | Path, current_png: str | Path,
             decoded_png: str | Path, bitstream_bin: str | Path, method: str) -> dict:
    stats = validate_hw2_advanced_payload(reference_png, bitstream_bin, method)
    current = _load_gray(current_png, 'current')
    decoded = _load_gray(decoded_png, 'submitted decoded')
    expected = (stats['height'], stats['width'])
    if current.shape != expected or decoded.shape != expected:
        raise ValueError('current / decoded image dimensions must match reference')
    mse = float(np.mean((current.astype(np.float64) - decoded.astype(np.float64)) ** 2))
    stats['mse_submitted_png'] = mse
    stats['psnr_db_submitted_png'] = (
        math.inf if mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    )
    stats['decoded_is_original_current'] = bool(np.array_equal(decoded, current))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description='HW2 Advanced student-side bitstream syntax validator '
                    '(not a canonical decoder)'
    )
    p.add_argument('reference_png')
    p.add_argument('current_png')
    p.add_argument('encoded_bin')
    p.add_argument('decoded_png')
    p.add_argument('--method', required=True, choices=tuple(METHOD_RANGES))
    a = p.parse_args()
    try:
        m = evaluate(a.reference_png, a.current_png, a.decoded_png, a.encoded_bin, a.method)
    except Exception as e:
        print(f'HW2 ADVANCED BITSTREAM SYNTAX: FAIL\nReason: {type(e).__name__}: {e}')
        raise SystemExit(1)
    print('HW2 ADVANCED BITSTREAM SYNTAX: PASS')
    print(f"Macroblocks: {m['macroblocks']}; mode 16x16: {m['mode_large']}; "
          f"mode 4x8x8: {m['mode_small']}")
    print(f"MVs parsed: {m['motion_vectors']}; residual 8x8 blocks: {m['residual_blocks']}")
    print('Header, Signed Exp-Golomb, HW1 VALUE/EOB, MV range/boundary, '
          'payload consumption and padding: PASS')
    print(f"Valid payload bits: {m['valid_payload_bits']}; final padding bits: {m['final_padding_bits']}")
    print(f"BPP (encoded.bin size): {m['bpp']:.6f}")
    print(f"PSNR (submitted decoded.png only): {m['psnr_db_submitted_png']:.6f} dB")
    if m['decoded_is_original_current']:
        print('WARNING: decoded.png equals original current image pixel-for-pixel; '
              'check that you actually decoded the bitstream.')
    print('IMPORTANT: Syntax PASS does not prove that the PNG matches a canonical '
          'decode of the .bin. The TA grader independently reconstructs and checks that.')


if __name__ == '__main__':
    main()
