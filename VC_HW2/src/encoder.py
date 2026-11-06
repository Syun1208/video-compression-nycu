#!/usr/bin/env python3
"""HW2 advanced student encoder. Implement inter-frame encoding yourself.

REQUIRED CLI (TA grader calls this exactly):
python encoder.py --reference REF.png --current CUR.png \
  --method full_search|four_step_search|four_step_rdo --output OUT.bin

This file is intentionally unfinished; code structure may be changed, but not
CLI, bitstream syntax, RDO objective, HW1 transform tables, or image semantics.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bitstream import BitWriter
from hw1_codec import forward_transform, reconstruct_pixels, write_residual_block
from motion import full_search, motion_compensate, three_step_search
from predictor_mvd import motion_vector_difference, mv_predictor

METHODS = ("full_search", "three_step_search", "three_step_rdo")
MACROBLOCK_SIZE = 16
SUB_BLOCK_SIZE = 8
SUB_BLOCK_OFFSETS = ((0, 0), (0, 8), (8, 0), (8, 8))
RDO_LAMBDA = 64
LARGE_MODE = 1
SMALL_MODE = 0
SEARCH_FUNCTIONS = {
    "full_search": full_search,
    "three_step_search": three_step_search,
    "three_step_rdo": three_step_search,
}


@dataclass
class MacroblockCandidate:
    mode: int
    writer: BitWriter
    reconstruction: np.ndarray
    mv_field: np.ndarray
    motion_vectors: list

    @property
    def rate(self):
        return self.writer.bit_count


@dataclass
class EncodedFrame:
    writer: BitWriter
    reconstruction: np.ndarray
    mv_field: np.ndarray
    macroblocks: list


def load_gray_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def validate_frames(reference, current, macroblock_size=MACROBLOCK_SIZE):
    if reference.shape != current.shape:
        raise ValueError("reference and current frames must have the same size")

    height, width = current.shape

    if height == 0 or width == 0 or height % macroblock_size or width % macroblock_size:
        raise ValueError("frame width and height must be positive multiples of 16")


def write_motion_vector_difference(writer, mv, predictor):
    mvd = motion_vector_difference(mv, predictor)
    writer.write_signed_exp_golomb(int(mvd[0]))
    writer.write_signed_exp_golomb(int(mvd[1]))


def encode_residual_block(writer, current_block, prediction):
    residual = current_block.astype(np.int32) - prediction.astype(np.int32)
    quantized = forward_transform(residual)
    write_residual_block(writer, quantized)

    return reconstruct_pixels(prediction, quantized)


def encode_large_mode(current, reference, mv_field, y, x, search,
                      size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE):
    field = mv_field.copy()
    row = y // sub_size
    col = x // sub_size
    mv, _ = search(current, reference, y, x, size)

    writer = BitWriter()
    writer.write_bits(LARGE_MODE, 1)
    write_motion_vector_difference(writer, mv, mv_predictor(field, row, col))

    prediction = motion_compensate(reference, y, x, mv, size)
    reconstruction = np.empty((size, size), dtype=np.uint8)

    for oy, ox in SUB_BLOCK_OFFSETS:
        current_block = current[y + oy:y + oy + sub_size, x + ox:x + ox + sub_size]
        prediction_block = prediction[oy:oy + sub_size, ox:ox + sub_size]
        reconstruction[oy:oy + sub_size, ox:ox + sub_size] = encode_residual_block(
            writer, current_block, prediction_block
        )

    field[row:row + size // sub_size, col:col + size // sub_size] = mv

    return MacroblockCandidate(LARGE_MODE, writer, reconstruction, field, [mv])


def encode_small_mode(current, reference, mv_field, y, x, search,
                      size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE):
    field = mv_field.copy()
    writer = BitWriter()
    writer.write_bits(SMALL_MODE, 1)

    reconstruction = np.empty((size, size), dtype=np.uint8)
    motion_vectors = []

    for oy, ox in SUB_BLOCK_OFFSETS:
        sub_y = y + oy
        sub_x = x + ox
        row = sub_y // sub_size
        col = sub_x // sub_size

        mv, _ = search(current, reference, sub_y, sub_x, sub_size)
        write_motion_vector_difference(writer, mv, mv_predictor(field, row, col))

        prediction = motion_compensate(reference, sub_y, sub_x, mv, sub_size)
        current_block = current[sub_y:sub_y + sub_size, sub_x:sub_x + sub_size]
        reconstruction[oy:oy + sub_size, ox:ox + sub_size] = encode_residual_block(
            writer, current_block, prediction
        )

        field[row, col] = mv
        motion_vectors.append(mv)

    return MacroblockCandidate(SMALL_MODE, writer, reconstruction, field, motion_vectors)


def rdo_cost(current_block, candidate, rdo_lambda=RDO_LAMBDA):
    error = current_block.astype(np.int64) - candidate.reconstruction.astype(np.int64)
    ssd = int((error * error).sum())

    return ssd + rdo_lambda * candidate.rate


def encode_macroblock(current, reference, mv_field, y, x, method, size=MACROBLOCK_SIZE):
    search = SEARCH_FUNCTIONS[method]
    large = encode_large_mode(current, reference, mv_field, y, x, search)

    if method != "three_step_rdo":
        return large

    small = encode_small_mode(current, reference, mv_field, y, x, search)
    current_block = current[y:y + size, x:x + size]

    if rdo_cost(current_block, small) < rdo_cost(current_block, large):
        return small

    return large


def encode_frame(reference, current, method, size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")

    validate_frames(reference, current, size)

    height, width = current.shape
    mv_field = np.zeros((height // sub_size, width // sub_size, 2), dtype=np.int32)
    writer = BitWriter()
    reconstruction = np.empty_like(current)
    macroblocks = []

    for y in range(0, height, size):
        for x in range(0, width, size):
            candidate = encode_macroblock(current, reference, mv_field, y, x, method, size)
            writer.extend(candidate.writer)
            reconstruction[y:y + size, x:x + size] = candidate.reconstruction
            mv_field = candidate.mv_field
            macroblocks.append(candidate)

    return EncodedFrame(writer, reconstruction, mv_field, macroblocks)


def encode(reference, current, method, output):
    # TODO: 16x16 macroblock loop; Q1/Q2 use only one 16x16 MV;
    # Q3 tries both one 16x16 MV and four independent 8x8 MVs;
    # Q3 RDO via reconstructed SSD + 64 * (macroblock syntax bits);
    # ALL Q1/Q2/Q3 ME searches: MAD ONLY; no MVD bit cost during MV selection.
    # MVD is still Signed Exp-Golomb coded and written for every chosen MV.
    # For Q3 only, block-size RDO uses SSD + 64 * actual macroblock syntax bits.
    # write mode+MVD+four HW1 coded residuals per macroblock;
    # write 4-byte big-endian valid payload bits header, final zero padding.
    reference_frame = load_gray_image(reference)
    current_frame = load_gray_image(current)
    encoded = encode_frame(reference_frame, current_frame, method)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.writer.to_bytes())

    return encoded

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--current',required=True)
    p.add_argument('--method',required=True,choices=['full_search','three_step_search','three_step_rdo'])
    p.add_argument('--output',required=True)
    a=p.parse_args()
    encode(a.reference,a.current,a.method,a.output)

if __name__=='__main__':main()
