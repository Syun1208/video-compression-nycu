import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from bitstream import BitReader
from hw1_codec import read_residual_block, reconstruct_pixels
from motion import motion_compensate
from predictor_mvd import mv_predictor, reconstruct_motion_vector

MACROBLOCK_SIZE = 16
SUB_BLOCK_SIZE = 8
SUB_BLOCK_OFFSETS = ((0, 0), (0, 8), (8, 0), (8, 8))
LARGE_MODE = 1
RDO_METHOD = "three_step_rdo"


def load_gray_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def read_motion_vector(reader, mv_field, row, col):
    predictor = mv_predictor(mv_field, row, col)
    mvd = (reader.read_signed_exp_golomb(), reader.read_signed_exp_golomb())
    mv = reconstruct_motion_vector(mvd, predictor)

    return int(mv[0]), int(mv[1])


def decode_residual_block(reader, prediction):
    return reconstruct_pixels(prediction, read_residual_block(reader))


def decode_large_mode(
    reader, reference, mv_field, y, x, size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE
):
    row = y // sub_size
    col = x // sub_size
    mv = read_motion_vector(reader, mv_field, row, col)
    prediction = motion_compensate(reference, y, x, mv, size)

    block = np.empty((size, size), dtype=np.uint8)

    for oy, ox in SUB_BLOCK_OFFSETS:
        prediction_block = prediction[oy:oy + sub_size, ox:ox + sub_size]
        block[oy:oy + sub_size, ox:ox + sub_size] = decode_residual_block(reader, prediction_block)

    mv_field[row:row + size // sub_size, col:col + size // sub_size] = mv

    return block


def decode_small_mode(
    reader, reference, mv_field, y, x, size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE
):
    block = np.empty((size, size), dtype=np.uint8)

    for oy, ox in SUB_BLOCK_OFFSETS:
        sub_y = y + oy
        sub_x = x + ox
        row = sub_y // sub_size
        col = sub_x // sub_size

        mv = read_motion_vector(reader, mv_field, row, col)
        prediction = motion_compensate(reference, sub_y, sub_x, mv, sub_size)
        block[oy:oy + sub_size, ox:ox + sub_size] = decode_residual_block(reader, prediction)
        mv_field[row, col] = mv

    return block


def decode_frame(reference, raw, method, size=MACROBLOCK_SIZE, sub_size=SUB_BLOCK_SIZE):
    height, width = reference.shape

    if height == 0 or width == 0 or height % size or width % size:
        raise ValueError("frame width and height must be positive multiples of 16")

    reader = BitReader(raw)
    mv_field = np.zeros((height // sub_size, width // sub_size, 2), dtype=np.int32)
    reconstruction = np.empty((height, width), dtype=np.uint8)

    for y in range(0, height, size):
        for x in range(0, width, size):
            mode = reader.read_bits(1)

            if mode != LARGE_MODE and method != RDO_METHOD:
                raise ValueError(
                    f"{method} must use mode 1 for every macroblock, got mode {mode} at {(y, x)}"
                )

            if mode == LARGE_MODE:
                block = decode_large_mode(reader, reference, mv_field, y, x, size, sub_size)
            else:
                block = decode_small_mode(reader, reference, mv_field, y, x, size, sub_size)

            reconstruction[y:y + size, x:x + size] = block

    if not reader.is_finished():
        raise ValueError(f"{reader.valid_bits - reader.pos} unused valid payload bits remain")

    return reconstruction


def decode(reference, encoded, method, output):
    # TODO: parse header and mode bit for each 16x16 macroblock;
    # recover MV via fixed predictor + MVD; decode four residual 8x8 blocks;
    # MC + inverse HW1 transform; round/clip/uint8; save PNG.
    # Do NOT use current/original image in the decoder.
    reference_frame = load_gray_image(reference)
    reconstruction = decode_frame(reference_frame, Path(encoded).read_bytes(), method)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reconstruction).save(output_path)

    return reconstruction

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--method',required=True,choices=['full_search','three_step_search','three_step_rdo'])
    p.add_argument('--output',required=True)
    a=p.parse_args()
    decode(a.reference,a.input,a.method,a.output)

if __name__=='__main__':main()
