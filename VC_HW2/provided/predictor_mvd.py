#!/usr/bin/env python3
"""HW2 fixed Motion Vector Predictor / MVD helper.

The algorithm in this file is part of the assignment specification and must not
be changed. Students may wrap/adapt the interface to fit their own code, but
must preserve the same semantics and document interface-only changes.
"""
from __future__ import annotations

import numpy as np


def _as_mv(value) -> np.ndarray:
    mv = np.asarray(value, dtype=np.int32)
    if mv.shape != (2,):
        raise ValueError("a motion vector must have shape (2,) as [dy, dx]")
    return mv


def mv_predictor(mv_field: np.ndarray, block_row: int, block_col: int) -> np.ndarray:
    """Return the fixed component-wise median predictor [dy, dx].

    Neighbors:
      A = Left
      B = Top
      C = Top-Right

    An unavailable neighbor is defined as [0, 0]. Blocks are assumed to be
    processed in raster order, so all available neighbors have already been
    finalized.
    """
    field = np.asarray(mv_field)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("mv_field must have shape [block_rows, block_cols, 2]")
    rows, cols, _ = field.shape
    if not (0 <= block_row < rows and 0 <= block_col < cols):
        raise IndexError("block index out of range")

    zero = np.array([0, 0], dtype=np.int32)
    left = _as_mv(field[block_row, block_col - 1]) if block_col > 0 else zero
    top = _as_mv(field[block_row - 1, block_col]) if block_row > 0 else zero
    top_right = (
        _as_mv(field[block_row - 1, block_col + 1])
        if block_row > 0 and block_col + 1 < cols
        else zero
    )

    stacked = np.stack([left, top, top_right], axis=0)
    # With three integer samples, the component-wise median is integer-valued.
    return np.median(stacked, axis=0).astype(np.int32)


def motion_vector_difference(mv, predictor) -> np.ndarray:
    """Return MVD = MV - MVpred as [dy, dx]."""
    return _as_mv(mv) - _as_mv(predictor)


def reconstruct_motion_vector(mvd, predictor) -> np.ndarray:
    """Decoder-side inverse: MV = MVpred + MVD."""
    return _as_mv(predictor) + _as_mv(mvd)
