import numpy as np

SEARCH_RANGE = 7
TSS_STEPS = (4, 2, 1)


def block_at(frame, y, x, size):
    return frame[y : y + size, x : x + size]


def is_valid_candidate(y, x, mv, size, frame_shape, search_range=SEARCH_RANGE):
    dy, dx = mv
    height, width = frame_shape

    if abs(dy) > search_range or abs(dx) > search_range:
        return False

    return 0 <= y + dy <= height - size and 0 <= x + dx <= width - size


def sum_of_absolute_differences(current_block, reference, y, x, mv):
    dy, dx = mv
    size = current_block.shape[0]
    candidate = block_at(reference, y + dy, x + dx, size)
    difference = current_block.astype(np.int32) - candidate.astype(np.int32)

    return int(np.abs(difference).sum())


def candidate_key(sad, mv):
    dy, dx = mv
    return sad, abs(dy) + abs(dx), dy, dx


def best_candidate(current_block, reference, y, x, candidates):
    best_mv = None
    best_key = None

    for mv in candidates:
        key = candidate_key(sum_of_absolute_differences(current_block, reference, y, x, mv), mv)

        if best_key is None or key < best_key:
            best_mv = mv
            best_key = key

    return best_mv, len(candidates)


def full_search(current, reference, y, x, size, search_range=SEARCH_RANGE):
    current_block = block_at(current, y, x, size)
    candidates = [
        (dy, dx)
        for dy in range(-search_range, search_range + 1)
        for dx in range(-search_range, search_range + 1)
        if is_valid_candidate(y, x, (dy, dx), size, reference.shape, search_range)
    ]

    return best_candidate(current_block, reference, y, x, candidates)


def three_step_search(current, reference, y, x, size, steps=TSS_STEPS, search_range=SEARCH_RANGE):
    current_block = block_at(current, y, x, size)
    center = (0, 0)
    evaluated = set()

    for step in steps:
        candidates = [
            (center[0] + sy * step, center[1] + sx * step) for sy in (-1, 0, 1) for sx in (-1, 0, 1)
        ]
        candidates = [
            mv
            for mv in candidates
            if is_valid_candidate(y, x, mv, size, reference.shape, search_range)
        ]
        center, _ = best_candidate(current_block, reference, y, x, candidates)
        evaluated.update(candidates)

    return center, len(evaluated)


def motion_compensate(reference, y, x, mv, size, search_range=SEARCH_RANGE):
    dy, dx = mv

    if not is_valid_candidate(y, x, mv, size, reference.shape, search_range):
        raise ValueError(
            f"motion vector {mv} at {(y, x)} is outside the search range or the reference frame"
        )

    return block_at(reference, y + dy, x + dx, size)
