#!/usr/bin/env python3
"""HW2 Advanced student-side submission check.

Checks folder names, required source/report/outputs, and parses every
advanced bitstream with provided/evaluator.py. This does NOT canonical-decode.
"""
import argparse
from pathlib import Path
from evaluator import evaluate

METHODS = ('full_search', 'three_step_search', 'three_step_rdo')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission', type=Path)
    a = parser.parse_args()
    root = a.submission.resolve()
    messages = []
    if not root.name.startswith('HW2_'):
        messages.append('folder must be HW2_<student_id>')
    for name in ('encoder.py', 'decoder.py'):
        if not (root / 'src' / name).is_file():
            messages.append('missing src/' + name)
    report = root / 'report' / f'{root.name}.pdf'
    if not report.is_file():
        messages.append(f'missing {report.relative_to(root)}')

    images = Path(__file__).resolve().parent.parent / 'images'
    ref, cur = images / 'one_gray.png', images / 'two_gray.png'
    for method in METHODS:
        result_dir = root / 'results/submitted' / method
        binary, png = result_dir / 'encoded.bin', result_dir / 'decoded.png'
        if not binary.is_file() or not png.is_file():
            messages.append(f'{method}: missing encoded.bin or decoded.png')
            continue
        try:
            stats = evaluate(ref, cur, png, binary, method)
            print(f"[SYNTAX PASS] {method}: macroblocks={stats['macroblocks']}, "
                  f"16x16={stats['mode_large']}, 4x8x8={stats['mode_small']}, "
                  f"BPP={stats['bpp']:.6f}")
        except Exception as exc:
            messages.append(f'{method}: {type(exc).__name__}: {exc}')

    if messages:
        for message in messages:
            print('[FAIL]', message)
        raise SystemExit(1)
    print('SUBMISSION STRUCTURE + BITSTREAM SYNTAX: PASS')
    print('IMPORTANT: This does not verify decoded.png matches TA canonical decoding.')


if __name__ == '__main__':
    main()
