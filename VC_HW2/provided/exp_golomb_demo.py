#!/usr/bin/env python3
"""HW2 VC student example: Signed Exp-Golomb write/read, NOT a full codec.

Run from student_package: python provided/exp_golomb_demo.py
Uses the public evaluator's BitReader as an independent syntax reader.
"""
from __future__ import annotations
import struct
from evaluator import BitReader


def code_num(value: int) -> int:
    return 2*value-1 if value>0 else -2*value


def encode_signed(value: int) -> str:
    """Zero-order signed Exp-Golomb with k leading zeros and (k+1)-bit value."""
    n=code_num(value)+1
    binary=format(n,'b')
    return '0'*(len(binary)-1)+binary


def pack_payload(bits: str) -> bytes:
    valid=len(bits)
    padded=bits+'0'*((-valid)%8)
    return struct.pack('>I',valid)+bytes(int(padded[i:i+8],2) for i in range(0,len(padded),8))


def main():
    for n in [0,1,-1,2,-2,3,-3]:
        word=encode_signed(n)
        print(f'v={n:>2}, codeNum={code_num(n):>2}, word={word:>9}, length={len(word)}')
        reader=BitReader(pack_payload(word))
        assert reader.read_signed_exp_golomb()==n
        assert reader.pos==reader.valid_bits

    # One 16x16 macroblock; assume MV=(6,1), MVP=(5,1) -> MVD=(+1,0).
    # All four 8x8 quantized residual blocks are zero -> four EOB words.
    mode='1';mvd_y=encode_signed(1);mvd_x=encode_signed(0);eob='111111111'
    payload=mode+mvd_y+mvd_x+4*eob
    raw=pack_payload(payload)
    print('\nMVD=(+1,0), one 16x16 macroblock with four EOB-only residuals')
    print('fields:',mode,'|',mvd_y,'|',mvd_x,'|',' | '.join([eob]*4))
    print('valid_payload_bits:',len(payload),'padding:',(-len(payload))%8)
    print('whole encoded.bin (hex):',raw.hex(' ').upper())
    reader=BitReader(raw)
    assert reader.read_bits(1)==1
    assert reader.read_signed_exp_golomb()==1
    assert reader.read_signed_exp_golomb()==0
    assert [reader.read_value_or_eob() for _ in range(4)]==[None]*4
    assert reader.pos==reader.valid_bits
    print('PASS: mode, MVD_y, MVD_x, 4 EOBs decoded without per-field bit lengths')

if __name__=='__main__':main()
