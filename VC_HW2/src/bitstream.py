import struct

from hw1_codec import CODE_TO_CATEGORY, EOB_PREFIX

HEADER = struct.Struct(">I")
MAX_EXP_GOLOMB_ZEROS = 31


def signed_code_num(value):
    if value > 0:
        return 2 * value - 1

    return -2 * value


def signed_exp_golomb(value):
    number = signed_code_num(value) + 1
    length = 2 * (number.bit_length() - 1) + 1

    return number, length


def signed_exp_golomb_bits(value):
    return signed_exp_golomb(value)[1]


class BitWriter:
    def __init__(self):
        self.bits = []

    @property
    def bit_count(self):
        return len(self.bits)

    def write_bits(self, value, count):
        if count < 0 or value < 0 or value >= (1 << count):
            raise ValueError(f"cannot write {value} in {count} bits")

        for shift in range(count - 1, -1, -1):
            self.bits.append((value >> shift) & 1)

    def write_signed_exp_golomb(self, value):
        number, length = signed_exp_golomb(value)
        self.write_bits(number, length)

    def extend(self, other):
        self.bits.extend(other.bits)

    def to_bytes(self, header=HEADER):
        valid_bits = self.bit_count
        padded = self.bits + [0] * ((-valid_bits) % 8)
        payload = bytearray()

        for start in range(0, len(padded), 8):
            byte = 0

            for bit in padded[start : start + 8]:
                byte = (byte << 1) | bit

            payload.append(byte)

        return header.pack(valid_bits) + bytes(payload)


class BitReader:
    def __init__(self, raw, header=HEADER):
        if len(raw) < header.size:
            raise ValueError("file is shorter than the 4-byte header")

        self.valid_bits = header.unpack(raw[: header.size])[0]
        expected_size = header.size + (self.valid_bits + 7) // 8

        if len(raw) != expected_size:
            raise ValueError(f"file size mismatch: expected {expected_size} bytes, got {len(raw)}")

        self.payload = raw[header.size :]
        padding_bits = (-self.valid_bits) % 8

        if padding_bits and self.payload[-1] & ((1 << padding_bits) - 1):
            raise ValueError("final padding bits are not zero")

        self.pos = 0

    def read_bits(self, count):
        if count < 0 or self.pos + count > self.valid_bits:
            raise ValueError(f"read past the valid payload at bit {self.pos}")

        value = 0

        for _ in range(count):
            bit = (self.payload[self.pos // 8] >> (7 - self.pos % 8)) & 1
            value = (value << 1) | bit
            self.pos += 1

        return value

    def read_signed_exp_golomb(self, max_zeros=MAX_EXP_GOLOMB_ZEROS):
        zeros = 0

        while self.read_bits(1) == 0:
            zeros += 1

            if zeros > max_zeros:
                raise ValueError("Exp-Golomb prefix is too long")

        number = (1 << zeros) | self.read_bits(zeros)
        code_num = number - 1

        if code_num % 2 == 1:
            return (code_num + 1) // 2

        return -(code_num // 2)

    def read_token(self, code_to_category=CODE_TO_CATEGORY, eob_prefix=EOB_PREFIX):
        prefix = ""

        while len(prefix) < len(eob_prefix):
            prefix += str(self.read_bits(1))

            if prefix == eob_prefix:
                return None

            if prefix in code_to_category:
                return code_to_category[prefix]

        raise ValueError(f"invalid VALUE/EOB prefix at bit {self.pos}")

    def is_finished(self):
        return self.pos == self.valid_bits
