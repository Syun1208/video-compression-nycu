import numpy as np

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

DCT_BASIS = np.array([
    [0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274,
     0.353553390593274, 0.353553390593274, 0.353553390593274, 0.353553390593274],
    [0.490392640201615, 0.415734806151273, 0.277785116509801, 0.097545161008064,
     -0.097545161008064, -0.277785116509801, -0.415734806151273, -0.490392640201615],
    [0.461939766255643, 0.191341716182545, -0.191341716182545, -0.461939766255643,
     -0.461939766255643, -0.191341716182545, 0.191341716182545, 0.461939766255643],
    [0.415734806151273, -0.097545161008064, -0.490392640201615, -0.277785116509801,
     0.277785116509801, 0.490392640201615, 0.097545161008064, -0.415734806151273],
    [0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274,
     0.353553390593274, -0.353553390593274, -0.353553390593274, 0.353553390593274],
    [0.277785116509801, -0.490392640201615, 0.097545161008064, 0.415734806151273,
     -0.415734806151273, -0.097545161008064, 0.490392640201615, -0.277785116509801],
    [0.191341716182545, -0.461939766255643, 0.461939766255643, -0.191341716182545,
     -0.191341716182545, 0.461939766255643, -0.461939766255643, 0.191341716182545],
    [0.097545161008064, -0.277785116509801, 0.415734806151273, -0.490392640201615,
     0.490392640201615, -0.415734806151273, 0.277785116509801, -0.097545161008064],
], dtype=np.float64)

ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)

VALUE_CODE = {
    0: (0b00, 2), 1: (0b010, 3), 2: (0b011, 3), 3: (0b100, 3), 4: (0b101, 3),
    5: (0b110, 3), 6: (0b1110, 4), 7: (0b11110, 5), 8: (0b111110, 6),
    9: (0b1111110, 7), 10: (0b11111110, 8), 11: (0b111111110, 9),
}
EOB_CODE = (0b111111111, 9)
CODE_TO_CATEGORY = {
    format(code, f"0{length}b"): category
    for category, (code, length) in VALUE_CODE.items()
}
EOB_PREFIX = format(EOB_CODE[0], f"0{EOB_CODE[1]}b")

BLOCK_SIZE = 8
MAX_COEFFICIENT = 2047
BASIS_ROWS = DCT_BASIS.tolist()


def dct_1d(values, basis_rows=BASIS_ROWS):
    result = []

    for basis_row in basis_rows:
        total = 0.0

        for weight, value in zip(basis_row, values):
            total += weight * value

        result.append(total)

    return result


def dct_2d_separable(block, basis_rows=BASIS_ROWS):
    after_rows = [dct_1d(row, basis_rows) for row in block.astype(np.float64).tolist()]
    columns = [dct_1d(column, basis_rows) for column in zip(*after_rows)]

    return np.array(columns, dtype=np.float64).T


def forward_transform(residual_block, basis_rows=BASIS_ROWS, quant_table=Q):
    coefficients = dct_2d_separable(residual_block, basis_rows)
    quantized = np.rint(coefficients / quant_table).astype(np.int32)

    if np.any(np.abs(quantized) > MAX_COEFFICIENT):
        raise ValueError("quantized coefficient outside [-2047, 2047]")

    return quantized


def inverse_transform(quantized, basis=DCT_BASIS, quant_table=Q):
    dequantized = quantized * quant_table
    after_first_pass = basis.T @ dequantized

    return after_first_pass @ basis


def reconstruct_pixels(prediction, quantized, basis=DCT_BASIS, quant_table=Q):
    residual = inverse_transform(quantized, basis, quant_table)
    spatial = prediction.astype(np.float64) + residual

    return np.clip(np.rint(spatial), 0, 255).astype(np.uint8)


def zigzag_scan(quantized, order=ZIGZAG):
    return quantized.reshape(-1)[order]


def inverse_zigzag(values, order=ZIGZAG):
    flat = np.zeros(BLOCK_SIZE * BLOCK_SIZE, dtype=np.int32)
    flat[order] = values

    return flat.reshape(BLOCK_SIZE, BLOCK_SIZE)


def block_symbols(zigzag):
    nonzero = np.flatnonzero(zigzag)
    last_index = int(nonzero[-1]) if nonzero.size else -1
    symbols = [("VALUE", int(value)) for value in zigzag[:last_index + 1]]

    if last_index < len(zigzag) - 1:
        symbols.append(("EOB", None))

    return symbols


def value_code_parts(value, value_code=VALUE_CODE):
    category = abs(value).bit_length()

    if category not in value_code:
        raise ValueError(f"coefficient value out of range: {value}")

    code, length = value_code[category]
    amplitude = value if value > 0 else value + (1 << category) - 1

    return code, length, category, amplitude


def write_residual_block(writer, quantized):
    for token, value in block_symbols(zigzag_scan(quantized)):
        if token == "EOB":
            writer.write_bits(*EOB_CODE)
            continue

        code, length, category, amplitude = value_code_parts(value)
        writer.write_bits(code, length)

        if category > 0:
            writer.write_bits(amplitude, category)


def read_value(reader, category):
    if category == 0:
        return 0

    amplitude = reader.read_bits(category)

    if amplitude >= (1 << (category - 1)):
        return amplitude

    return amplitude - (1 << category) + 1


def read_residual_block(reader):
    values = np.zeros(BLOCK_SIZE * BLOCK_SIZE, dtype=np.int32)
    position = 0

    while position < values.size:
        category = reader.read_token()

        if category is None:
            break

        values[position] = read_value(reader, category)
        position += 1

    return inverse_zigzag(values)
