# README

## Encoding

Encode a grayscale PNG image into a binary file:

```bash
python encoder.py lena_gray.png output.bin
```

To generate a trace for a selected 8×8 block:

```bash
python encoder.py lena_gray.png output.bin --trace encoder_trace.txt --block-index 0
```

`--block-index` follows raster order and starts from `0`.

## Decoding

Decode the binary file and reconstruct the image:

```bash
python decoder.py output.bin reconstructed.png
```

To generate a decoder trace:

```bash
python decoder.py output.bin reconstructed.png --trace decoder_trace.txt --block-index 0
```

## Evaluation

Evaluate the reconstructed image:

```bash
python evaluator.py lena_gray.png reconstructed.png output.bin
```

The evaluator reports MSE, PSNR, and BPP.
