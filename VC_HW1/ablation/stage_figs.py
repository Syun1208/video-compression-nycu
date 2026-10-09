"""Block-level input/output visualisations for the report (Lena and Mandrill)."""
import os
import numpy as np
from PIL import Image

import ablation as A
import encoder
from codec_parameters import DCT_BASIS as C, Q, ZIGZAG

OUT = os.path.join(A.ROOT, "report", "figures")
NAMES = {"Lena": "lena", "Mandrill": "mandril"}
CROPS = {"Lena": (216, 216), "Mandrill": (24, 272)}


def save_gray(arr01, path, scale=1):
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


LUT = {k: np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lut_{k}.npy")) for k in ("viridis", "magma")}


def save_cmap(arr01, path, cmap="viridis", size=512):
    idx = np.rint(np.clip(arr01, 0, 1) * 255).astype(int)
    rgb = (LUT[cmap][idx] * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((size, size), Image.NEAREST).save(path)


def tile(blocks):
    """(4096, 8, 8) -> (512, 512) placing each block at its image position."""
    return blocks.reshape(64, 64, 8, 8).transpose(0, 2, 1, 3).reshape(512, 512)


def block_bits(qf_blocks, scan):
    bits = np.zeros(len(qf_blocks))
    for k, QF in enumerate(qf_blocks):
        w = encoder.BitWriter()
        for t, a in encoder.encode_block_symbols(QF.reshape(64)[scan]):
            encoder.write_symbol(w, t, a)
        bits[k] = w.bit_count
    return bits.reshape(64, 64)


stats = {}
for name, short in NAMES.items():
    img = A.load(A.IMAGES[name])
    blocks = np.array(A.blocks_of(img)).astype(float) - 128.0
    F = np.array([A.dct_separable_numpy(b) for b in blocks])
    QF = np.rint(F / Q).astype(int)
    Fh = QF * Q
    rec = np.array([np.clip(np.rint(C.T @ b @ C + 128), 0, 255) for b in Fh])
    bits_zz = block_bits(QF, ZIGZAG)
    bits_ra = block_bits(QF, A.RASTER)

    # ---- standard pipeline: input and output of each block
    lf = np.log1p(np.abs(tile(F))) / np.log1p(1024)
    lq = np.log1p(np.abs(tile(QF))) / np.log1p(64)
    lfh = np.log1p(np.abs(tile(Fh))) / np.log1p(1024)
    save_gray(lf, f"{OUT}/{short}_stage_dct.png")
    save_gray((tile(QF) != 0).astype(float), f"{OUT}/{short}_stage_quant.png")
    save_gray(lfh, f"{OUT}/{short}_stage_dequant.png")
    save_cmap(bits_zz / 300.0, f"{OUT}/{short}_bits_zigzag.png")
    save_cmap(bits_ra / 300.0, f"{OUT}/{short}_bits_raster.png")

    # ---- ablation 1: reconstructions with different tables (crops)
    cy, cx = CROPS[name]
    variants = {"standard": Q, "q10": A.ijg_table(10), "q90": A.ijg_table(90),
                "flat16": np.full((8, 8), 16.0), "reversed": Q[::-1, ::-1].copy()}
    a1 = {}
    for vn, Qt in variants.items():
        r = A.run_codec(img, Qt, tag=f"fig_{short}_{vn}")
        save_gray(r["rec"][cy:cy + 128, cx:cx + 128] / 255.0, f"{OUT}/{short}_q_{vn}.png", scale=2)
        a1[vn] = (r["bpp"], r["psnr"])

    # ---- ablation 3: DCT output of the two methods
    Fd = np.array([A.dct_direct_loops(b) for b in blocks])
    dF = np.abs(Fd - F)
    save_gray(np.log1p(np.abs(tile(Fd))) / np.log1p(1024), f"{OUT}/{short}_dct_direct.png")

    stats[name] = dict(
        bits_zz_mean=bits_zz.mean(), bits_zz_max=bits_zz.max(), bits_ra_mean=bits_ra.mean(), bits_ra_max=bits_ra.max(),
        extra_mean=(bits_ra - bits_zz).mean(), extra_max=(bits_ra - bits_zz).max(),
        frac_more=np.mean(bits_ra > bits_zz), frac_equal=np.mean(bits_ra == bits_zz), frac_less=np.mean(bits_ra < bits_zz),
        zero_blocks=np.mean((QF != 0).sum(axis=(1, 2)) == 0), nz_mean=(QF != 0).sum(axis=(1, 2)).mean(),
        dF_max=dF.max(), a1=a1,
        rec_ok=np.array_equal(tile(rec).astype(np.uint8),
                              np.asarray(Image.open(f"{OUT}/{short}_reconstructed.png"))))
for k, v in stats.items():
    print(k, {kk: (float(f'{float(vv):.4g}') if not isinstance(vv, dict) else {a: tuple(round(x, 3) for x in b) for a, b in vv.items()}) for kk, vv in v.items()})
