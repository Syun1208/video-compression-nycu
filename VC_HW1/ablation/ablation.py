"""Ablation experiments for HW1 (not part of the submission).

Every configuration produces a real .bin file using the submitted encoder's
BitWriter / write_symbol and is decoded with the submitted decoder's
BitReader / read_block, so BPP is measured from the actual file size.
"""
import json, math, os, platform, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # VC_HW1/
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image

import encoder, decoder
from codec_parameters import DCT_BASIS as C, HEADER, Q as Q_STD, ZIGZAG

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "data")
IMAGES = {"Lena": "lena_gray.png", "Mandrill": "mandril_gray.png"}
RASTER = np.arange(64, dtype=np.int32)
COLUMN = np.arange(64, dtype=np.int32).reshape(8, 8).T.reshape(64)


def load(name):
    return np.asarray(Image.open(os.path.join(IMG_DIR, name)).convert("L"), dtype=np.uint8)


def blocks_of(img):
    h, w = img.shape
    return [img[y:y + 8, x:x + 8] for y in range(0, h, 8) for x in range(0, w, 8)]


# ---------------------------------------------------------------- DCT variants
def dct_separable_numpy(f):
    return C @ (f @ C.T)


C_LIST = C.tolist()


def dct_separable_loops(f):
    """Same loop style as dct_direct_loops: R = f C^T (rows), then F = C R (columns)."""
    f = f.tolist(); c = C_LIST
    R = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        fi = f[i]; Ri = R[i]
        for u in range(8):
            cu = c[u]; s = 0.0
            for x in range(8):
                s += fi[x] * cu[x]
            Ri[u] = s
    F = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        cu = c[u]; Fu = F[u]
        for j in range(8):
            s = 0.0
            for x in range(8):
                s += cu[x] * R[x][j]
            Fu[j] = s
    return np.array(F)


def dct_direct_loops(f):
    """F(u,v) = sum_x sum_y C(u,x) f(x,y) C(v,y): explicit double summation."""
    f = f.tolist(); c = C_LIST
    F = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        cu = c[u]
        for v in range(8):
            cv = c[v]; s = 0.0
            for x in range(8):
                fx = f[x]; cux = cu[x]
                for y in range(8):
                    s += cux * fx[y] * cv[y]
            F[u][v] = s
    return np.array(F)


# 64x64 kernel K[(u,v),(x,y)] = C(u,x) C(v,y): the same double sum, vectorised
KERNEL = np.einsum("ux,vy->uvxy", C, C).reshape(64, 64)


def dct_direct_numpy(f):
    return (KERNEL @ f.reshape(64)).reshape(8, 8)


DCT_VARIANTS = {
    "sep_numpy": dct_separable_numpy,
    "sep_loops": dct_separable_loops,
    "direct_loops": dct_direct_loops,
    "direct_numpy": dct_direct_numpy,
}


# ------------------------------------------------------------ codec pipeline
def ijg_table(quality):
    """IJG libjpeg quality scaling (jpeg_quality_scaling + jpeg_add_quant_table)."""
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((Q_STD * s + 50) / 100), 1, 255)


def run_codec(img, Qt=Q_STD, scan=ZIGZAG, dct=dct_separable_numpy, tag="tmp"):
    h, w = img.shape
    writer = encoder.BitWriter()
    qfs = []
    n_tokens = n_zero_tokens = no_eob = 0
    last_pos = []
    for b in blocks_of(img):
        F = dct(b.astype(np.float64) - 128.0)
        QF = np.rint(F / Qt).astype(np.int32)
        if np.any(np.abs(QF) > 2047):
            raise ValueError("coefficient out of range")
        qfs.append(QF)
        seq = QF.reshape(64)[scan]
        symbols = encoder.encode_block_symbols(seq)
        for tok, arg in symbols:
            encoder.write_symbol(writer, tok, arg)
        n_tokens += len(symbols)
        n_zero_tokens += sum(1 for t, a in symbols if t == "VALUE" and a == 0)
        no_eob += symbols[-1][0] != "EOB"
        nz = np.flatnonzero(seq)
        last_pos.append(int(nz[-1]) if nz.size else -1)
    payload = writer.finish()
    path = os.path.join(HERE, f"{tag}.bin")
    with open(path, "wb") as fh:
        fh.write(HEADER.pack(writer.bit_count) + payload)
    file_bytes = os.path.getsize(path)

    # decode from the actual file with the submitted decoder primitives
    raw = open(path, "rb").read()
    vb = HEADER.unpack(raw[:4])[0]
    reader = decoder.BitReader(raw[4:], vb)
    rec = np.empty_like(img)
    k = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            seq, _ = decoder.read_block(reader)
            flat = np.zeros(64, dtype=np.int32); flat[scan] = seq
            qf = flat.reshape(8, 8)
            assert np.array_equal(qf, qfs[k]); k += 1
            spatial = C.T @ (qf * Qt) @ C + 128.0
            rec[y:y + 8, x:x + 8] = np.clip(np.rint(spatial), 0, 255).astype(np.uint8)
    assert reader.pos == vb
    os.remove(path)
    mse = float(np.mean((img.astype(np.float64) - rec.astype(np.float64)) ** 2))
    nb = len(qfs)
    return dict(
        payload_bits=vb, file_bytes=file_bytes, bpp=file_bytes * 8 / (h * w), mse=mse,
        psnr=10 * math.log10(255 ** 2 / mse), tokens_per_block=n_tokens / nb,
        zero_tokens_per_block=n_zero_tokens / nb, blocks_without_eob=no_eob,
        mean_last_nonzero=float(np.mean(last_pos)), rec=rec, qf=np.array(qfs),
    )


def strip(d):
    return {k: v for k, v in d.items() if k not in ("rec", "qf")}


def bd_psnr(r1, p1, r2, p2):
    """Bjontegaard delta-PSNR of curve 2 w.r.t. curve 1 (cubic fit in log10 rate)."""
    l1, l2 = np.log10(r1), np.log10(r2)
    c1, c2 = np.polyfit(l1, p1, 3), np.polyfit(l2, p2, 3)
    lo, hi = max(l1.min(), l2.min()), min(l1.max(), l2.max())
    i1 = np.polyval(np.polyint(c1), hi) - np.polyval(np.polyint(c1), lo)
    i2 = np.polyval(np.polyint(c2), hi) - np.polyval(np.polyint(c2), lo)
    return (i2 - i1) / (hi - lo)


def bd_rate(r1, p1, r2, p2):
    """Bjontegaard delta-rate (%) of curve 2 w.r.t. curve 1."""
    l1, l2 = np.log10(r1), np.log10(r2)
    c1, c2 = np.polyfit(p1, l1, 3), np.polyfit(p2, l2, 3)
    lo, hi = max(min(p1), min(p2)), min(max(p1), max(p2))
    i1 = np.polyval(np.polyint(c1), hi) - np.polyval(np.polyint(c1), lo)
    i2 = np.polyval(np.polyint(c2), hi) - np.polyval(np.polyint(c2), lo)
    return (10 ** ((i2 - i1) / (hi - lo)) - 1) * 100


def main():
    out = {"env": dict(python=platform.python_version(), numpy=np.__version__,
                       machine=platform.machine(), platform=platform.platform())}
    imgs = {k: load(v) for k, v in IMAGES.items()}

    # ---------------- Ablation 1: quantisation table
    qualities = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    flats = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    a1 = {}
    for name, img in imgs.items():
        r = {"ijg": {}, "flat": {}, "special": {}}
        for q in qualities:
            r["ijg"][q] = strip(run_codec(img, ijg_table(q)))
        for s in flats:
            r["flat"][s] = strip(run_codec(img, np.full((8, 8), float(s))))
        r["special"]["standard"] = strip(run_codec(img, Q_STD))
        r["special"]["reversed"] = strip(run_codec(img, Q_STD[::-1, ::-1].copy()))
        r["special"]["transposed"] = strip(run_codec(img, Q_STD.T.copy()))
        r["special"]["flat_mean"] = strip(run_codec(img, np.full((8, 8), float(round(Q_STD.mean())))))
        r["special"]["dc_only_fine"] = strip(run_codec(img, np.where(np.arange(64).reshape(8, 8) == 0, 1.0, Q_STD)))
        ijg_r = np.array([r["ijg"][q]["bpp"] for q in qualities]); ijg_p = np.array([r["ijg"][q]["psnr"] for q in qualities])
        fl_r = np.array([r["flat"][s]["bpp"] for s in flats]); fl_p = np.array([r["flat"][s]["psnr"] for s in flats])
        r["bd_psnr_flat_vs_ijg"] = float(bd_psnr(ijg_r, ijg_p, fl_r, fl_p))
        r["bd_rate_flat_vs_ijg"] = float(bd_rate(ijg_r, ijg_p, fl_r, fl_p))
        a1[name] = r
        print(name, "A1 done", r["bd_psnr_flat_vs_ijg"], r["bd_rate_flat_vs_ijg"], flush=True)
    out["quant"] = {"qualities": qualities, "flats": flats, "mean_Q": float(Q_STD.mean()), "results": a1}

    # ---------------- Ablation 2: scan order
    a2 = {}
    for name, img in imgs.items():
        r = {}
        for q in [25, 50, 75, 90]:
            Qt = ijg_table(q)
            res = {s: run_codec(img, Qt, order) for s, order in
                   [("zigzag", ZIGZAG), ("raster", RASTER), ("column", COLUMN)]}
            same = all(np.array_equal(res["zigzag"]["rec"], res[s]["rec"]) for s in ("raster", "column"))
            r[q] = {s: strip(v) for s, v in res.items()}
            r[q]["reconstruction_identical"] = bool(same)
        a2[name] = r
        print(name, "A2 done", flush=True)
    out["scan"] = a2

    # ---------------- Ablation 3: separable vs direct 2D DCT
    reps = {"sep_numpy": 20, "direct_numpy": 20, "sep_loops": 5, "direct_loops": 5}
    a3 = {}
    for name, img in imgs.items():
        shifted = [b.astype(np.float64) - 128.0 for b in blocks_of(img)]
        ref = np.array([dct_separable_numpy(b) for b in shifted])
        r = {}
        for vn, fn in DCT_VARIANTS.items():
            fn(shifted[0])  # warm-up
            times = []
            for _ in range(reps[vn]):
                t0 = time.perf_counter()
                for b in shifted:
                    fn(b)
                times.append(time.perf_counter() - t0)
            Fv = np.array([fn(b) for b in shifted])
            codec = run_codec(img, Q_STD, ZIGZAG, fn)
            qf_ref = np.rint(ref / Q_STD); qf_v = np.rint(Fv / Q_STD)
            r[vn] = dict(
                median_s=statistics.median(times), mean_s=statistics.mean(times),
                std_s=statistics.stdev(times) if len(times) > 1 else 0.0, reps=reps[vn],
                us_per_block=statistics.median(times) / len(shifted) * 1e6,
                max_abs_dF=float(np.abs(Fv - ref).max()),
                qf_mismatches=int(np.sum(qf_ref != qf_v)),
                **{k: codec[k] for k in ("payload_bits", "file_bytes", "bpp", "mse", "psnr")},
            )
            print(name, vn, r[vn]["median_s"], r[vn]["payload_bits"], flush=True)
        a3[name] = r
    out["dct"] = a3
    out["dct_mults_per_block"] = {"separable": 2 * 8 * 8 * 8, "direct": 8 ** 4}

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("saved results.json")


if __name__ == "__main__":
    main()
