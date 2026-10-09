"""Re-run Ablation 3 with matched loop styles and compute supporting statistics."""
import json, statistics, time
import numpy as np
import ablation as A
from codec_parameters import Q as Q_STD, ZIGZAG

d = json.load(open(A.os.path.join(A.HERE, "results.json")))
imgs = {k: A.load(v) for k, v in A.IMAGES.items()}

# ---- Ablation 3 re-run (all four variants, interleaved repetitions)
reps = {"sep_numpy": 30, "direct_numpy": 30, "sep_loops": 7, "direct_loops": 7}
a3 = {}
for name, img in imgs.items():
    shifted = [b.astype(np.float64) - 128.0 for b in A.blocks_of(img)]
    ref = np.array([A.dct_separable_numpy(b) for b in shifted])
    times = {vn: [] for vn in A.DCT_VARIANTS}
    for fn in A.DCT_VARIANTS.values():
        fn(shifted[0])
    for rep in range(max(reps.values())):
        for vn, fn in A.DCT_VARIANTS.items():
            if rep >= reps[vn]:
                continue
            t0 = time.perf_counter()
            for b in shifted:
                fn(b)
            times[vn].append(time.perf_counter() - t0)
    r = {}
    for vn, fn in A.DCT_VARIANTS.items():
        Fv = np.array([fn(b) for b in shifted])
        codec = A.run_codec(img, Q_STD, ZIGZAG, fn)
        t = times[vn]
        r[vn] = dict(median_s=statistics.median(t), mean_s=statistics.mean(t), std_s=statistics.stdev(t),
                     reps=len(t), us_per_block=statistics.median(t) / len(shifted) * 1e6,
                     max_abs_dF=float(np.abs(Fv - ref).max()),
                     qf_mismatches=int(np.sum(np.rint(ref / Q_STD) != np.rint(Fv / Q_STD))),
                     **{k: codec[k] for k in ("payload_bits", "file_bytes", "bpp", "mse", "psnr")})
        print(name, vn, {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in r[vn].items()}, flush=True)
    a3[name] = r
d["dct"] = a3

# ---- nonzero-probability maps and orientation statistics at the standard table
maps = {}
for name, img in imgs.items():
    qf = A.run_codec(img)["qf"]  # (4096, 8, 8)
    p = (qf != 0).mean(axis=0)
    maps[name] = dict(p=p.round(4).tolist(),
                      first_row_nz=float((qf[:, 0, 1:] != 0).sum(axis=1).mean()),
                      first_col_nz=float((qf[:, 1:, 0] != 0).sum(axis=1).mean()),
                      nz_per_block=float((qf != 0).sum(axis=(1, 2)).mean()),
                      energy_row=float((qf[:, 0, 1:].astype(float) ** 2).sum(axis=1).mean()),
                      energy_col=float((qf[:, 1:, 0].astype(float) ** 2).sum(axis=1).mean()))
    print(name, {k: v for k, v in maps[name].items() if k != "p"})
    print(np.array(maps[name]["p"]))
d["nonzero_maps"] = maps

# ---- PSNR of the flat-table curve at the rate of each IJG point (log-rate interpolation)
q = d["quant"]
for name, r in q["results"].items():
    fr = np.array([r["flat"][str(s)]["bpp"] for s in q["flats"]])
    fp = np.array([r["flat"][str(s)]["psnr"] for s in q["flats"]])
    o = np.argsort(fr)
    matched = {}
    for qq in q["qualities"]:
        x = r["ijg"][str(qq)]
        if fr.min() <= x["bpp"] <= fr.max():
            pf = float(np.interp(np.log(x["bpp"]), np.log(fr[o]), fp[o]))
            matched[qq] = dict(bpp=x["bpp"], psnr_ijg=x["psnr"], psnr_flat=pf, delta=pf - x["psnr"])
    r["matched"] = matched
    lo = [qq for qq in q["qualities"] if qq <= 50]
    hi = [qq for qq in q["qualities"] if qq >= 70]
    for label, sel in (("low", lo), ("high", hi)):
        ir = np.array([r["ijg"][str(k)]["bpp"] for k in sel]); ip = np.array([r["ijg"][str(k)]["psnr"] for k in sel])
        r[f"bd_psnr_{label}"] = float(A.bd_psnr(ir, ip, fr, fp))
    print(name, "matched:", {k: round(v["delta"], 3) for k, v in matched.items()},
          "BD low/high:", round(r["bd_psnr_low"], 3), round(r["bd_psnr_high"], 3))
    # tokens per block at the operating points closest in rate
    print(name, "tokens IJG50:", r["ijg"]["50"]["tokens_per_block"], r["ijg"]["50"]["zero_tokens_per_block"],
          "| flat24:", r["flat"]["24"]["tokens_per_block"], r["flat"]["24"]["zero_tokens_per_block"],
          "| IJG95:", r["ijg"]["95"]["tokens_per_block"], "| flat4/6:", r["flat"]["4"]["tokens_per_block"], r["flat"]["6"]["tokens_per_block"])

json.dump(d, open(A.os.path.join(A.HERE, "results.json"), "w"), indent=1, default=float)
print("updated results.json")
