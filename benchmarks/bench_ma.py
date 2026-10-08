"""
Standalone benchmark: xupy.ma (GPU) vs raw cupy vs numpy.ma (host).

Usage: python benchmarks/bench_ma.py [--sizes 500,2000,4000] [--repeat 10] [--no-numpy]

Sizes are n for n x n float64 arrays.  GPU timings synchronize around each
call (after warm-up); the numpy.ma column is skipped with --no-numpy.
"""
import argparse
import gc
import sys
import time

import numpy as np


def _time(fn, repeat, sync=None):
    """Median wall time of ``fn()`` in ms, after a warm-up call."""
    fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--sizes", default="500,2000,4000", help="comma-separated n for n x n arrays")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--no-numpy", action="store_true", help="skip the numpy.ma column")
    args = ap.parse_args()

    import xupy
    from xupy import _core

    cp = _core._cupy
    if cp is None:
        print("No usable GPU/cupy found: nothing to benchmark.")
        return 0
    M = sys.modules["xupy.ma"]
    sync = cp.cuda.Device().synchronize

    for n in (int(s) for s in args.sizes.split(",")):
        rng = np.random.default_rng(0)
        da, db = rng.random((n, n)) + 0.1, rng.random((n, n)) + 0.1
        ma_, mb_ = rng.random((n, n)) < 0.1, rng.random((n, n)) < 0.1
        with xupy.backend("gpu"):
            xa, xb = M.masked_array(da, mask=ma_), M.masked_array(db, mask=mb_)
        ca, cb = cp.asarray(da), cp.asarray(db)
        ccond = ca > 0.5
        xcond = xa.data > 0.5
        if args.no_numpy:
            na = nb = ncond = None
        else:
            na, nb = np.ma.masked_array(da, mask=ma_), np.ma.masked_array(db, mask=mb_)
            ncond = da > 0.5

        ops = [
            ("a + b", lambda: xa + xb, lambda: ca + cb, lambda: na + nb),
            ("sqrt(a)", lambda: M.sqrt(xa), lambda: cp.sqrt(ca), lambda: np.ma.sqrt(na)),
            ("sum(axis=0)", lambda: xa.sum(axis=0), lambda: ca.sum(axis=0), lambda: na.sum(axis=0)),
            ("mean()", lambda: xa.mean(), lambda: ca.mean(), lambda: na.mean()),
            ("std(axis=1)", lambda: xa.std(axis=1), lambda: ca.std(axis=1), lambda: na.std(axis=1)),
            ("where(c, a, b)", lambda: M.where(xcond, xa, xb), lambda: cp.where(ccond, ca, cb),
             lambda: np.ma.where(ncond, na, nb)),
            ("average(axis=0)", lambda: M.average(xa, axis=0), lambda: cp.average(ca, axis=0),
             lambda: np.ma.average(na, axis=0)),
            ("median(axis=0)", lambda: sys.modules["xupy.ma.extras"].median(xa, axis=0),
             lambda: cp.median(ca, axis=0), lambda: np.ma.median(na, axis=0)),
        ]

        print(f"\n=== {n} x {n} float64 (repeat={args.repeat}, median ms) ===")
        hdr = f"{'op':<17}{'xupy.ma':>10}{'cupy':>10}{'numpy.ma':>11}{'xupy/cupy':>11}{'np.ma/xupy':>12}"
        print(hdr)
        print("-" * len(hdr))
        with xupy.backend("gpu"):
            for name, fx, fc, fn in ops:
                tx = _time(fx, args.repeat, sync)
                tc = _time(fc, args.repeat, sync)
                tn = None if args.no_numpy else _time(fn, max(1, min(args.repeat, 3)))
                s_n = f"{tn:11.3f}" if tn is not None else f"{'-':>11}"
                s_sp = f"{tn / tx:12.1f}" if tn is not None else f"{'-':>12}"
                print(f"{name:<17}{tx:10.3f}{tc:10.3f}{s_n}{tx / tc:11.2f}{s_sp}")

        del xa, xb, ca, cb, ccond, xcond, na, nb, da, db, ma_, mb_
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    sys.exit(main())
