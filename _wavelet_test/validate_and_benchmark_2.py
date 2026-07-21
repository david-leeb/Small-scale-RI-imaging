"""
Validates that fast_wavelets_2d.wavedec2/waverec2 (via Wavelet2D) produce
results IDENTICAL to ptwt.wavedec2/waverec2(..., mode="zero") for db1-db8
at your actual decomposition level and image sizes, then benchmarks both,
including a CUDA-graph replay benchmark that mirrors ProxOpSARAPos's real
usage pattern (one graph capture, `max_iter` replays per __call__).

I could not run this myself: this sandbox has no GPU and no network
access to install torch/ptwt. Run it yourself before trusting this in
your pipeline.

Usage:
    python validate_and_benchmark.py --quick          # CPU correctness only, no timing
    python validate_and_benchmark.py                  # full GPU correctness + benchmarks
    python validate_and_benchmark.py --deterministic   # force deterministic cuDNN algos
    python validate_and_benchmark.py --img 800 600 --level 4
"""
from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F
import ptwt

from fast_wavelets_2d import (
    SeparableWavelet2D,
    Wavelet2D,
    build_2d_filters,
    build_schedules,
    build_separable_filters,
    make_wavelet_transform,
    wavedec2,
    wavedec2_separable,
    waverec2,
    waverec2_separable,
)

WAVELETS = [f"db{i}" for i in range(1, 9)]
SEPARABLE_WAVELETS = list(WAVELETS)  # db1 now goes through the interleave path -- see fast_wavelets_2d.py
MODE = "zero"  # the only mode this module (and ProxOpSARAPos) uses

# Separable isn't bit-identical to ptwt (different float summation
# order); this tolerance sits comfortably above the ~1e-7 GPU noise floor
# characterized via check_op_level_determinism/check_full_call_repro
# while still catching a genuine several-orders-of-magnitude-larger bug
# (the kind db1 showed).
SEPARABLE_ATOL = 1e-4
SEPARABLE_RTOL = 1e-3

DEFAULT_IMG_SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (513, 511),   # mixed odd parity -- exercises the asymmetric-pad branch
    (2048, 2048),  # actual production size -- never skip this one
]


# --------------------------------------------------------------------------- #
# Flat-list <-> ptwt nested-tuple conversion, used only in this validation
# script so both sides of the comparison can be checked tensor-by-tensor.
# --------------------------------------------------------------------------- #
def _flatten_ptwt_coeffs(ptwt_coeffs):
    flat = [ptwt_coeffs[0]]
    for lh, hl, hh in ptwt_coeffs[1:]:
        flat.extend([lh, hl, hh])
    return flat


def _unflatten_to_ptwt(flat_coeffs):
    approx = flat_coeffs[0]
    details = [tuple(flat_coeffs[i:i + 3]) for i in range(1, len(flat_coeffs), 3)]
    return (approx, *details)


# --------------------------------------------------------------------------- #
# Non-raising comparison so one failing tensor doesn't hide the others,
# and so we always get the actual magnitude of a mismatch.
# --------------------------------------------------------------------------- #
def _compare(
    a: torch.Tensor, b: torch.Tensor, label: str, results: list,
    atol: float = 0.0, rtol: float = 0.0,
) -> None:
    """atol=rtol=0 (the default) requires bit-exact equality -- used for
    the dense Wavelet2D path. Pass nonzero atol/rtol for the separable
    path, which is mathematically identical but not bit-identical."""
    if a.shape != b.shape:
        results.append((label, False, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"))
        return

    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else (0.0 if max_abs == 0 else float("inf"))
    n_mismatch = int((diff != 0).sum().item())
    detail = (f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} "
              f"mismatched_elems={n_mismatch}/{a.numel()}")

    if atol == 0.0 and rtol == 0.0:
        ok = torch.equal(a, b)
    else:
        ok = bool(torch.all(diff <= (atol + rtol * b.float().abs())))

    results.append((label, ok, None if ok else detail))


def check_schedule_matches_runtime(wavelet, img_size, level, device, dtype):
    """Sanity-check the pure-Python shape arithmetic in build_schedules
    against what ptwt.wavedec2 actually produces at runtime, for every
    level -- independent of whether the numeric conv results match."""
    _, _, filt_len = build_2d_filters(wavelet, device, dtype)
    fwd_schedule, crop_schedule, full_shapes = build_schedules(img_size, filt_len, level)

    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)
    coeffs = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)

    actual_approx = tuple(coeffs[0].shape[-2:])
    assert actual_approx == full_shapes[level], (
        f"{wavelet} {img_size} lvl{level}: approx shape {actual_approx} "
        f"!= predicted {full_shapes[level]}"
    )
    for k in range(1, level + 1):
        predicted = full_shapes[level - k + 1]
        for name, t in zip(("lh", "hl", "hh"), coeffs[k]):
            actual = tuple(t.shape[-2:])
            assert actual == predicted, (
                f"{wavelet} {img_size} lvl{level}: level {k} {name} shape "
                f"{actual} != predicted {predicted}"
            )
    return fwd_schedule, crop_schedule


def check_correctness(wavelet, img_size, level, device, dtype):
    """Returns a list of (label, ok, detail) for every tensor compared --
    does NOT raise, so a reconstruction failure still shows whether
    decomposition passed."""
    results: list = []
    fwd_schedule, crop_schedule = check_schedule_matches_runtime(
        wavelet, img_size, level, device, dtype
    )
    dec_filt, rec_filt, _ = build_2d_filters(wavelet, device, dtype)

    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

    ref_nested = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
    ref_coeffs = _flatten_ptwt_coeffs(ref_nested)
    fast_coeffs = wavedec2(x, dec_filt, fwd_schedule)

    labels = ["approx"]
    for k in range(1, level + 1):
        labels += [f"level{k}_lh", f"level{k}_hl", f"level{k}_hh"]
    for label, a, b in zip(labels, ref_coeffs, fast_coeffs):
        _compare(a, b, label, results)

    ref_recon = ptwt.waverec2(ref_nested, wavelet)
    fast_recon = waverec2(fast_coeffs, rec_filt, crop_schedule)
    _compare(ref_recon, fast_recon, "reconstruction", results)

    # Cross-checks localize a failure to one half (dec vs rec).
    cross_recon_a = ptwt.waverec2(_unflatten_to_ptwt(fast_coeffs), wavelet)  # fast dec -> orig rec
    _compare(ref_recon, cross_recon_a, "cross_fastdec_origrec", results)
    cross_recon_b = waverec2(ref_coeffs, rec_filt, crop_schedule)  # orig dec -> fast rec
    _compare(ref_recon, cross_recon_b, "cross_origdec_fastrec", results)

    return results


def run_correctness_suite(device, dtype, level, img_sizes):
    print(f"\n=== Correctness: device={device}, dtype={dtype}, level={level} ===")
    n_ok, n_fail = 0, 0
    failing_combos = []
    for wavelet, img_size in itertools.product(WAVELETS, img_sizes):
        results = check_correctness(wavelet, img_size, level, device, dtype)
        failed = [r for r in results if not r[1]]
        if failed:
            n_fail += 1
            failing_combos.append((wavelet, img_size))
            print(f"  FAIL  {wavelet:5s} {img_size}")
            for label, _, detail in failed:
                print(f"           {label:24s} {detail}")
        else:
            n_ok += 1
    print(f"  {n_ok} passed, {n_fail} failed ({n_ok + n_fail} total combinations)")
    return failing_combos


def check_correctness_separable(wavelet, img_size, level, device, dtype,
                                 atol=SEPARABLE_ATOL, rtol=SEPARABLE_RTOL):
    """Same structure as check_correctness, but for wavedec2_separable/
    waverec2_separable, compared against ptwt with a tolerance instead of
    exact equality."""
    results: list = []
    fwd_schedule, crop_schedule = check_schedule_matches_runtime(
        wavelet, img_size, level, device, dtype
    )
    (w_dec_width, w_dec_height, w_rec_height, w_rec_width,
     _, rec_lo, rec_hi) = build_separable_filters(wavelet, device, dtype)

    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

    ref_nested = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
    ref_coeffs = _flatten_ptwt_coeffs(ref_nested)
    sep_coeffs = wavedec2_separable(x, w_dec_width, w_dec_height, fwd_schedule)

    labels = ["approx"]
    for k in range(1, level + 1):
        labels += [f"level{k}_lh", f"level{k}_hl", f"level{k}_hh"]
    for label, a, b in zip(labels, ref_coeffs, sep_coeffs):
        _compare(a, b, label, results, atol, rtol)

    ref_recon = ptwt.waverec2(ref_nested, wavelet)
    sep_recon = waverec2_separable(sep_coeffs, rec_lo, rec_hi, w_rec_height, w_rec_width, crop_schedule)
    _compare(ref_recon, sep_recon, "reconstruction", results, atol, rtol)

    cross_recon_a = ptwt.waverec2(_unflatten_to_ptwt(sep_coeffs), wavelet)  # sep dec -> orig rec
    _compare(ref_recon, cross_recon_a, "cross_sepdec_origrec", results, atol, rtol)
    cross_recon_b = waverec2_separable(ref_coeffs, rec_lo, rec_hi, w_rec_height, w_rec_width, crop_schedule)  # orig dec -> sep rec
    _compare(ref_recon, cross_recon_b, "cross_origdec_seprec", results, atol, rtol)

    return results


def run_correctness_suite_separable(device, dtype, level, img_sizes,
                                     atol=SEPARABLE_ATOL, rtol=SEPARABLE_RTOL):
    print(f"\n=== Separable correctness (tolerance atol={atol:.0e}, rtol={rtol:.0e}): "
          f"device={device}, dtype={dtype}, level={level} ===")
    n_ok, n_fail = 0, 0
    for wavelet, img_size in itertools.product(SEPARABLE_WAVELETS, img_sizes):
        results = check_correctness_separable(wavelet, img_size, level, device, dtype, atol, rtol)
        failed = [r for r in results if not r[1]]
        if failed:
            n_fail += 1
            print(f"  FAIL  {wavelet:5s} {img_size}")
            for label, _, detail in failed:
                print(f"           {label:24s} {detail}")
        else:
            n_ok += 1
    print(f"  {n_ok} passed, {n_fail} failed ({n_ok + n_fail} total combinations)")
    return n_fail == 0


# --------------------------------------------------------------------------- #
# Determinism diagnostics -- kept from earlier rounds: db3/db4 showed
# ptwt.waverec2 itself is not bit-reproducible run-to-run on at least one
# GPU/cuDNN build (~1e-7). If run_correctness_suite reports a
# "reconstruction" mismatch at that same tiny magnitude, these confirm
# it's inherited from ptwt, not introduced by this module.
# --------------------------------------------------------------------------- #
def check_op_level_determinism(device, dtype):
    if device.type != "cuda":
        return
    print("\n=== Op-level determinism check (conv_transpose2d alone, no ptwt) ===")
    any_bad = False
    for filt_len in (2, 4, 6, 8, 10, 12, 14, 16):
        for spatial in (16, 32, 64, 128, 256):
            x = torch.randn(1, 4, spatial, spatial, device=device, dtype=dtype)
            filt = torch.randn(4, 1, filt_len, filt_len, device=device, dtype=dtype)
            a = F.conv_transpose2d(x, filt, stride=2)
            b = F.conv_transpose2d(x, filt, stride=2)
            if not torch.equal(a, b):
                any_bad = True
                diff = (a - b).abs().max().item()
                print(f"  NON-DETERMINISTIC  filt_len={filt_len:2d} spatial={spatial:4d}  "
                      f"max|diff|={diff:.3e}")
    if not any_bad:
        print("  conv_transpose2d was self-consistent for every (filt_len, spatial) tried.")
    else:
        print("  -> conv_transpose2d is non-deterministic on this GPU for at least some "
              "kernel/spatial sizes. This is a property of the cuDNN build, not of "
              "fast_wavelets_2d.py -- ptwt's own waverec2 is equally affected.")


def check_full_call_repro(device, dtype, level, failing_combos):
    if device.type != "cuda" or not failing_combos:
        return
    print("\n=== Repeatability check: does ptwt.waverec2 match itself twice in a row? ===")
    for wavelet, img_size in failing_combos:
        x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)
        coeffs = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
        r1 = ptwt.waverec2(coeffs, wavelet)
        r2 = ptwt.waverec2(coeffs, wavelet)
        if torch.equal(r1, r2):
            print(f"  {wavelet:5s} {img_size}: ptwt.waverec2 IS self-consistent "
                  f"(the mismatch is a real discrepancy -- please share this output)")
        else:
            diff = (r1 - r2).abs().max().item()
            print(f"  {wavelet:5s} {img_size}: ptwt.waverec2 is NOT self-consistent "
                  f"(max|diff|={diff:.3e}) -- confirms GPU floating-point "
                  f"non-determinism unrelated to fast_wavelets_2d.py")


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #
def _time_cuda(fn, n_iters=50, n_warmup=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters  # ms/call


def benchmark_single_call(device, dtype, level):
    print("\n=== Per-call benchmark (single wavedec2+waverec2) ===")
    print(f"{'wavelet':6s} {'img_size':14s} {'orig (ms)':>10s} {'dense (ms)':>11s} "
          f"{'sep (ms)':>9s} {'dense x':>8s} {'sep x':>7s}")
    for wavelet in WAVELETS:
        for img_size in ((512, 512), (1024, 1024), (2048, 2048)):
            x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)
            w2d_dense = Wavelet2D(wavelet, img_size, level, device, dtype)
            is_separable = wavelet in SEPARABLE_WAVELETS
            w2d_sep = SeparableWavelet2D(wavelet, img_size, level, device, dtype) if is_separable else None

            def orig(x=x, wavelet=wavelet):
                c = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
                ptwt.waverec2(c, wavelet)

            def dense(x=x, w2d=w2d_dense):
                c = w2d.decompose(x)
                w2d.reconstruct(c)

            t_orig = _time_cuda(orig)
            t_dense = _time_cuda(dense)

            if is_separable:
                def sep(x=x, w2d=w2d_sep):
                    c = w2d.decompose(x)
                    w2d.reconstruct(c)
                t_sep = _time_cuda(sep)
                print(f"{wavelet:6s} {str(img_size):14s} {t_orig:10.4f} {t_dense:11.4f} "
                      f"{t_sep:9.4f} {t_orig / t_dense:7.2f}x {t_orig / t_sep:6.2f}x")
            else:
                print(f"{wavelet:6s} {str(img_size):14s} {t_orig:10.4f} {t_dense:11.4f} "
                      f"{'n/a':>9s} {t_orig / t_dense:7.2f}x {'n/a':>6s}")


def _build_device_wt(wavelet_name: str, device, dtype):
    """Builds a device-resident WaveletTensorTuple, the way ProxOpSARAPos
    does -- needed for the 'original' side of the graph-replay benchmark
    since a bare wavelet-name string makes ptwt rebuild filters from
    Python float lists via an unpinned host->device copy on every call,
    which CUDA graph capture refuses to allow."""
    import pywt
    wt = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet(wavelet_name), dtype=dtype)
    wt = ptwt.WaveletTensorTuple(*(
        t.to(device) if isinstance(t, torch.Tensor) else t for t in wt
    ))
    return wt


def benchmark_graph_replay(device, dtype, level, img_size=(512, 512), max_iter=20):
    print(f"\n=== CUDA graph replay benchmark ({max_iter} replays/call), img_size={img_size} ===")
    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

    def make_orig_iteration():
        buf = x.clone()
        wts = [_build_device_wt(w, device, dtype) for w in WAVELETS]

        def _iter():
            for wt in wts:
                c = ptwt.wavedec2(buf, wt, level=level, mode=MODE)
                ptwt.waverec2(c, wt)

        return _iter

    def make_dense_iteration():
        buf = x.clone()
        transforms = [Wavelet2D(w, img_size, level, device, dtype) for w in WAVELETS]

        def _iter():
            for w2d in transforms:
                c = w2d.decompose(buf)
                w2d.reconstruct(c)

        return _iter

    def make_separable_iteration():
        """All 8 wavelets, but db1 still goes through Wavelet2D internally
        (via make_wavelet_transform) -- this is what you'd actually
        deploy, not an all-separable ideal that doesn't support db1."""
        buf = x.clone()
        transforms = [make_wavelet_transform(w, img_size, level, device, dtype) for w in WAVELETS]

        def _iter():
            for sw in transforms:
                c = sw.decompose(buf)
                sw.reconstruct(c)

        return _iter

    for name, maker in (
        ("original", make_orig_iteration),
        ("dense", make_dense_iteration),
        ("separable", make_separable_iteration),
    ):
        iter_fn = maker()
        for _ in range(5):
            iter_fn()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            iter_fn()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(max_iter):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        print(f"  {name:9s}: {total_ms:8.3f} ms total, {total_ms / max_iter:7.4f} ms/replay")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="CPU correctness only, no benchmarks")
    parser.add_argument("--level", type=int, default=4,
                         help="decomposition level (matches ProxOpSARAPos's dec_lev default)")
    parser.add_argument("--img", type=int, nargs=2, default=None,
                         help="extra image size to add to the correctness sweep, e.g. --img 800 600")
    parser.add_argument("--deterministic", action="store_true",
                         help="force torch/cuDNN deterministic algorithms before running")
    args = parser.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Deterministic algorithms requested (warn_only=True).")

    img_sizes = list(DEFAULT_IMG_SIZES)
    if args.img is not None:
        img_sizes.append(tuple(args.img))

    if args.quick or not torch.cuda.is_available():
        if not torch.cuda.is_available():
            print("CUDA not available -- running CPU correctness check only.")
        run_correctness_suite(torch.device("cpu"), torch.float32, args.level, img_sizes)
        return

    device = torch.device("cuda")
    dtype = torch.float32

    failing_combos = run_correctness_suite(device, dtype, args.level, img_sizes)
    if failing_combos:
        check_op_level_determinism(device, dtype)
        check_full_call_repro(device, dtype, args.level, failing_combos)
        print("\nIf the checks above confirm GPU non-determinism: this affects "
              "ptwt's own waverec2 too, independent of fast_wavelets_2d.py. Try "
              "rerunning with --deterministic to see whether forcing deterministic "
              "cuDNN algorithms resolves it (likely at some throughput cost).")

    run_correctness_suite_separable(device, dtype, args.level, img_sizes)

    benchmark_single_call(device, dtype, args.level)
    # 512: overhead-bound regime (launch overhead dominates in eager mode,
    # already eliminated by the graph). 2048: your actual production size,
    # compute-bound even under graph replay -- this is the number that
    # matters, and where separable should show a real gap over dense.
    benchmark_graph_replay(device, dtype, args.level, img_size=(512, 512))
    benchmark_graph_replay(device, dtype, args.level, img_size=(2048, 2048))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Integration notes for ProxOpSARAPos (in your attached file)
# --------------------------------------------------------------------------- #
#
# In __init__, replace:
#
#     for b in self._wl_dict:
#         wt = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet(b), dtype=dtype)
#         wt = ptwt.WaveletTensorTuple(*(t.to(device) if isinstance(t, torch.Tensor)
#                                         else t for t in wt))
#         self._wavelets.append(wt)
#         ... jit.trace(_traceable_wavedec2, ...) / jit.trace(_traceable_waverec2, ...)
#
# with:
#
#     from fast_wavelets_2d import make_wavelet_transform
#     self._transforms = [
#         make_wavelet_transform(b, img_size, dec_lev, device, dtype)
#         for b in self._wl_dict
#     ]
#
# make_wavelet_transform picks SeparableWavelet2D for everything except
# db1 (which gets Wavelet2D/dense -- see fast_wavelets_2d.py's module
# docstring for why). Use this factory rather than either class directly
# so the db1 exclusion lives in one place. Only do this once
# run_correctness_suite_separable has passed at your actual image size
# (2048x2048 is now in DEFAULT_IMG_SIZES) -- if you want maximum caution,
# use Wavelet2D everywhere instead (still 2-3x over ptwt in eager mode,
# just without separable's additional compute-bound win at large sizes).
#
# Then _wavedec2_dict / _waverec2_dict_into become:
#
#     def _wavedec2_dict(self, x):
#         scaled = x / self._scale_factor
#         coeff = [sw.decompose(scaled) for sw in self._transforms]
#         if self._dirac:
#             coeff.append(scaled.clone())
#         return coeff
#
#     def _waverec2_dict_into(self, y):
#         self._recon.zero_()
#         for sw, y_i in zip(self._transforms, y):
#             self._recon.add_(sw.reconstruct(y_i) / self._scale_factor)
#         if self._dirac:
#             self._recon.add_(y[-1] / self._scale_factor)
#
# _traceable_wavedec2/_traceable_waverec2 and torch.jit.trace can both be
# deleted entirely: they existed specifically to convert ptwt's nested
# (approx, (lh,hl,hh), ...) structure into a uniform List[torch.Tensor]
# for the tracer. decompose() already returns that flat list natively
# (both Wavelet2D and SeparableWavelet2D), so there's nothing left to
# convert.
#
# Filter/schedule construction is now plain Python-level constants baked
# in at construction time, so when _wavedec2_dict/_waverec2_dict_into are
# later called inside torch.cuda.graph(...) capture, only the actual
# F.pad/conv2d/conv_transpose2d/slice kernels get captured -- no filter-
# construction kernels are replayed on every one of the max_iter replays
# per __call__.