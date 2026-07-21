"""
Usage:
    python validate_and_benchmark.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import ptwt
import itertools

from wavelets_custom import (
    wavedec2_new,
    waverec2_new,
    prepare_wavedec2_new,
    prepare_waverec2_new,
)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

WAVELETS = [f"db{i}" for i in range(1, 9)]
SEPARABLE_WAVELETS = list(WAVELETS) 
MODE = "zero" 

DEFAULT_IMG_SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (513, 511),   
    (2048, 2048), 
]

def _flatten_ptwt_coeffs(ptwt_coeffs):
    flat = [ptwt_coeffs[0]]
    for lh, hl, hh in ptwt_coeffs[1:]:
        flat.extend([lh, hl, hh])
    return flat

def _unflatten_to_ptwt(flat_coeffs):
    approx = flat_coeffs[0]
    details = [tuple(flat_coeffs[i:i + 3]) for i in range(1, len(flat_coeffs), 3)]
    return (approx, *details)

def _compare(
    a: torch.Tensor, b: torch.Tensor, label: str, results: list,
    atol: float = 1e-5, rtol: float = 1e-5,
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

def check_correctness(wavelet, img_size, level, device, dtype):
    """Returns a list of (label, ok, detail) for every tensor compared --
    does NOT raise, so a reconstruction failure still shows whether
    decomposition passed."""
    results: list = []

    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

    ref_nested = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
    ref_coeffs = _flatten_ptwt_coeffs(ref_nested)
    
    w_dec_width, w_dec_height, fwd_schedule = prepare_wavedec2_new(
        wavelet, img_size, device, dtype, level=level
    )
    new_coeffs = wavedec2_new(x, w_dec_width, w_dec_height, fwd_schedule)

    labels = ["approx"]
    for k in range(1, level + 1):
        labels += [f"level{k}_lh", f"level{k}_hl", f"level{k}_hh"]
    for label, a, b in zip(labels, ref_coeffs, new_coeffs):
        _compare(a, b, label, results)

    ref_recon = ptwt.waverec2(ref_nested, wavelet)
    w_rec_height, w_rec_width, crop_schedule = prepare_waverec2_new(wavelet, img_size, device, dtype, level=level)
    new_recon = waverec2_new(new_coeffs, w_rec_height, w_rec_width, crop_schedule)
    _compare(ref_recon, new_recon, "reconstruction", results)

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

# Benchmarking
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
    print(f"{'wavelet':6s} {'img_size':14s} {'orig (ms)':>10s} {'new (ms)':>11s} {'new x':>8s}")
    for wavelet in WAVELETS:
        for img_size in ((512, 512), (1024, 1024), (2048, 2048)):
            x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

            w_dec_width, w_dec_height, fwd_schedule = prepare_wavedec2_new(
                wavelet, img_size, device, dtype, level=level
            )
            w_rec_height, w_rec_width, crop_schedule = prepare_waverec2_new(wavelet, img_size, device, dtype, level=level)
            
            def orig(x=x, wavelet=wavelet):
                c = ptwt.wavedec2(x, wavelet, level=level, mode=MODE)
                ptwt.waverec2(c, wavelet)

            def new(x=x, w_dec_width=w_dec_width, w_dec_height=w_dec_height,
                    fwd_schedule=fwd_schedule, w_rec_height=w_rec_height, w_rec_width=w_rec_width, crop_schedule=crop_schedule):
                c = wavedec2_new(x, w_dec_width, w_dec_height, fwd_schedule)
                waverec2_new(c, w_rec_height, w_rec_width, crop_schedule)

            t_orig = _time_cuda(orig)
            t_dense = _time_cuda(new)

            print(f"{wavelet:6s} {str(img_size):14s} {t_orig:10.4f} {t_dense:11.4f} {t_orig / t_dense:7.2f}x")


def _build_device_wt(wavelet_name: str, device, dtype):
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

    def make_new_iteration():
        buf = x.clone()
        dec_args = [prepare_wavedec2_new(w, img_size, device, dtype, level=level) for w in WAVELETS]
        rec_args = [prepare_waverec2_new(w, img_size, device, dtype, level=level) for w in WAVELETS]
 
        def _iter():
            for (w_dec_width, w_dec_height, fwd_schedule), (w_rec_height, w_rec_width, crop_schedule) in zip(dec_args, rec_args):
                c = wavedec2_new(buf, w_dec_width, w_dec_height, fwd_schedule)
                waverec2_new(c, w_rec_height, w_rec_width, crop_schedule)
 
        return _iter

    for name, maker in (
        ("original", make_orig_iteration),
        ("new", make_new_iteration),
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

    img_sizes = list(DEFAULT_IMG_SIZES)
    device = torch.device("cuda")
    dtype = torch.float32
    level = 4

    run_correctness_suite(device, dtype, level, img_sizes)
    
    benchmark_single_call(device, dtype, level)

    benchmark_graph_replay(device, dtype, level, img_size=(2048, 2048))

if __name__ == "__main__":
    main()