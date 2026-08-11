"""
Single-GPU comparison of the ORIGINAL (create_meas_op_ROP_vmap_mf_mod_KB_same_aa,
vmap-based CROP/MROP, no w-stacking, no multi-GPU) and OPTIMIZED
(create_meas_op_ROP_dist, batched matmul, precomputed symmetrisation
indices, torch.distributed-based) RAPHA/MROP measurement operator
implementations, on a REAL dataset. Compares forward time, adjoint time,
and driver memory only -- no numerical/quality comparison, see caveat
below. P, M_K, M_B are derived automatically from your chosen N_ratio/
epsilon_n via the same solve_epsilon_same_aa call the real pipeline uses
-- you supply the tuning knobs, not the derived shapes.

METHODOLOGICAL CHOICES (read before interpreting results):

1. NUFFT layer held constant. The original's constructor accepts
   grid_size/kernel_dim/mode -- signs of an older, non-FINUFFT gridding
   operator this script has no visibility into. To isolate the RAPHA/
   MROP-layer implementation difference specifically, BOTH classes here
   are composed with MeasOpPytorchFinufft (its own **kwargs silently
   absorbs the unused grid_size/kernel_dim/mode arguments). State this
   explicitly if this result is reported.

2. Single-stack (no w-correction) fallback for the optimized side, since
   the original never supported w-stacking at all. With REAL data, this
   is now a deliberate approximation you are choosing to make for this
   specific speed comparison -- your dataset almost certainly has a
   non-negligible w-spread, and neither implementation's timing here
   reflects what a correctly w-corrected reconstruction would cost. This
   is fine for a pure speed/memory comparison (uncorrected NUFFT calls
   are not meaningfully cheaper or more expensive than w-corrected ones
   at the SAME visibility count) but should not be conflated with a
   correctness or reconstruction-quality statement.

3. data["y"] (the acquired visibility VALUES) is not needed and is
   dropped right after loading: both implementations are timed by
   applying forward_op to a synthetic image, not to real visibilities.
   Only the real (u, v) geometry, real weights, and real (flagged)
   ant1/ant2/batches bookkeeping are used.

4. ant1/ant2/batches device placement: kept on CPU (as returned by
   load_dataset(device=torch.device("cpu"))) for the OPTIMIZED class,
   whose _init_rop explicitly expects them on CPU (see its own
   "# CPU tensors" comment) and does its own .to(device) on derived
   index arrays internally; moved to `device` only for the ORIGINAL
   class's construction call, which indexes them directly against GPU
   tensors.

CAVEAT -- NOT numerically equivalent: the optimized class's alpha
includes a `scale_alpha = P_vec ** 0.25` normalisation the original's
gen_ROP does not apply. This script measures implementation EFFICIENCY
only. A correctness/agreement check between the two would require first
reconciling this scaling difference.
"""

import gc
import time
from pathlib import Path
from typing import Tuple, Optional
import sys

import torch
import torch.distributed as dist

file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parents[1]))  # adjust if needed

from src.utils.io_combined import load_dataset
from src.mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa
from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from src.mrop_ri_measurement_operator import weighting_correction
from src.mrop_ri_measurement_operator.src.mrop_dist import create_meas_op_ROP_dist
from src.mrop_ri_measurement_operator.src.mrop_vmap_og import create_meas_op_ROP_vmap_mf_mod_KB_same_aa
from src.utils.gpu_utils_dist import setup_distributed, cleanup_distributed


def _driver_mem_gb(device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**3


def _as_long_tensor(x) -> torch.Tensor:
    """Defensive conversion -- load_dataset's ant1/ant2/batches type
    (numpy array vs torch tensor) isn't verified here; this normalises
    either to a CPU torch.long tensor."""
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.as_tensor(x, dtype=torch.long)


def _time_calls(fn, n_iters: int, n_warmup: int, device) -> Tuple[float, float]:
    """
    Wall-clock timing via perf_counter + synchronize, used identically
    for both classes. The optimized class issues real (if trivially
    single-participant at world_size=1) NCCL collective calls internally
    on every forward_op/adjoint_op call -- a genuine cost of that code
    path, not something to strip out of the measurement.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(n_iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()


def load_real_config(
    data_file: str, img_size: Tuple[int, int], nfreqs: int,
    N_ratio: float, epsilon_n: float, dtype: torch.dtype,
    Q=27, superresolution=1.87, freq_num: int = 0, im_pixel_size=None,
    flag_data_weighting: bool = False, weight_type: str = "natural", weight_robustness: float = 0.0,
    vis_remove=17.7, dl_shift: float = 0.0, dm_shift: float = 0.0, seed: int = 0,
) -> Tuple[dict, dict, int]:
    """
    Loads a real dataset and derives P, M_K, M_B automatically via the
    same solve_epsilon_same_aa call the real pipeline (imager.py) uses --
    mirroring imager()'s own setup phase up to (but not including)
    w-stacking, since neither implementation compared here uses it (see
    module docstring, point 2). Returns (data, ROP_param, K).
    """
    data = load_dataset(
        data_path=data_file, Q=Q, super_resolution=superresolution,
        image_pixel_size=im_pixel_size, img_size=img_size, nfreqs=nfreqs, freq_num=freq_num,
        data_weighting=flag_data_weighting, weight_type=weight_type,
        weight_robustness=weight_robustness, vis_remove=vis_remove,
        dl_shift=dl_shift, dm_shift=dm_shift, dtype=dtype, device=torch.device("cpu"),
    )

    if Q is None:
        assert "Q" in data, "number of antennas Q is not in data and not provided"
        Q = int(data["Q"])

    N = int(img_size[0] * img_size[1])
    K = int(data["nFreqs"])
    V = int(Q * (Q - 1) // 2)

    if "flag" in data and data["flag"] is not None and "B" not in data:
        data["B"] = data["flag"].shape[-1] / V
    assert "B" in data, "number of snapshots B is not in data and not provided"
    B_total = int(data["B"])
    B_per_ch = int(B_total / K)

    ROP_param = dict(
        ROP_type="MROP", Q=Q, B=B_total, N_ratio=N_ratio, epsilon_n=epsilon_n,
        rv_type="unitary", ROP_seed=seed, weight_type=weight_type
    )

    epsilon, P, M_B, M_K = solve_epsilon_same_aa(N, Q, B_per_ch, K, N_ratio=N_ratio, n=epsilon_n, verbose=True)
    ROP_param.update(P=P, M_K=M_K, M_B=M_B, M=M_K * M_B)
    print(f"INFO: derived P={P}, M_K={M_K}, M_B={M_B} (epsilon={epsilon:.4f}) "
          f"at N_ratio={N_ratio}, epsilon_n={epsilon_n}", flush=True)

    data, weight_corr = weighting_correction(data, ROP_param, rapha=True)

    del data["y"]  # not needed -- see module docstring, point 3
    gc.collect()

    data["ant1"] = _as_long_tensor(data["ant1"])
    data["ant2"] = _as_long_tensor(data["ant2"])
    data["batches"] = _as_long_tensor(data["batches"])

    return data, ROP_param, K


def build_original_op(data: dict, ROP_param: dict, K: int, img_size: Tuple[int, int],
                       dtype: torch.dtype, device: torch.device):
    ROP_param_original = {**ROP_param, "same_ab_all": False, "same_ab_B": False, "ROP_vmap_chunk_size": None}
    u = data["u"].to(device=device, dtype=dtype)
    v = data["v"].to(device=device, dtype=dtype)
    nW = data["nW"].to(device=device, dtype=dtype)
    nWimag = data["nWimag"].to(device=device, dtype=dtype)

    cls = create_meas_op_ROP_vmap_mf_mod_KB_same_aa(MeasOpPytorchFinufft)
    return cls(
        ROP_param=ROP_param_original, u=u, v=v, num_chs=K,
        # ant1=data["ant1"], ant2=data["ant2"], batches=data["batches"],
        ant1=data["ant1"].to(device), ant2=data["ant2"].to(device), batches=data["batches"].to(device),
        img_size=img_size, real_flag=True,
        natural_weight=nW, image_weight=nWimag,
        precond_weight=torch.ones(1, 1, device=device, dtype=dtype),
        device=device, dtype=dtype,
    )


def build_optimized_op(data: dict, ROP_param: dict, K: int, img_size: Tuple[int, int],
                        dtype: torch.dtype, device: torch.device, rank: int, world_size: int):
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    u = data["u"].to(device=device, dtype=dtype)
    v = data["v"].to(device=device, dtype=dtype)
    nW = data["nW"].to(device=device, dtype=dtype)
    nWimag = data["nWimag"].to(device=device, dtype=dtype)
    n_vis = u.numel()

    # Single-stack fallback (Section 4.6) -- no w-correction, matching what
    # the original class does implicitly by never supporting w-stacking.
    identity_correction = torch.ones(1, 1, *img_size, dtype=complex_dtype, device=device)
    w_stack_data = {
        "w_center": torch.zeros(1, device=device),
        "corrections": [identity_correction],
        "stack_idx": torch.zeros(n_vis, dtype=torch.int32, device=device),
        "meas_op": [MeasOpPytorchFinufft(
            u=u, v=v, img_size=img_size, real_flag=True, dtype=dtype, device=device,
            natural_weight=nW, image_weight=nWimag,
        )],
    }

    cls = create_meas_op_ROP_dist(MeasOpPytorchFinufft)
    return cls(
        img_size=img_size, w_stack_data=w_stack_data, num_chs=K,
        rank=rank, world_size=world_size, use_ROP=True, ROP_param=ROP_param,
        ant1=data["ant1"], ant2=data["ant2"], batches=data["batches"],  # CPU -- see module docstring, point 4
        device=device, dtype=dtype, real_flag=True, verbose=False,
    )


def compare_original_vs_optimized(
    data_file: str, img_size: Tuple[int, int], nfreqs: int,
    N_ratio: float, epsilon_n: float, dtype: torch.dtype = torch.float32,
    n_iters: int = 20, n_warmup: int = 5, **load_kwargs,
) -> dict:
    """
    Loads a real dataset once, derives P/M_K/M_B, builds both
    implementations on identical real geometry/weights/antenna-pair
    bookkeeping, and times forward_op/adjoint_op separately for each,
    plus driver memory after construction. Must be run via
    `torchrun --nproc_per_node=1 ...` (see _time_calls docstring).
    """
    assert dist.is_initialized(), "Call setup_distributed() first (run via torchrun --nproc_per_node=1)."
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    data, ROP_param, K = load_real_config(
        data_file, img_size, nfreqs, N_ratio, epsilon_n, dtype, **load_kwargs
    )
    x = torch.randn(1, 1, *img_size, device=device, dtype=dtype)

    print(f"\n=== Original vs optimized (real data): K={K}, Q={ROP_param['Q']}, B={ROP_param['B']}, "
          f"P={ROP_param['P']}, M_K={ROP_param['M_K']}, M_B={ROP_param['M_B']}, "
          f"n_vis={data['u'].numel()}, img_size={img_size}, dtype={dtype} ===")

    def _build_and_time(build_fn, label):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        mem_before = _driver_mem_gb(device)

        op = build_fn()

        torch.cuda.synchronize(device)
        mem_gb = _driver_mem_gb(device) - mem_before

        fwd_mean_ms, fwd_std_ms = _time_calls(lambda: op.forward_op(x), n_iters, n_warmup, device)
        y = op.forward_op(x)  # untimed -- realistic adjoint input
        adj_mean_ms, adj_std_ms = _time_calls(lambda: op.adjoint_op(y), n_iters, n_warmup, device)

        print(f"  {label}: fwd={fwd_mean_ms:.4f} ms (std {fwd_std_ms:.4f})  "
              f"adj={adj_mean_ms:.4f} ms (std {adj_std_ms:.4f})  mem={mem_gb:.3f} GB")

        del op, y
        gc.collect()
        torch.cuda.empty_cache()
        return fwd_mean_ms, fwd_std_ms, adj_mean_ms, adj_std_ms, mem_gb

    orig_fwd_ms, orig_fwd_std, orig_adj_ms, orig_adj_std, orig_mem_gb = _build_and_time(
        lambda: build_original_op(data, ROP_param, K, img_size, dtype, device), "original"
    )
    opt_fwd_ms, opt_fwd_std, opt_adj_ms, opt_adj_std, opt_mem_gb = _build_and_time(
        lambda: build_optimized_op(data, ROP_param, K, img_size, dtype, device, rank, world_size), "optimized"
    )

    fwd_speedup = orig_fwd_ms / opt_fwd_ms
    adj_speedup = orig_adj_ms / opt_adj_ms
    print(f"  fwd_speedup={fwd_speedup:.2f}x  adj_speedup={adj_speedup:.2f}x")

    return {
        "original_fwd_ms": orig_fwd_ms, "original_fwd_std": orig_fwd_std,
        "original_adj_ms": orig_adj_ms, "original_adj_std": orig_adj_std,
        "original_mem_gb": orig_mem_gb,
        "optimized_fwd_ms": opt_fwd_ms, "optimized_fwd_std": opt_fwd_std,
        "optimized_adj_ms": opt_adj_ms, "optimized_adj_std": opt_adj_std,
        "optimized_mem_gb": opt_mem_gb,
        "fwd_speedup": fwd_speedup, "adj_speedup": adj_speedup,
    }


def print_original_vs_optimized_table(result: dict):
    print(r"\begin{tabular}{lrrr}")
    print(r"\hline\hline")
    print(r"Implementation & \shortstack{Fwd.\ Time\\(ms)} & \shortstack{Adj.\ Time\\(ms)} & Memory (GB) \\")
    print(r"\hline")
    print(f"Original & "
          f"${result['original_fwd_ms']:.3f} \\pm {result['original_fwd_std']:.3f}$ & "
          f"${result['original_adj_ms']:.3f} \\pm {result['original_adj_std']:.3f}$ & "
          f"{result['original_mem_gb']:.3f} \\\\")
    print(f"Optimized & "
          f"${result['optimized_fwd_ms']:.3f} \\pm {result['optimized_fwd_std']:.3f}$ "
          f"({result['fwd_speedup']:.2f}$\\times$) & "
          f"${result['optimized_adj_ms']:.3f} \\pm {result['optimized_adj_std']:.3f}$ "
          f"({result['adj_speedup']:.2f}$\\times$) & "
          f"{result['optimized_mem_gb']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    rank, world_size, device = setup_distributed()
    assert world_size == 1, "Run via torchrun --nproc_per_node=1 -- this comparison is single-GPU only."

    print("---single---")
    result = compare_original_vs_optimized(
        data_file="../data/273-X08-dmog",   # PLACEHOLDER
        img_size=(1024, 1024),          # PLACEHOLDER
        nfreqs=64,                      # PLACEHOLDER
        N_ratio=1.0,                    # PLACEHOLDER -- your chosen operating point
        epsilon_n=1.0,                  # PLACEHOLDER
        dtype=torch.float32,            # PLACEHOLDER
    )
    print_original_vs_optimized_table(result)
    
    print("---double---")
    result = compare_original_vs_optimized(
        data_file="../data/273-X08-dmog",   # PLACEHOLDER
        img_size=(1024, 1024),          # PLACEHOLDER
        nfreqs=64,                      # PLACEHOLDER
        N_ratio=1.0,                    # PLACEHOLDER -- your chosen operating point
        epsilon_n=1.0,                  # PLACEHOLDER
        dtype=torch.float64,            # PLACEHOLDER
    )
    print_original_vs_optimized_table(result)

    cleanup_distributed()