import contextlib
import datetime
import gc
from typing import List, Tuple, Optional
import sys
from pathlib import Path
import torch

file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parents[1]))
__package__ = "measOperator"

from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft, SharedNUFFTPlanPair


class _Tee:
    """Writes to multiple streams at once -- lets console output and a
    saved log file both receive every print() inside the `with` block in
    __main__, without touching any of the print() calls themselves."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# --------------------------------------------------------------------------- #
# Synthetic / real uv-coverage sources
# --------------------------------------------------------------------------- #

def _make_synthetic_uv(n_vis: int, device, dtype, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = (torch.rand(n_vis, generator=g) * 2 - 1) * torch.pi
    v = (torch.rand(n_vis, generator=g) * 2 - 1) * torch.pi
    return u.view(1, 1, -1).to(device=device, dtype=dtype), v.view(1, 1, -1).to(device=device, dtype=dtype)


def _make_synthetic_uv_per_stack(n_wstacks: int, n_vis_total: int, device, dtype,
                                  seed: int = 0, imbalanced: bool = False,
                                  imbalance_alpha: float = 0.5):
    """
    Splits n_vis_total across n_wstacks sub-operators, either evenly
    (imbalanced=False) or via a Dirichlet-distributed split (imbalanced=
    True) -- a cheap, data-free approximation of the visibility-count
    imbalance real w-stacks exhibit, for use when a real per-stack
    uv-coverage export (see _load_real_uv_per_stack) isn't available yet.
    Smaller imbalance_alpha -> more skewed split.
    """
    torch.manual_seed(seed)
    if imbalanced:
        weights = torch.distributions.Dirichlet(
            torch.full((n_wstacks,), imbalance_alpha)
        ).sample()
        counts = torch.clamp((weights * n_vis_total).round().long(), min=1)
    else:
        base = n_vis_total // n_wstacks
        counts = torch.full((n_wstacks,), base, dtype=torch.long)

    return [
        _make_synthetic_uv(int(c), device, dtype, seed=seed + i)
        for i, c in enumerate(counts)
    ]


def _load_real_uv_per_stack(path: str, device, dtype):
    """
    Loads a real per-stack (u, v) uv-coverage previously exported from an
    actual dataset run, for use in place of synthetic points -- see the
    module docstring's discussion of when this matters (gpu_method and
    group_size benchmarks specifically). Expected format: a list of
    (u, v) tensor pairs, each shaped (1, 1, n_vis_k), saved via
    torch.save -- e.g. produced once via:

        stacks = []
        for k in range(len(w_stack_data["meas_op"])):
            idx = torch.where(w_stack_data["stack_idx"] == k)[0]
            stacks.append((data_i["u"][:, :, idx].cpu(), data_i["v"][:, :, idx].cpu()))
        torch.save(stacks, "real_uv_per_stack.pt")

    inside process_device_global, right after w_stack_idx is computed.
    """
    stacks = torch.load(path, map_location="cpu")
    return [(u.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)) for u, v in stacks]


def _make_synthetic_image(img_size: Tuple[int, int], device, dtype, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, 1, *img_size, generator=g).to(device=device, dtype=dtype)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype in (torch.float32, torch.float) else torch.complex128


def _driver_mem_gb(device) -> float:
    """Total GPU memory in use, from the CUDA driver's perspective -- this
    is the measurement that actually captures cufinufft's plan memory,
    since it is invisible to torch.cuda.memory_allocated/reserved (see
    Section 4.9's discussion of this exact gap)."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**3

# --------------------------------------------------------------------------- #
# eps / upsampfac execution-time benchmark
# --------------------------------------------------------------------------- #

def _time_calls(fn, n_iters=50, n_warmup=10) -> Tuple[float, float]:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(n_iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()


def benchmark_eps_upsampfac(
    eps_values: Tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    upsampfac_values: Tuple[float, ...] = (1.25, 2.0),
    dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
    img_size: Tuple[int, int] = (4096, 4096),
    n_vis: int = 100_000_000,
    device: torch.device = torch.device("cuda:0"),
    uv_source: str = "synthetic",
    real_uv_path: Optional[str] = None,
    seed: int = 0,
) -> List[dict]:
    """
    Per-call time and driver memory across the full (eps, upsampfac,
    dtype) grid, using the individual-plan path (plan built once at
    construction, timed calls are execute()-only) so results isolate the
    settings' effect on execution time, not plan construction. Purely a
    speed/memory characterisation -- accuracy is covered separately by
    check_correctness (NUFFT-level) and the full-pipeline sweep script
    (reconstruction-level).

    uv_source="real" is recommended for consistency with the other
    parameter benchmarks in this file: real, non-uniform interferometric
    uv-coverage differs from uniform synthetic points in ways that can
    affect execution time. real_uv_path should point to a single
    torch.save'd (u, v) tensor pair, same format as
    benchmark_gpu_method_kerevalmeth.
    """
    results = []
    for dtype in dtypes:
        if uv_source == "real":
            if real_uv_path is None:
                raise ValueError("uv_source='real' requires real_uv_path.")
            stacks = torch.load(real_uv_path, map_location="cpu")
            u, v = stacks[20]
            u, v = u.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)
        else:
            u, v = _make_synthetic_uv(n_vis, device, dtype, seed=seed)
        x = _make_synthetic_image(img_size, device, dtype, seed=seed + 1)

        print(f"\n=== eps/upsampfac benchmark: dtype={dtype}, img_size={img_size}, "
              f"n_vis={u.numel()}, uv_source={uv_source} ===")
        for eps in eps_values:
            for upsampfac in upsampfac_values:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                mem_before = _driver_mem_gb(device)

                op = MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac)

                torch.cuda.synchronize()
                mem_gb = _driver_mem_gb(device) - mem_before

                def call():
                    y = op._GA(x)
                    op._AtGt(y)

                mean_ms, std_ms = _time_calls(call)
                results.append({"dtype": dtype, "eps": eps, "upsampfac": upsampfac,
                                 "mean_ms": mean_ms, "std_ms": std_ms, "mem_gb": mem_gb})
                print(f"  eps={eps:10.0e} upsampfac={upsampfac:6.2f}  {mean_ms:9.4f} ms "
                      f"(std {std_ms:.4f})  {mem_gb:.3f} GB")

                del op
                gc.collect()
                torch.cuda.empty_cache()

    return results


def print_eps_upsampfac_table(results: List[dict]):
    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline\hline")
    print(r"Precision & \texttt{eps} & \texttt{upsampfac} & Time (ms) & Memory (GB) \\")
    print(r"\hline")
    for r in results:
        prec = "single" if r["dtype"] == torch.float32 else "double"
        print(f"{prec} & {r['eps']:.0e} & {r['upsampfac']:.2f} & "
              f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ & {r['mem_gb']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

# --------------------------------------------------------------------------- #
# gpu_method / gpu_kerevalmeth benchmark  (requires the PATCH above)
# --------------------------------------------------------------------------- #

def benchmark_gpu_method_kerevalmeth(
    gpu_methods: Tuple[int, ...] = (1, 2),
    gpu_kerevalmeths: Tuple[int, ...] = (0, 1),
    eps: float = 1e-3,
    upsampfac: float = 1.25,
    dtype: torch.dtype = torch.float32,
    img_size: Tuple[int, int] = (4096, 4096),
    n_vis: int = 100_000_000,
    device: torch.device = torch.device("cuda:0"),
    uv_source: str = "synthetic",
    real_uv_path: Optional[str] = None,
    seed: int = 0,
) -> List[dict]:
    """
    Per-call time across the (gpu_method, gpu_kerevalmeth) grid. Requires
    MeasOpPytorchFinufft to accept these as constructor kwargs -- see the
    PATCH note at the top of this file; raises a clear error if not patched.

    uv_source="real" is recommended here over "synthetic": gpu_method
    trades off a non-uniform-points-driven strategy against a
    shared-memory/binned one, and their relative performance depends on
    how points are spatially distributed (binning/load-balance is
    sensitive to point density), which uniform random points in
    [-pi,pi]^2 do not represent well relative to real, non-uniform
    interferometric uv-coverage. real_uv_path should point to a single
    torch.save'd (u, v) tensor pair (not the per-stack list format used
    by benchmark_group_size below). A run with uv_source="synthetic" is
    still informative as a first pass, but treat its magnitude as
    provisional until cross-checked against real coverage.
    """
    if uv_source == "real":
        if real_uv_path is None:
            raise ValueError("uv_source='real' requires real_uv_path.")
        w_stacks = torch.load(real_uv_path, map_location="cpu")
        u, v = w_stacks[20]
        u, v = u.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)
    else:
        u, v = _make_synthetic_uv(n_vis, device, dtype, seed=seed)

    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 1)

    results = []
    print(f"\n=== gpu_method/gpu_kerevalmeth benchmark: dtype={dtype}, img_size={img_size}, "
          f"n_vis={u.numel()}, uv_source={uv_source} ===")
    for gm in gpu_methods:
        for gk in gpu_kerevalmeths:
            try:
                op = MeasOpPytorchFinufft(
                    u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac,
                    gpu_method=gm, gpu_kerevalmeth=gk,  # requires PATCH
                )
            except TypeError as e:
                raise TypeError(
                    "MeasOpPytorchFinufft does not accept gpu_method/gpu_kerevalmeth -- "
                    "apply the PATCH described in this file's module docstring first."
                ) from e

            def call():
                y = op._GA(x)
                op._AtGt(y)

            mean_ms, std_ms = _time_calls(call)
            results.append({"gpu_method": gm, "gpu_kerevalmeth": gk, "mean_ms": mean_ms, "std_ms": std_ms})
            print(f"  gpu_method={gm} gpu_kerevalmeth={gk}  {mean_ms:9.4f} ms  (std {std_ms:.4f})")

            del op
            torch.cuda.empty_cache()

    return results


def print_gpu_method_table(results: List[dict]):
    print(r"\begin{tabular}{rrr}")
    print(r"\hline\hline")
    print(r"\shortstack{\texttt{gpu\_}\\\texttt{method}} & "
          r"\shortstack{\texttt{gpu\_}\\\texttt{kerevalmeth}} & Time (ms) \\")
    print(r"\hline")
    for r in results:
        print(f"{r['gpu_method']} & {r['gpu_kerevalmeth']} & "
              f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# --------------------------------------------------------------------------- #
# Grouped plan sharing: memory and time vs. group size
# --------------------------------------------------------------------------- #

def benchmark_num_plans(
    num_plans: Tuple[int, ...] = (40, 20, 10, 5, 2, 1, 0),
    n_wstacks: int = 32,
    eps: float = 1e-3,
    upsampfac: float = 1.25,
    img_size: Tuple[int, int] = (4096, 4096),
    n_vis_total: int = 500_000_000,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda:0"),
    uv_source: str = "synthetic_imbalanced",
    real_uv_path: Optional[str] = None,
    n_gpus: int = 1,
    gpu_id: int = 0,
    seed: int = 0,
) -> List[dict]:
    """
    Sweeps the grouped-plan-sharing group size g in {1, ..., n_wstacks}
    (Section 4.9): g=1 recovers one dedicated plan per stack (no repeated
    setpts); g>=n_wstacks recovers a single fully-shared plan pair.
    Reports, per g: driver memory used by the constructed plans (the
    quantity that motivated grouping in the first place, and is invisible
    to torch's own allocator -- see _driver_mem_gb), and per-call
    (forward+adjoint, all stacks) execution time.

    uv_source options:
      "synthetic"            -- n_wstacks equal-sized synthetic stacks.
      "synthetic_imbalanced" -- Dirichlet-split synthetic stacks (default),
                                 a data-free approximation of the
                                 visibility-count imbalance real w-stacks
                                 exhibit.
      "real"                 -- per-stack uv-coverage loaded from
                                 real_uv_path (see module docstring); the
                                 most faithful option, since imbalance
                                 interacts directly with which stacks share
                                 a plan at a given group size.
    """
    if uv_source == "real":
        if real_uv_path is None:
            raise ValueError("uv_source='real' requires real_uv_path.")
        uv_per_stack = _load_real_uv_per_stack(real_uv_path, device, dtype)
        n_wstacks = len(uv_per_stack)
        
        simulated_uv_per_stack = []
        for u_stack, v_stack in uv_per_stack:
            # Keep ONLY this GPU's visibilities and discard the rest
            u_gpu = u_stack[..., gpu_id::n_gpus]
            v_gpu = v_stack[..., gpu_id::n_gpus]

            simulated_uv_per_stack.append((u_gpu, v_gpu))

        uv_per_stack = simulated_uv_per_stack
        
    elif uv_source == "synthetic_imbalanced":
        uv_per_stack = _make_synthetic_uv_per_stack(
            n_wstacks, n_vis_total, device, dtype, seed=seed, imbalanced=True
        )
    else:
        uv_per_stack = _make_synthetic_uv_per_stack(
            n_wstacks, n_vis_total, device, dtype, seed=seed, imbalanced=False
        )

    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 100)
    results = []

    print(f"\n=== Num plans sweep: n_wstacks={n_wstacks}, img_size={img_size}, "
          f"dtype={dtype}, uv_source={uv_source} ===")

    for p in num_plans:
        if p > n_wstacks:
            continue  # nothing new beyond the fully-shared case already covered by g=n_wstacks

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = _driver_mem_gb(device)

        plan_pairs = None
        if p >= n_wstacks:
            ops = [
                MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac,
                                    shared_plan_pair=None)
                for i, (u, v) in enumerate(uv_per_stack)
            ]
        else:
            plan_pair = SharedNUFFTPlanPair(img_size, dtype, device, eps=eps, upsampfac=upsampfac)
            if p == 0:
                dedicated_indices = set()
            else:
                stride = n_wstacks / p
                dedicated_indices = {int(round(i * stride)) for i in range(p)}
            
            ops = []
            for i, (u, v) in enumerate(uv_per_stack):
                if i in dedicated_indices:
                    plan_to_use = None
                else:
                    plan_to_use = plan_pair
                ops.append(MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac, shared_plan_pair=plan_to_use))

        torch.cuda.synchronize()
        mem_after = _driver_mem_gb(device)
        mem_delta_gb = mem_after - mem_before

        def call_all():
            for op in ops:
                y = op._GA(x)
                op._AtGt(y)

        mean_ms, std_ms = _time_calls(call_all, n_iters=20, n_warmup=5)

        results.append({
            "num_plans": p, "mem_delta_gb": mem_delta_gb,
            "mean_ms": mean_ms, "std_ms": std_ms,
        })
        print(f"  p={p:4d}  mem={mem_delta_gb:7.3f} GB  "
              f"time={mean_ms:9.4f} ms  (std {std_ms:.4f})")

        del ops, plan_pairs
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_num_plans_table(results: List[dict]):
    print(r"\begin{tabular}{rrrr}")
    print(r"\hline\hline")
    print(r"Plans $p$ & Memory (GB) & Time (ms) \\")
    print(r"\hline")
    for r in results:
        print(f"{r['num_plans']} & {r['mem_delta_gb']:.3f} & "
              f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def compare_baseline_vs_optimized(
    baseline_cls,
    optimized_cls,
    optimized_eps: float,
    optimized_upsampfac: float,
    img_size: Tuple[int, int] = (2048, 2048),
    n_vis: int = 100_000_000,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda:0"),
    n_iters: int = 20,
    n_warmup: int = 5,
    uv_source: str = "synthetic",
    real_uv_path: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """
    Compares forward time, adjoint time, and driver memory between the
    ORIGINAL (baseline_cls) and OPTIMIZED (optimized_cls) NUFFT
    sub-operator implementations, on identical synthetic uv-coverage and
    image. Forward and adjoint are timed as two separate calls rather
    than one combined call, so a speedup concentrated in one direction
    (e.g. persistent-plan reuse helping adjoint's gridding step more
    than forward's interpolation step) isn't averaged away.

    See this function's earlier docstring for the memory caveat (baseline
    holds no persistent state, so its memory figure is expected to be
    near zero -- this is a speed-for-memory trade, not a win on both axes).
    """
    if uv_source == "real":
        if real_uv_path is None:
            raise ValueError("uv_source='real' requires real_uv_path.")
        stacks = torch.load(real_uv_path, map_location="cpu")
        u, v = stacks[20]
        u, v = u.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)
    else:
        u, v = _make_synthetic_uv(n_vis, device, dtype, seed=seed)
    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 1)

    def _build_and_time(build_fn, label):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = _driver_mem_gb(device)

        op = build_fn()

        torch.cuda.synchronize()
        mem_gb = _driver_mem_gb(device) - mem_before

        y = op._GA(x)  # untimed -- just to get a realistic adjoint input

        def call_fwd():
            op._GA(x)

        def call_adj():
            op._AtGt(y)

        fwd_mean_ms, fwd_std_ms = _time_calls(call_fwd, n_iters=n_iters, n_warmup=n_warmup)
        adj_mean_ms, adj_std_ms = _time_calls(call_adj, n_iters=n_iters, n_warmup=n_warmup)
        print(f"  {label}: fwd={fwd_mean_ms:.4f} ms (std {fwd_std_ms:.4f})  "
              f"adj={adj_mean_ms:.4f} ms (std {adj_std_ms:.4f})  mem={mem_gb:.3f} GB")

        del op, y
        gc.collect()
        torch.cuda.empty_cache()
        return fwd_mean_ms, fwd_std_ms, adj_mean_ms, adj_std_ms, mem_gb

    print(f"\n=== Baseline vs optimized: img_size={img_size}, n_vis={u.numel()}, "
          f"dtype={dtype}, uv_source={uv_source} ===")
    base_fwd_ms, base_fwd_std, base_adj_ms, base_adj_std, base_mem_gb = _build_and_time(
        lambda: baseline_cls(u, v, img_size, device=device, dtype=dtype), "baseline"
    )
    opt_fwd_ms, opt_fwd_std, opt_adj_ms, opt_adj_std, opt_mem_gb = _build_and_time(
        lambda: optimized_cls(u, v, img_size, device=device, dtype=dtype,
                               eps=optimized_eps, upsampfac=optimized_upsampfac),
        "optimized",
    )

    fwd_speedup = base_fwd_ms / opt_fwd_ms
    adj_speedup = base_adj_ms / opt_adj_ms
    print(f"  fwd_speedup={fwd_speedup:.2f}x  adj_speedup={adj_speedup:.2f}x")

    return {
        "baseline_fwd_ms": base_fwd_ms, "baseline_fwd_std": base_fwd_std,
        "baseline_adj_ms": base_adj_ms, "baseline_adj_std": base_adj_std,
        "baseline_mem_gb": base_mem_gb,
        "optimized_eps": optimized_eps, "optimized_upsampfac": optimized_upsampfac,
        "optimized_fwd_ms": opt_fwd_ms, "optimized_fwd_std": opt_fwd_std,
        "optimized_adj_ms": opt_adj_ms, "optimized_adj_std": opt_adj_std,
        "optimized_mem_gb": opt_mem_gb,
        "fwd_speedup": fwd_speedup, "adj_speedup": adj_speedup,
    }


def print_baseline_vs_optimized_table(result: dict):
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\hline\hline")
    print(r"Implementation & \texttt{eps} & \texttt{upsampfac} & "
          r"\shortstack{Fwd.\ Time\\(ms)} & \shortstack{Adj.\ Time\\(ms)} & Memory (GB) \\")
    print(r"\hline")
    print(f"Baseline & default & 2.00 & "
          f"${result['baseline_fwd_ms']:.3f} \\pm {result['baseline_fwd_std']:.3f}$ & "
          f"${result['baseline_adj_ms']:.3f} \\pm {result['baseline_adj_std']:.3f}$ & "
          f"{result['baseline_mem_gb']:.3f} \\\\")
    print(f"Optimized & {result['optimized_eps']:.0e} & {result['optimized_upsampfac']:.2f} & "
          f"${result['optimized_fwd_ms']:.3f} \\pm {result['optimized_fwd_std']:.3f}$ "
          f"({result['fwd_speedup']:.2f}$\\times$) & "
          f"${result['optimized_adj_ms']:.3f} \\pm {result['optimized_adj_std']:.3f}$ "
          f"({result['adj_speedup']:.2f}$\\times$) & "
          f"{result['optimized_mem_gb']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent / f"nufft_benchmark_log_{timestamp}.txt"

    with open(log_path, "w") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):

        eu = benchmark_eps_upsampfac(
            uv_source="real",
            real_uv_path="meerkat_4096_220ch_1gpu_uvstacks.pt",
        )
        print("\n--- eps/upsampfac LaTeX table ---\n")
        print_eps_upsampfac_table(eu)

        # gm = benchmark_gpu_method_kerevalmeth(uv_source="real", real_uv_path="meerkat_4096_220ch_1gpu_uvstacks.pt")
        # print("\n--- gpu_method/gpu_kerevalmeth LaTeX table ---\n")
        # print_gpu_method_table(gm)

        # np = benchmark_num_plans(uv_source="real", real_uv_path="meerkat_4096_220ch_1gpu_uvstacks.pt", n_gpus=1)
        # print("\n--- num plans LaTeX table ---\n")
        # print_num_plans_table(np)
        
        from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft_original import MeasOpPytorchFinufft as MeasOpPytorchFinufft_OG
        
        result = compare_baseline_vs_optimized(
            baseline_cls=MeasOpPytorchFinufft_OG,
            optimized_cls=MeasOpPytorchFinufft, 
            optimized_eps=1e-3,       
            optimized_upsampfac=1.25, 
            dtype=torch.float32,
            img_size=(4096, 4096),
            uv_source="real",
            real_uv_path="meerkat_4096_220ch_1gpu_uvstacks.pt",
        )
        print_baseline_vs_optimized_table(result)

    print(f"\n[log saved to {log_path}]")