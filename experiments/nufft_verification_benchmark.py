import time
from typing import List, Tuple
import sys
from pathlib import Path
import torch


file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parents[1])) 
__package__ = "measOperator"

from measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft, SharedNUFFTPlanPair


def _make_synthetic_uv(n_vis: int, device, dtype, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = (torch.rand(n_vis, generator=g) * 2 - 1) * torch.pi
    v = (torch.rand(n_vis, generator=g) * 2 - 1) * torch.pi
    return u.view(1, 1, -1).to(device=device, dtype=dtype), v.view(1, 1, -1).to(device=device, dtype=dtype)


def _make_synthetic_image(img_size: Tuple[int, int], device, dtype, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, 1, *img_size, generator=g).to(device=device, dtype=dtype)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if dtype in (torch.float32, torch.float) else torch.complex128


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def check_correctness(
    configs: List[Tuple[float, float]] = ((1e-2, 1.25), (1e-3, 1.25), (1e-6, 1.25), (1e-6, 2.0)),
    ref_config: Tuple[float, float] = (1e-13, 2.0),
    img_size: Tuple[int, int] = (64, 64),
    n_vis: int = 2000,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> None:
    """
    For each (eps, upsampfac) in `configs`: compares forward/adjoint
    against a high-precision reference (`ref_config`), computed with the
    SAME MeasOpPytorchFinufft code -- isolating the eps/upsampfac
    approximation error specifically. Error is EXPECTED to grow as eps
    loosens; this characterizes the accuracy/speed trade-off rather than
    being a pass/fail check.

    Also checks forward/adjoint consistency (<Ax,y> ~= <x,A^H y>) at
    every config -- independent of eps accuracy, catches a different
    class of bug (sign/conjugate errors, plan misuse). This one SHOULD
    hold tightly regardless of eps; if it doesn't, that's a real bug,
    not an expected accuracy trade-off.
    """
    u, v = _make_synthetic_uv(n_vis, device, dtype, seed=seed)
    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 1)
    cdtype = _complex_dtype(dtype)

    def build(eps, upsampfac):
        return MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac)

    print(f"\n=== Reference: eps={ref_config[0]:.0e}, upsampfac={ref_config[1]} ===")
    ref_op = build(*ref_config)
    y_ref = ref_op._GA(x)
    adj_ref = ref_op._AtGt(y_ref)

    print(f"\n{'eps':>10s} {'upsampfac':>10s} {'fwd rel err':>14s} {'adj rel err':>14s} {'adjoint gap':>14s}")
    for eps, upsampfac in configs:
        op = build(eps, upsampfac)
        y = op._GA(x)
        adj = op._AtGt(y)

        fwd_rel_err = (y - y_ref).norm() / y_ref.norm()
        adj_rel_err = (adj - adj_ref).norm() / adj_ref.norm()

        # forward/adjoint consistency: <A x_test, y_test> should equal
        # <x_test, A^H y_test> for ANY x_test/y_test if A^H truly is the
        # adjoint of A. Using fresh random vectors here, not y/x above,
        # so this checks the adjoint relationship itself rather than
        # anything about the specific forward call already made.
        x_test = torch.randn(1, 1, *img_size, device=device, dtype=cdtype)
        y_test = torch.randn_like(y_ref)
        lhs = torch.sum(op._GA(x_test) * y_test.conj())
        rhs = torch.sum(x_test * op._AtGt(y_test).conj())
        adjoint_gap = (lhs - rhs).abs() / (lhs.abs() + 1e-30)

        print(f"{eps:10.0e} {upsampfac:10.2f} {fwd_rel_err.item():14.3e} "
              f"{adj_rel_err.item():14.3e} {adjoint_gap.item():14.3e}")

# --------------------------------------------------------------------------- #
# Benchmarking
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
    configs: List[Tuple[float, float]] = ((1e-2, 1.25), (1e-3, 1.25), (1e-6, 1.25), (1e-6, 2.0), (1e-9, 2.0)),
    img_size: Tuple[int, int] = (2048, 2048),
    n_vis: int = 1_000_000,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> None:
    """Per-call forward+adjoint time at each (eps, upsampfac), all using
    the individual-plan path (plan built once at construction, timed
    calls are execute()-only) so this isolates the settings' effect on
    execution time, not on plan construction."""
    u, v = _make_synthetic_uv(n_vis, device, dtype, seed=seed)
    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 1)

    print(f"\n=== eps/upsampfac benchmark: img_size={img_size}, n_vis={n_vis}, dtype={dtype} ===")
    print(f"{'eps':>10s} {'upsampfac':>10s} {'mean (ms)':>11s} {'std (ms)':>10s}")
    for eps, upsampfac in configs:
        op = MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac)

        def call():
            y = op._GA(x)
            op._AtGt(y)

        mean_ms, std_ms = _time_calls(call)
        print(f"{eps:10.0e} {upsampfac:10.2f} {mean_ms:11.4f} {std_ms:10.4f}")


def benchmark_shared_vs_individual(
    n_wstacks: int = 8,
    eps: float = 1e-6,
    upsampfac: float = 1.25,
    img_size: Tuple[int, int] = (2048, 2048),
    n_vis_per_stack: int = 1_000_000,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> None:
    """
    Simulates n_wstacks sub-operators on one device, each with its own
    uv-points but the same img_size -- the actual scenario
    SharedNUFFTPlanPair is for. Reports construction time (should scale
    with n_wstacks for individual, stay ~flat for shared -- one grid
    instead of n_wstacks of them) and per-call execution time (expected
    to go the other way: individual pays no per-call setpts() cost since
    each plan's points are fixed at construction, shared pays setpts()
    on every call since the same plan gets repointed for each w-stack).
    """
    uv_per_stack = [_make_synthetic_uv(n_vis_per_stack, device, dtype, seed=seed + i) for i in range(n_wstacks)]
    x = _make_synthetic_image(img_size, device, dtype, seed=seed + 100)

    print(f"\n=== Shared vs individual plan: {n_wstacks} w-stacks, img_size={img_size}, "
          f"n_vis/stack={n_vis_per_stack} ===")

    # --- individual: one plan per w-stack ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ops_individual = [
        MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac)
        for u, v in uv_per_stack
    ]
    torch.cuda.synchronize()
    construct_individual_s = time.perf_counter() - t0

    def call_all_individual():
        for op in ops_individual:
            y = op._GA(x)
            op._AtGt(y)

    mean_individual_ms, std_individual_ms = _time_calls(call_all_individual)

    # --- shared: one plan pair reused across all w-stacks ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    shared_pair = SharedNUFFTPlanPair(img_size, dtype, device, eps=eps, upsampfac=upsampfac)
    ops_shared = [
        MeasOpPytorchFinufft(u, v, img_size, device=device, dtype=dtype, eps=eps, upsampfac=upsampfac,
                              shared_plan_pair=shared_pair)
        for u, v in uv_per_stack
    ]
    torch.cuda.synchronize()
    construct_shared_s = time.perf_counter() - t0

    def call_all_shared():
        for op in ops_shared:
            y = op._GA(x)
            op._AtGt(y)

    mean_shared_ms, std_shared_ms = _time_calls(call_all_shared)

    print(f"\n{'':12s} {'construct (s)':>14s} {'all-stacks call (ms)':>22s} {'std (ms)':>10s}")
    print(f"{'individual':12s} {construct_individual_s:14.4f} {mean_individual_ms:22.4f} {std_individual_ms:10.4f}")
    print(f"{'shared':12s} {construct_shared_s:14.4f} {mean_shared_ms:22.4f} {std_shared_ms:10.4f}")


if __name__ == "__main__":
    check_correctness()
    
    # Visibility count for 220chs: 532_063_731
    benchmark_eps_upsampfac(n_vis=100_000_000, dtype=torch.float32)
    benchmark_eps_upsampfac(n_vis=100_000_000, dtype=torch.float64)
    benchmark_shared_vs_individual(n_vis_per_stack=100_000_000, dtype=torch.float32)
    benchmark_shared_vs_individual(n_vis_per_stack=100_000_000, dtype=torch.float64)