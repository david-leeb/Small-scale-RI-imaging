import torch
import sys

sys.path.insert(0, "/mnt/pvc/diss/Small-scale-RI-imaging-mrop")

from src.prox_operator.prox_op_sara_original import ProxOpSARAPos_original as ProxOpSARAPos_original
from src.prox_operator.prox_op_sara import ProxOpSARAPos as ProxOpSARAPos
from src.prox_operator.prox_op_sara_optimized_ptwt import ProxOpSARAPos as ProxOpSARAPos_optimized

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

REL_ERROR_TOL = 1e-5


def run_verification(img_size=(2048, 2048), dtype=torch.float32, n_iters=5, verbose_each_iter=False):
    """
    Runs `n_iters` chained calls (mirroring the reweighting loop's repeated
    calls with updated weights) and reports one summary line per
    implementation -- the worst-case max-absolute and relative error across
    all iterations -- rather than printing the full detail every iteration.
    Set verbose_each_iter=True to also print a compact per-iteration line.
    """
    device = torch.device("cuda")
    obj_tol = 0

    prox_op_og = ProxOpSARAPos_original(
        img_size, device=device, dtype=dtype, verbose=False, max_iter=20, obj_tol=obj_tol
    )
    prox_op_opt = ProxOpSARAPos(
        img_size, device=device, dtype=dtype, verbose=False, max_iter=20, obj_tol=obj_tol
    )
    prox_op_opt_new = ProxOpSARAPos_optimized(
        img_size, device=device, dtype=dtype, verbose=False, max_iter=20, obj_tol=obj_tol
    )

    print(f"\n--- Verification: img_size={img_size}, dtype={dtype} ---")
    torch.manual_seed(1337)

    img_og = torch.randn(img_size, dtype=dtype, device=device).unsqueeze(0)
    img_opt = img_og.clone()
    img_opt_new = img_og.clone()

    prox_op_og.update(img_og, initialisation=True)
    prox_op_opt.update(img_opt, initialisation=True)
    prox_op_opt_new.update(img_opt_new, initialisation=True)

    worst = {
        "OPT CUSTOM": {"max_abs": 0.0, "rel": 0.0, "iter": -1},
        "OPT PTWT": {"max_abs": 0.0, "rel": 0.0, "iter": -1},
    }
    all_passed = True

    for i in range(n_iters):
        out_og = prox_op_og(img_og)
        out_opt = prox_op_opt(img_opt)
        out_opt_new = prox_op_opt_new(img_opt_new)

        prox_op_og.update(out_og)
        prox_op_opt.update(out_opt)
        prox_op_opt_new.update(out_opt_new)

        diff_max_opt = torch.max(torch.abs(out_og - out_opt)).item()
        diff_rel_opt = (torch.norm(out_og - out_opt) / torch.norm(out_og)).item()

        diff_max_new = torch.max(torch.abs(out_og - out_opt_new)).item()
        diff_rel_new = (torch.norm(out_og - out_opt_new) / torch.norm(out_og)).item()

        if diff_rel_opt > worst["OPT CUSTOM"]["rel"]:
            worst["OPT CUSTOM"] = {"max_abs": diff_max_opt, "rel": diff_rel_opt, "iter": i}
        if diff_rel_new > worst["OPT PTWT"]["rel"]:
            worst["OPT PTWT"] = {"max_abs": diff_max_new, "rel": diff_rel_new, "iter": i}

        if diff_rel_opt >= REL_ERROR_TOL or diff_rel_new >= REL_ERROR_TOL:
            all_passed = False

        if verbose_each_iter:
            print(f"  iter {i}: OPT CUSTOM rel={diff_rel_opt:.3e}  OPT PTWT rel={diff_rel_new:.3e}")

    for name, w in worst.items():
        status = "\u2705" if w["rel"] < REL_ERROR_TOL else "\u274c"
        print(f"{status} {name:11s}: worst max_abs={w['max_abs']:.3e}  worst rel={w['rel']:.3e}  (iter {w['iter']})")

    if all_passed:
        print("\u2705 All implementations match within tolerance across all iterations.")
    else:
        print("\u274c At least one implementation exceeded tolerance -- see per-implementation summary above.")

    return all_passed


def _time_per_sub_iteration(prox_op, x: torch.Tensor, n_repeats: int = 20, n_warmup: int = 3):
    """
    Mean/std time (ms) for ONE sub-iteration of prox_op.

    Times prox_op(x) end-to-end per call and divides by prox_op._max_iter
    -- exact, not an estimate, because obj_tol=0 means the early-stop
    check (obj_rel_var < obj_tol) can never fire, so every call runs
    exactly max_iter sub-iterations.

    Warm-up calls happen first and are not timed -- for the CUDA-graph
    implementations this is where the one-time capture happens (3
    internal warm-up iterations + the capture itself), which would
    otherwise dominate a small-sample mean/std. update() is never called
    between timed repeats: it invalidates the graph (forces recapture on
    the next call), and __call__ without an intervening update() is the
    steady-state cost that matches how often each is actually called in
    practice (update() once per reweighting cycle, __call__ every inner
    iteration).
    """
    for _ in range(n_warmup):
        prox_op(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(n_repeats):
        start.record()
        prox_op(x)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end) / prox_op._max_iter)

    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()


def run_benchmark(
    img_sizes=((256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)),
    dtypes=(torch.float32, torch.float64),
    n_repeats: int = 20,
    n_warmup: int = 5,
):
    """
    Benchmarks all three prox implementations across every (img_size, dtype)
    combination. Returns a flat list of result dicts, one per combination
    and implementation, ready for `print_latex_table`.

    Wrapped in a per-implementation try/except for CUDA OOM: at the larger
    end of img_sizes x float64, some implementation/size combinations may
    not fit in memory. A skipped implementation is reported and the sweep
    continues; if the baseline itself OOMs for a given (img_size, dtype),
    that whole combination is skipped (no baseline to compute speedup
    against).
    """
    device = torch.device("cuda")
    obj_tol = 0  # every __call__ runs exactly max_iter sub-iterations

    implementations = {
        "ORIGINAL": ProxOpSARAPos_original,
        "OPT CUSTOM": ProxOpSARAPos,
        "OPT PTWT": ProxOpSARAPos_optimized,
    }
    base_name = "ORIGINAL"

    all_results = []

    for img_size in img_sizes:
        for dtype in dtypes:
            print(f"\n--- Benchmark: img_size={img_size}, dtype={dtype} ---")
            torch.manual_seed(1337)
            x = torch.randn(img_size, dtype=dtype, device=device).unsqueeze(0)

            timings = {}
            for name, cls in implementations.items():
                try:
                    prox_op = cls(
                        img_size, device=device, dtype=dtype, verbose=False, max_iter=20, obj_tol=obj_tol
                    )
                    prox_op.update(x, initialisation=True)
                    mean_ms, std_ms = _time_per_sub_iteration(
                        prox_op, x, n_repeats=n_repeats, n_warmup=n_warmup
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{name:11s}: OOM at img_size={img_size}, dtype={dtype} -- skipping.")
                    torch.cuda.empty_cache()
                    continue

                timings[name] = (mean_ms, std_ms)
                print(f"{name:11s}: {mean_ms:8.4f} ms/iter  (std {std_ms:.4f} ms, n={n_repeats})")

                del prox_op
                torch.cuda.empty_cache()

            if base_name not in timings:
                print(f"  (baseline OOM'd for img_size={img_size}, dtype={dtype} -- skipping this combination)")
                continue

            base_mean = timings[base_name][0]
            for name, (mean_ms, std_ms) in timings.items():
                all_results.append(
                    {
                        "img_size": img_size,
                        "dtype": dtype,
                        "impl": name,
                        "mean_ms": mean_ms,
                        "std_ms": std_ms,
                        "speedup": base_mean / mean_ms,
                    }
                )

    return all_results


def _dtype_label(dtype: torch.dtype) -> str:
    return {"torch.float32": "single", "torch.float64": "double"}.get(str(dtype), str(dtype))


def print_latex_table(all_results, base_name: str = "ORIGINAL"):
    """
    Prints `all_results` (from run_benchmark) as a LaTeX `tabular` body,
    one row per (image size, precision), with the baseline's time and one
    (time, speedup) column pair per remaining implementation -- ready to
    paste directly into a `table` environment.

    Rows/combinations skipped in run_benchmark (e.g. due to OOM) are
    simply absent from all_results and therefore absent from the table.
    """
    grouped = {}
    for r in all_results:
        key = (r["img_size"], r["dtype"])
        grouped.setdefault(key, {})[r["impl"]] = r

    impl_names = list(dict.fromkeys(r["impl"] for r in all_results))
    non_base = [n for n in impl_names if n != base_name]

    col_spec = "ll" + "r" * (1 + 2 * len(non_base))
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\hline\hline")
    header = ["Image Size", "Precision", f"{base_name} (ms)"]
    for name in non_base:
        header += [f"{name} (ms)", f"{name} Speedup"]
    print(" & ".join(header) + r" \\")
    print(r"\hline")

    for (img_size, dtype), impls in grouped.items():
        if base_name not in impls:
            continue  # baseline missing for this combination -- no speedup reference
        size_str = f"${img_size[0]}\\times{img_size[1]}$"
        prec_str = _dtype_label(dtype)
        row = [size_str, prec_str, f"{impls[base_name]['mean_ms']:.3f}"]
        for name in non_base:
            if name in impls:
                row.append(f"{impls[name]['mean_ms']:.3f}")
                row.append(f"{impls[name]['speedup']:.2f}$\\times$")
            else:
                row.append("--")
                row.append("--")
        print(" & ".join(row) + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    run_verification()
    results = run_benchmark()
    print("\n--- LaTeX table ---\n")
    print_latex_table(results)