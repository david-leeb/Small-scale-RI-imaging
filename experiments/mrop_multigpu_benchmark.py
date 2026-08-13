"""
Single- vs multi-GPU scaling benchmark for the distributed measurement
operator (Section 4.9's torch.distributed architecture). Times
forward_adjoint_op only -- the full broadcast -> local NUFFT+symmetrise+
RAPHA-forward (all-reduce) -> RAPHA-adjoint (no comm) -> local adjoint
NUFFT (reduce to rank 0) round trip -- since the proximal operator runs
on a single device regardless of world_size and is therefore irrelevant
to a single- vs multi-GPU comparison.

Each invocation of this script (via `torchrun --nproc_per_node=W ...`) is
ONE (world_size, nfreqs) measurement, appended as one line to a shared
JSONL results file. Run it multiple times at increasing W (and, for weak
scaling, proportionally increasing nfreqs) to build up the full sweep,
then call print_scaling_tables() once, standalone, to print both tables.

CAVEAT (RAPHA-enabled runs only): weak scaling here increases nfreqs to
keep channels-per-GPU roughly constant, but solve_epsilon_same_aa derives
M_K from the TOTAL channel count K (at fixed epsilon_n), so increasing
nfreqs also changes RAPHA's own compression shape (M_K grows too) -- a
second effect layered on top of "more GPUs, proportionally more work".
This is not necessarily a confound (if M_K scales roughly with K,
per-GPU compute may stay flat regardless), but it does mean a RAPHA-
enabled weak-scaling result isn't purely isolating the distribution
mechanism. --use_ROP defaults to off (classical model) for the cleanest
read on scaling behaviour; pass it to also characterise RAPHA's own
scaling, bearing this coupling in mind.

USAGE (fill in --data_file and the PLACEHOLDER fields in __main__ first):

    # Strong scaling (fixed nfreqs, increasing GPU count)
    for W in 1 2 4 8; do
        torchrun --nproc_per_node=$W mrop_multigpu_benchmark.py --scaling_mode strong
    done

    # Weak scaling (nfreqs scales with GPU count, e.g. 55 channels/GPU)
    for W in 1 2 4; do
        NFREQS=$((55 * W))
        torchrun --nproc_per_node=$W mrop_multigpu_benchmark.py \\
            --data_file /path/to/dataset --nfreqs $NFREQS --scaling_mode weak
    done

    # After all runs complete (no GPU needed):
    python -c "from mrop_multigpu_benchmark import print_scaling_tables; print_scaling_tables()"

Import paths below are copied directly from imager.py's own relative
imports, converted to absolute -- adjust if this script lives somewhere
else relative to your `src/` package root.
"""

import gc
import json
import time
import datetime
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parents[1]))  # adjust if needed

from src.utils.io_combined import load_dataset
from src.utils.wstacking import compute_global_w_stacking, process_device_global
from src.mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa
from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from src.mrop_ri_measurement_operator import weighting_correction
from src.mrop_ri_measurement_operator.src.mrop_dist import create_meas_op_ROP_dist
from src.utils.gpu_utils_dist import (
    broadcast_object, scatter_channel_data, assign_channels_striped,
    setup_distributed, cleanup_distributed,
)


def build_measurement_operator(param_optimiser: Dict, param_measop: Dict,
                                rank: int, world_size: int, device: torch.device):
    """
    Mirrors imager()'s setup phase up through meas_op construction --
    load data, (if use_ROP) solve RAPHA parameters and apply the
    weighting correction, w-stack, distribute across ranks, and build
    the measurement operator -- then stops there. Deliberately skips
    compress_data, the proximal operator, and the optimiser/imaging loop
    entirely: only the measurement operator's forward/adjoint
    performance is of interest for this benchmark.
    """
    is_root = rank == 0
    if param_measop["nfreqs"] < world_size:
        raise ValueError(
            f"nfreqs={param_measop['nfreqs']} < world_size={world_size} -- every rank needs at "
            "least one channel (see assign_channels_striped); reduce world_size or increase nfreqs."
        )

    use_ROP = param_measop["use_ROP"]

    data = None
    metadata = None
    weight_corr = None
    if is_root:
        data = load_dataset(
            data_path=param_optimiser["data_file"],
            Q=param_measop["Q"],
            super_resolution=param_measop["superresolution"],
            image_pixel_size=param_measop["im_pixel_size"],
            img_size=param_measop["img_size"],
            nfreqs=param_measop["nfreqs"],
            freq_num=param_measop["freq_num"],
            data_weighting=param_measop["flag_data_weighting"],
            weight_type=param_measop["weight_type"],
            weight_robustness=param_measop["weight_robustness"],
            vis_remove=param_measop["vis_remove"],
            dl_shift=param_measop["dl_shift"],
            dm_shift=param_measop["dm_shift"],
            dtype=param_measop["dtype"],
            device=torch.device("cpu"),
        )

        if use_ROP:
            if param_measop["ROP_param"]["Q"] is None:
                assert "Q" in data, "number of antennas Q is not in data and not provided"
                param_measop["ROP_param"]["Q"] = int(data["Q"])

            N = int(np.prod(param_measop["img_size"]))
            K = int(data["nFreqs"])
            V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)

            if param_measop["ROP_param"]["B"] is None:
                if "flag" in data and data["flag"] is not None and "B" not in data:
                    data["B"] = data["flag"].shape[-1] / V
                assert "B" in data, "number of snapshots B is not in data and not provided"
                param_measop["ROP_param"]["B"] = int(data["B"])

            B = int(data["B"] / data["nFreqs"])

            epsilon, P_Q, M_B, M_K = solve_epsilon_same_aa(
                N, param_measop["ROP_param"]["Q"], B, K,
                N_ratio=param_measop["ROP_param"]["N_ratio"],
                n=param_measop["ROP_param"]["epsilon_n"], verbose=True,
            )
            param_measop["ROP_param"]["M_K"] = M_K
            param_measop["ROP_param"]["M_B"] = M_B
            param_measop["ROP_param"]["P"] = P_Q
            param_measop["ROP_param"]["M"] = M_K * M_B
            print(f"INFO: [world_size={world_size}] RAPHA set with P={P_Q}, M_K={M_K}, M_B={M_B}, "
                  f"K={K}, PM/N={P_Q * M_K * M_B / N:.4f}", flush=True)

            data, weight_corr = weighting_correction(data, param_measop["ROP_param"], rapha=True)
            torch.cuda.empty_cache()

        data["y"] = data["y"] * data["nW"] * data["nWimag"]
        data = compute_global_w_stacking(data, param_measop)

        metadata = dict(
            nFreqs=int(data["nFreqs"]),
            Q=int(data["Q"]),
            B=int(data.get("B", 0)),
            B_per_ch=int(data.get("B_per_ch", 0)),
            image_pixel_size=float(data["image_pixel_size"]),
            fov_radians=data["fov_radians"],
            num_wstacks=int(data["num_wstacks"]),
            n_vis_total=int(data["u"].shape[-1]),
            chan_offsets=[int(x) for x in data["chan_offsets"]],
            w_center=data["w_center"].tolist(),
        )

    metadata = broadcast_object(metadata, src=0)
    if use_ROP:
        param_measop["ROP_param"] = broadcast_object(param_measop["ROP_param"] if is_root else None, src=0)
        weight_corr = broadcast_object(weight_corr if is_root else None, src=0)
        weight_corr = weight_corr.to(device=device, dtype=param_measop["dtype"])

    channel_lists = assign_channels_striped(metadata["nFreqs"], world_size)
    w_center = torch.tensor(metadata["w_center"], dtype=param_measop["dtype"], device=device)

    complex_dtype = torch.complex128 if param_measop["dtype"] == torch.float64 else torch.complex64
    dtypes = {
        "u": param_measop["dtype"], "v": param_measop["dtype"], "w": param_measop["dtype"],
        "nW": param_measop["dtype"], "nWimag": param_measop["dtype"],
        "y": complex_dtype, "stack_idx": torch.int32,
    }
    scatter_keys = ["u", "v", "w", "nW", "y", "nWimag", "stack_idx"]
    data = data if is_root else {}
    data = scatter_channel_data(
        data, scatter_keys, channel_lists, metadata["chan_offsets"], device, dtypes, src=0
    )
    torch.cuda.empty_cache()

    if is_root:
        del data["u"], data["v"], data["w"], data["y"]
        gc.collect()

    local_data_for_wstack = {
        "u_dev": [data["u_dev"]], "v_dev": [data["v_dev"]],
        "nW_dev": [data["nW_dev"]], "nWimag_dev": [data["nWimag_dev"]],
        "stack_idx_dev": [data["stack_idx_dev"]],
    }
    w_stack_data = process_device_global(
        0, device, local_data_for_wstack, param_measop,
        metadata["fov_radians"], metadata["num_wstacks"], w_center, rank=rank,
    )
    gc.collect()
    torch.cuda.empty_cache()

    nufft_op = create_meas_op_ROP_dist(MeasOpPytorchFinufft)
    meas_op = nufft_op(
        img_size=param_measop["img_size"],
        w_stack_data=w_stack_data,
        num_chs=metadata["nFreqs"],
        rank=rank,
        world_size=world_size,
        use_ROP=use_ROP,
        ROP_param=param_measop["ROP_param"] if use_ROP else None,
        ant1=data["ant1"] if is_root else None,
        ant2=data["ant2"] if is_root else None,
        batches=data["batches"] if is_root else None,
        device=device,
        dtype=param_measop["dtype"],
        real_flag=True,
        verbose=is_root,
    )
    torch.cuda.empty_cache()

    del data
    gc.collect()
    torch.cuda.empty_cache()

    return meas_op


def _time_distributed_calls(meas_op, x: torch.Tensor, n_iters: int, n_warmup: int, device: torch.device):
    """
    Wall-clock timing (not CUDA events): forward_adjoint_op is a
    multi-process, multi-GPU round trip involving NCCL collectives
    (broadcast, all_reduce, reduce), which CUDA events -- per-device and
    blind to cross-process synchronisation -- cannot correctly capture.
    Uses torch.cuda.synchronize() + dist.barrier() before and after every
    call, measured with perf_counter() (meaningful on rank 0, since every
    rank participates in and is synchronised by the same collectives).
    """
    for _ in range(n_warmup):
        meas_op.forward_adjoint_op(x)
    torch.cuda.synchronize(device)
    dist.barrier()

    times_ms = []
    for _ in range(n_iters):
        torch.cuda.synchronize(device)
        dist.barrier()
        t0 = time.perf_counter()
        meas_op.forward_adjoint_op(x)
        torch.cuda.synchronize(device)
        dist.barrier()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()


def run_benchmark(param_optimiser: Dict, param_measop: Dict, rank: int, world_size: int, device: torch.device,
                   n_iters: int = 50, n_warmup: int = 10, scaling_mode: str = "strong",
                   results_file: str = "mrop_multigpu_results.jsonl") -> None:
    meas_op = build_measurement_operator(param_optimiser, param_measop, rank, world_size, device)

    x = torch.randn(1, 1, *param_measop["img_size"], device=device, dtype=param_measop["dtype"])
    dist.broadcast(x, src=0)  # forward_op re-broadcasts internally regardless; done here too so every
                              # rank's buffer is identical before timing starts, rather than relying on that.

    mean_ms, std_ms = _time_distributed_calls(meas_op, x, n_iters, n_warmup, device)

    if rank == 0:
        free, total = torch.cuda.mem_get_info(device)
        driver_mem_gb = (total - free) / 1024**3

        result = {
            "scaling_mode": scaling_mode,
            "world_size": world_size,
            "nfreqs": param_measop["nfreqs"],
            "use_ROP": param_measop["use_ROP"],
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "driver_mem_gb_rank0": driver_mem_gb,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        print(f"\n=== [{scaling_mode}] world_size={world_size}, nfreqs={param_measop['nfreqs']}, "
              f"use_ROP={param_measop['use_ROP']} ===")
        print(f"  {mean_ms:.4f} ms (std {std_ms:.4f})  driver_mem(rank0)={driver_mem_gb:.3f} GB")

        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")


def print_scaling_tables(results_file: str = "mrop_multigpu_results.jsonl") -> None:
    """
    Reads the accumulated results file (written by run_benchmark across
    potentially many separate torchrun invocations at different
    world_size) and prints strong- and weak-scaling LaTeX tables. No GPU
    needed -- run this after all torchrun launches have completed.

    Strong scaling: speedup relative to the smallest world_size present.
    Weak scaling: raw time only, no speedup column -- ideal weak scaling
    is a FLAT time as world_size (and nfreqs) grow together, not a
    speedup, so read this as "how close to constant" rather than
    "how much faster".
    """
    results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    strong = sorted([r for r in results if r["scaling_mode"] == "strong"], key=lambda r: r["world_size"])
    weak = sorted([r for r in results if r["scaling_mode"] == "weak"], key=lambda r: r["world_size"])

    if strong:
        base_ms = strong[0]["mean_ms"]
        print("\n--- Strong scaling LaTeX table ---\n")
        print(r"\begin{tabular}{rrrr}")
        print(r"\hline\hline")
        print(r"GPUs & \texttt{nfreqs} & Time (ms) & Speedup \\")
        print(r"\hline")
        for r in strong:
            speedup = base_ms / r["mean_ms"]
            print(f"{r['world_size']} & {r['nfreqs']} & "
                  f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ & {speedup:.2f}$\\times$ \\\\")
        print(r"\hline")
        print(r"\end{tabular}")

    if weak:
        print("\n--- Weak scaling LaTeX table ---\n")
        print(r"\begin{tabular}{rrr}")
        print(r"\hline\hline")
        print(r"GPUs & \texttt{nfreqs} & Time (ms) \\")
        print(r"\hline")
        for r in weak:
            print(f"{r['world_size']} & {r['nfreqs']} & "
                  f"${r['mean_ms']:.3f} \\pm {r['std_ms']:.3f}$ \\\\")
        print(r"\hline")
        print(r"\end{tabular}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling_mode", type=str, choices=["strong", "weak"], required=True)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--results_file", type=str, default="mrop_multigpu_results.jsonl")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    param_optimiser_meerkat = {"data_file": "../data/meerkat"}
    param_measop_meerkat = {
        "nfreqs": 100,
        "use_ROP": False,
        "img_size": (4096, 4096),        # PLACEHOLDER -- your real config
        "Q": 59,                        # PLACEHOLDER
        "superresolution": [],           # PLACEHOLDER
        "im_pixel_size": 1.68,            # PLACEHOLDER
        "freq_num": 60,                    # PLACEHOLDER
        "flag_data_weighting": True,      # PLACEHOLDER
        "weight_type": "briggs",          # PLACEHOLDER
        "weight_robustness": 0.0,         # PLACEHOLDER
        "vis_remove": 0,               # PLACEHOLDER
        "dl_shift": 0.0,                  # PLACEHOLDER
        "dm_shift": 0.0,                  # PLACEHOLDER
        "dtype": torch.float32,           # PLACEHOLDER
        "nufft_num_plans": None,         # PLACEHOLDER -- see process_device_global
        "ROP_param": {                    # PLACEHOLDER, only used if --use_ROP
            "Q": None, "B": None, "N_ratio": 1.0, "epsilon_n": 1.0,
            "rv_type": "unitary", "ROP_seed": 0, "ROP_type": "MROP", "weight_type": "briggs",
        },
    }
    
    param_optimiser_3c273 = {"data_file": "../data/273-X08-dmog"}
    param_measop_3c273 = {
        "nfreqs": 64,
        "use_ROP": False,
        "img_size": (1024, 1024),        # PLACEHOLDER -- your real config
        "Q": 27,                        # PLACEHOLDER
        "superresolution": 1.87,           # PLACEHOLDER
        "im_pixel_size": None,            # PLACEHOLDER
        "freq_num": 0,                    # PLACEHOLDER
        "flag_data_weighting": False,      # PLACEHOLDER
        "weight_type": "natural",          # PLACEHOLDER
        "weight_robustness": 0.0,         # PLACEHOLDER
        "vis_remove": 17.7,               # PLACEHOLDER
        "dl_shift": 0.0,                  # PLACEHOLDER
        "dm_shift": 0.0,                  # PLACEHOLDER
        "dtype": torch.float32,           # PLACEHOLDER
        "nufft_num_plans": None,         # PLACEHOLDER -- see process_device_global
        "ROP_param": {                    # PLACEHOLDER, only used if --use_ROP
            "Q": None, "B": None, "N_ratio": 1.0, "epsilon_n": 1.0,
            "rv_type": "unitary", "ROP_seed": 0, "ROP_type": "MROP", "weight_type": "natural",
        },
    }

    run_benchmark(
        param_optimiser_meerkat, param_measop_meerkat, rank, world_size, device,
        n_iters=args.n_iters, n_warmup=args.n_warmup,
        scaling_mode=args.scaling_mode, results_file=args.results_file,
    )

    cleanup_distributed()