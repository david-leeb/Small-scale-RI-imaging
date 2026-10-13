"""
Compares the w-stack partitioned operator against the current (channel-striped)
operator on a toy dataset, on CPU with the gloo backend:

    PYTHONPATH=. torchrun --nproc_per_node=4 tests/test_operator_wstack.py

Runs on GPU with the real cufinufft operator when CUDA is available (NCCL),
otherwise on CPU with gloo and the dense NUDFT stub in place of
meas_op_nufft_pytorch_finufft.py. Needs the *current* (pre-patch) wstacking.py
and mrop.py saved as tests/old/wstacking_old.py and tests/old/mrop_old.py.

    GPU: PYTHONPATH=. torchrun --nproc_per_node=4 tests/test_operator_wstack.py
"""
import importlib.util
import os
import sys

import torch
import torch.distributed as dist

from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from src.utils.gpu_utils import assign_channels_striped, scatter_channel_data, scatter_wstack_data
from src.utils.vis_exchange import build_vis_exchange
from src.utils.wstacking import process_device_global
from src.mrop_ri_measurement_operator.src.mrop import create_meas_op_ROP
import src.mrop_ri_measurement_operator.src.mrop_old as old_mrop
import src.utils.wstacking_old as old_wstacking


def toy_dataset(num_chs=9, B_per_ch=20, Q=8, num_wstacks=11, seed=0):
    """Same structure as load_dataset + compute_global_w_stacking output."""
    g = torch.Generator().manual_seed(seed)
    triu_r, triu_c = torch.triu_indices(Q, Q, offset=1)
    V = triu_r.numel()
    ant1, ant2, batches, chan_offsets = [], [], [], [0]
    for ch in range(num_chs):
        keep = torch.rand(B_per_ch * V, generator=g) < 0.8          # flagging
        idx = torch.nonzero(keep).view(-1)
        ant1.append(triu_r[idx % V])
        ant2.append(triu_c[idx % V])
        batches.append(idx // V + B_per_ch * ch)
        chan_offsets.append(chan_offsets[-1] + idx.numel())
    n = chan_offsets[-1]
    w = torch.randn(n, generator=g) * 1000
    edges = torch.quantile(w, torch.linspace(0, 1, num_wstacks + 1)[1:-1])
    stack_idx = torch.bucketize(w, edges).to(torch.int32)
    w_center = torch.stack([w[stack_idx == k].mean() for k in range(num_wstacks)])
    data = {
        "u": (torch.rand(n, generator=g) * 2 - 1).view(1, 1, -1) * 3.0,
        "v": (torch.rand(n, generator=g) * 2 - 1).view(1, 1, -1) * 3.0,
        "w": w.view(1, 1, -1),
        "nW": (torch.rand(n, generator=g) + 0.5).view(1, 1, -1),
        "nWimag": (torch.rand(n, generator=g) + 0.5).view(1, 1, -1),
        "y": torch.randn(n, dtype=torch.complex64, generator=g).view(1, 1, -1),
        "stack_idx": stack_idx.view(1, 1, -1),
        "ant1": torch.cat(ant1), "ant2": torch.cat(ant2), "batches": torch.cat(batches),
    }
    rop = dict(ROP_type="MROP", B=B_per_ch * num_chs, Q=Q, P=3, M_K=2, M_B=3,
               rv_type="unitary", ROP_seed=1)
    return data, chan_offsets, w_center, rop


def build_op(mode, device, img_size=(8, 8), fov=(0.05, 0.05), num_chs=9, num_wstacks=11):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0
    data, chan_offsets, w_center, rop = toy_dataset(num_chs=num_chs, num_wstacks=num_wstacks)
    data = data if is_root else {}
    param_measop = dict(img_size=img_size, dtype=torch.float32, nufft_num_plans=None)
    channel_lists = assign_channels_striped(num_chs, world_size)
    dtypes = {"u": torch.float32, "v": torch.float32, "w": torch.float32, "nW": torch.float32,
              "nWimag": torch.float32, "y": torch.complex64, "stack_idx": torch.int32}

    if mode == "old":
        data = scatter_channel_data(data, ["u", "v", "w", "nW", "nWimag", "y", "stack_idx"],
                                    channel_lists, chan_offsets, device, dtypes)
        local = {k: [data[k]] for k in ("u_dev", "v_dev", "nW_dev", "nWimag_dev", "stack_idx_dev")}
        wsd = old_wstacking.process_device_global(0, device, local, param_measop, fov, num_wstacks, w_center, rank=rank)
        Op = old_mrop.create_meas_op_ROP(MeasOpPytorchFinufft)
        op = Op(img_size=img_size, w_stack_data=wsd, num_chs=num_chs, rank=rank, world_size=world_size,
                use_ROP=True, ROP_param=rop, ant1=data.get("ant1"), ant2=data.get("ant2"),
                batches=data.get("batches"), device=device, dtype=torch.float32, verbose=False)
    else:
        ex = build_vis_exchange(data["stack_idx"] if is_root else None, chan_offsets, channel_lists,
                                num_wstacks, device, wstack=(mode == "wstack"))
        data = scatter_wstack_data(data, ["u", "v", "nW", "nWimag"], ex, device, dtypes)
        data = scatter_channel_data(data, ["y"], channel_lists, chan_offsets, device, dtypes)
        wsd = process_device_global(device, data, param_measop, fov, w_center, ex, rank=rank)
        Op = create_meas_op_ROP(MeasOpPytorchFinufft)
        op = Op(img_size=img_size, w_stack_data=wsd, vis_exchange=ex, num_chs=num_chs, rank=rank,
                world_size=world_size, use_ROP=True, ROP_param=rop, ant1=data.get("ant1"),
                ant2=data.get("ant2"), batches=data.get("batches"), device=device,
                dtype=torch.float32, verbose=False)
    return op, data["y_dev"]


def run_all(op, y_ch, device):
    """All operator entry points the optimiser uses."""
    out = {}
    x = torch.rand(1, 1, *op._img_size, generator=torch.Generator().manual_seed(3)).to(device)
    z_rand = torch.randn(op.M_K, op.M_B, op.P_vec, dtype=torch.complex64,
                         generator=torch.Generator().manual_seed(4)).to(device)

    op.use_ROP = True
    out["forward_op"] = op.forward_op(x.clone())
    out["adjoint_op"] = op.adjoint_op(z_rand)
    out["forward_adjoint_op (MROP)"] = op.forward_adjoint_op(x.clone())
    out["compress_data"] = op.compress_data(y_ch)[0]
    out["psf (MROP)"] = op.get_psf()
    torch.manual_seed(5)
    out["op_norm (MROP)"] = torch.tensor(op.get_op_norm(compute_flag=True))

    op.use_ROP = False
    out["forward_adjoint_op (classical)"] = op.forward_adjoint_op(x.clone())
    out["adjoint_classical"] = op.adjoint_classical(y_ch)
    torch.manual_seed(6)
    out["op_norm_prime (classical)"] = torch.tensor(op.get_op_norm_prime())
    op.use_ROP = True

    # adjoint test <A x, z> = <x, A^H z> (x real, so take the real part)
    lhs = torch.vdot(out["forward_op"].reshape(-1), z_rand.reshape(-1)).real
    rhs = torch.dot(x.reshape(-1), out["adjoint_op"].reshape(-1)) if out["adjoint_op"] is not None else None
    out["_adjoint_test"] = (lhs, rhs)
    return out


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # same as setup_distributed in gpu_utils
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # cufinufft needs a fine grid larger than its kernel -> bigger image on GPU.
    # GPU spreading uses atomics and the point order inside a stack changes,
    # so results agree to fp32 rounding only -> looser tolerance.
    img_size = (64, 64) if use_gpu else (8, 8)
    tol_op, tol_adj = (1e-4, 1e-4) if use_gpu else (1e-5, 1e-5)

    results = {}
    for mode in ("old", "channel", "wstack"):
        op, y_ch = build_op(mode, device, img_size=img_size)
        results[mode] = run_all(op, y_ch, device)
        del op
        if use_gpu:
            torch.cuda.empty_cache()

    failed = False
    if rank == 0:
        print(f"\nworld_size={world_size}: relative max error vs current operator", flush=True)
        for key, ref in results["old"].items():
            if key.startswith("_"):
                continue
            for mode in ("channel", "wstack"):
                err = rel_err(results[mode][key], ref)
                # op_norm_prime: the current code multiplies ch-ordered visibilities with
                # stack-ordered weights, so it is expected to differ (see notes)
                expected_diff = key.startswith("op_norm_prime")
                tol = 1e-3 if key.startswith("op_norm") else tol_op
                ok = err < tol
                if not ok and not expected_diff:
                    failed = True
                print(f"  {key:<32} {mode:<8} {err:.2e}  {'OK' if ok else ('DIFFERS (expected)' if expected_diff else 'FAIL')}", flush=True)
        err = rel_err(results["wstack"]["op_norm_prime (classical)"], results["channel"]["op_norm_prime (classical)"])
        print(f"  {'op_norm_prime wstack vs channel':<32} {'':<8} {err:.2e}  {'OK' if err < 1e-3 else 'FAIL'}", flush=True)
        failed |= err >= 1e-3
        lhs, rhs = results["wstack"]["_adjoint_test"]
        err = abs(lhs - rhs).item() / abs(lhs).item()
        print(f"  {'adjoint test (wstack)':<32} {'':<8} {err:.2e}  {'OK' if err < tol_adj else 'FAIL'}", flush=True)
        failed |= err >= tol_adj

    flag = torch.tensor([int(failed)], device=device)   # NCCL: must be on GPU
    dist.broadcast(flag, src=0)
    dist.destroy_process_group()
    sys.exit(int(flag.item()))