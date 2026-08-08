from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise the default (NCCL) process group. Call once, at process start."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_root(rank: Optional[int] = None) -> bool:
    return (dist.get_rank() if rank is None else rank) == 0


def assign_channels_striped(num_chs: int, world_size: int) -> list[list[int]]:
    """Same striping as the original single-process code -- kept identical so
    channel -> rank ownership matches on every rank without communication."""
    return [list(range(i, num_chs, world_size)) for i in range(world_size)]


def mem(label: str) -> None:
    rank = dist.get_rank()
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / 1024**3
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    free, total = torch.cuda.mem_get_info(dev)
    driver = (total - free) / 1024**3
    print(
        f"[MEM] {label:<45} rank={rank} torch={alloc:.2f} GB "
        f"peak={peak:.2f} GB  driver={driver:.2f} GB",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(dev)


def broadcast_object(obj, src: int = 0):
    box = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def send_tensor(tensor: torch.Tensor, dst: int) -> None:
    dist.send(tensor.contiguous(), dst=dst)
 
 
def recv_tensor(numel: int, dtype: torch.dtype, device: torch.device, src: int) -> torch.Tensor:
    buf = torch.empty(numel, dtype=dtype, device=device)
    dist.recv(buf, src=src)
    return buf


def scatter_channel_data(
    data: dict,
    keys: list[str],
    channel_lists: list[list[int]],
    chan_offsets,
    device: torch.device,
    dtypes: dict,
    src: int = 0,
) -> dict:
    """Rank `src` holds the full CPU tensors in `data[key]` (shape (1,1,N)).
    Every rank ends up with `data[f"{key}_dev"]`, its own channel-based
    slice, on `device` -- via point-to-point send/recv (see module
    docstring for why not dist.scatter).
 
    Keys whose value is a single broadcastable scalar (e.g. nWimag == [1.0]
    when data weighting is disabled -- see load_dataset) are detected
    automatically and broadcast whole, matching the original single-process
    code's `if full.numel() == 1` special case.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
 
    rank_n_vis = [
        sum(int(chan_offsets[c + 1]) - int(chan_offsets[c]) for c in channel_lists[r])
        for r in range(world_size)
    ]
    n_local = rank_n_vis[rank]
 
    is_scalar = {key: data[key].numel() == 1 for key in keys} if rank == src else None
    is_scalar = broadcast_object(is_scalar, src=src)
 
    for key in keys:
        dtype = dtypes[key]
 
        if is_scalar[key]:
            if rank == src:
                val = data[key].to(device=device, dtype=dtype).reshape(1, 1, 1)
            else:
                val = torch.empty((1, 1, 1), dtype=dtype, device=device)
            dist.broadcast(val, src=src)
            data[f"{key}_dev"] = val
            continue
 
        if rank == src:
            full = data[key].to(device=device, dtype=dtype)
            out = None
            for r in range(world_size):
                pieces = [full[:, :, int(chan_offsets[c]):int(chan_offsets[c + 1])] for c in channel_lists[r]]
                chunk = torch.cat(pieces, dim=-1).reshape(-1).contiguous() if pieces else full.new_zeros(0)
                if r == src:
                    out = chunk
                else:
                    send_tensor(chunk, dst=r)
        else:
            out = recv_tensor(n_local, dtype, device, src=src)
 
        data[f"{key}_dev"] = out.view(1, 1, n_local)
 
    data["N_vis_dev"] = n_local
    return data