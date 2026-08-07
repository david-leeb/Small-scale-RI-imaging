from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# process group lifecycle
# --------------------------------------------------------------------------- #

def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise the default (NCCL) process group. Call once, at process start."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_root(rank: Optional[int] = None) -> bool:
    return (dist.get_rank() if rank is None else rank) == 0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

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
    """Broadcast a small picklable Python object (dict/int/list of ints, etc.)
    from `src` to every rank. Internally this serialises to a byte tensor and
    calls dist.broadcast, so unlike scatter_object_list it IS NCCL-safe."""
    box = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def broadcast_full_index_tensor(
    tensor: Optional[torch.Tensor],
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    """Broadcast a full-length tensor (e.g. ant1/ant2/batches) from `src` to
    every rank. `numel`/`dtype` must already be known on every rank (send
    them via broadcast_object first) since every rank needs to pre-allocate
    a receive buffer of the right shape before an NCCL broadcast."""
    rank = dist.get_rank()
    if rank == src:
        t = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        t = torch.empty(numel, dtype=dtype, device=device)
    dist.broadcast(t, src=src)
    return t


# --------------------------------------------------------------------------- #
# native dist.gather / dist.scatter for uneven per-rank sizes (pad-to-max)
# --------------------------------------------------------------------------- #

def gather_uneven(
    local: torch.Tensor, sizes: list[int], dst: int = 0
) -> Optional[torch.Tensor]:
    """Collect `local` (flat, 1-D) from every rank onto `dst` via native
    dist.gather, where each rank's true length may differ (`sizes[rank]`).
    `gather_list[r]` is allocated at exactly `sizes[r]` -- no padding.
    Returns the concatenation (rank order) on `dst`, None everywhere else.
    Every rank must call this collectively.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == dst:
        gather_list = [torch.empty(sizes[r], dtype=local.dtype, device=local.device) for r in range(world_size)]
        dist.gather(local, gather_list=gather_list, dst=dst)
        return torch.cat(gather_list)
    else:
        dist.gather(local, dst=dst)
        return None


def scatter_uneven(
    full: Optional[torch.Tensor],
    offsets: list[int],
    local_numel: int,
    dtype: torch.dtype,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    """Inverse of gather_uneven, via native dist.scatter. `full` (flat, 1-D,
    valid only on `src`) is split at `offsets` (length world_size + 1,
    cumulative sizes) and every rank gets its own slice back, sized exactly
    to its own true `local_numel` -- no padding. Every rank must call this
    collectively, including `src`.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    out = torch.empty(local_numel, dtype=dtype, device=device)

    if rank == src:
        scatter_list = [full[offsets[r]:offsets[r + 1]].contiguous() for r in range(world_size)]
        dist.scatter(out, scatter_list=scatter_list, src=src)
    else:
        dist.scatter(out, src=src)

    return out


# --------------------------------------------------------------------------- #
# replacement for send_to_devices(): scatter the per-visibility CPU tensors
# from rank 0 (where load_dataset ran) out to every rank's own GPU
# --------------------------------------------------------------------------- #

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
    Every rank ends up with `data[f"{key}_dev"]`, its own channel-based slice,
    on `device`, via native dist.scatter (see scatter_uneven -- ranks can
    receive different numbers of visibilities, so each rank's chunk is
    padded to the group max before the collective call and trimmed after).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    rank_n_vis = [
        sum(int(chan_offsets[c + 1]) - int(chan_offsets[c]) for c in channel_lists[r])
        for r in range(world_size)
    ]
    n_local = rank_n_vis[rank]
    offsets = [0]
    for n in rank_n_vis:
        offsets.append(offsets[-1] + n)

    for key in keys:
        dtype = dtypes[key]
        if rank == src:
            full = data[key].to(device=device, dtype=dtype)
            # re-order by rank's channel assignment (not necessarily
            # contiguous in the original channel order) before the flat
            # offset-based split that scatter_uneven expects
            pieces_by_rank = [
                torch.cat(
                    [full[:, :, int(chan_offsets[c]):int(chan_offsets[c + 1])] for c in channel_lists[r]],
                    dim=-1,
                ).reshape(-1) if channel_lists[r] else full.new_zeros(0)
                for r in range(world_size)
            ]
            full_reordered = torch.cat(pieces_by_rank)
        else:
            full_reordered = None
        out = scatter_uneven(full_reordered, offsets, n_local, dtype, device, src=src)
        data[f"{key}_dev"] = out.view(1, 1, n_local)

    data["N_vis_dev"] = n_local
    return data