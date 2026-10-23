import torch
import torch.distributed as dist

from src.utils.gpu_utils import broadcast_object, send_tensor, recv_tensor


def assign_wstacks(n_vis_per_stack, world_size, stack_cost=None):
    """
    Greedy assignment of whole w-stacks to ranks: biggest stack first, always to
    the least loaded rank. The cost of a stack is n_vis + stack_cost, where
    stack_cost is the fixed per-stack cost (FFT + image-domain ops) expressed in
    visibilities. Default: the mean stack size, i.e. fixed and gridding cost of
    an average stack are equal (roughly what the strong scaling fit shows).
    """
    if stack_cost is None:
        stack_cost = n_vis_per_stack.sum().item() / n_vis_per_stack.numel()
    cost = n_vis_per_stack.double() + stack_cost

    load = [0.0] * world_size
    stack_lists = [[] for _ in range(world_size)]
    for k in torch.argsort(cost, descending=True, stable=True).tolist():
        r = min(range(world_size), key=lambda i: load[i])
        stack_lists[r].append(k)
        load[r] += cost[k].item()
    return [sorted(s) for s in stack_lists]


def plan_vis_exchange(stack_idx, chan_offsets, channel_lists, num_wstacks, wstack=True, stack_cost=None, device=torch.device("cpu")):
    """
    Builds the mapping between the two layouts:

    ch-layout: rank r holds the channels channel_lists[r] (same order as
               scatter_channel_data). Used by y and MROP.
    w-layout:  rank r holds whole w-stacks, each stack contiguous. Used by the NUFFT.
               With wstack=False the w-layout is the rank's own ch-layout sorted by
               stack (current behaviour, no communication).

    Returns a dict with per-rank lists:
        ch_ids / w_ids: global visibility index of each local position
        send_idx:       ch positions in all-to-all send order (grouped by destination)
        recv_idx:       w position of each slot in the all-to-all receive buffer
        counts:         (G, G) number of visibilities going from rank src to rank dst
        stack_lists / stack_offsets: local stacks and their slices in the w-layout
    """
    world_size = len(channel_lists)
    stack_idx = stack_idx.view(-1).to(device=device, dtype=torch.long)
    n_vis = stack_idx.numel()

    # ch-layout
    ch_ids = [
        torch.cat([torch.arange(chan_offsets[c], chan_offsets[c + 1], device=device) for c in cl])
        if cl else torch.empty(0, dtype=torch.long, device=device)
        for cl in channel_lists
    ]
    ch_owner = torch.empty(n_vis, dtype=torch.long, device=device)
    for r in range(world_size):
        ch_owner[ch_ids[r]] = r

    # w-layout
    if wstack:
        stack_lists = assign_wstacks(torch.bincount(stack_idx, minlength=num_wstacks).cpu(), world_size, stack_cost)
        stack_owner = torch.empty(num_wstacks, dtype=torch.long, device=device)
        for r, ks in enumerate(stack_lists):
            stack_owner[ks] = r
        w_owner = stack_owner[stack_idx]
        by_stack = torch.argsort(stack_idx, stable=True)
        w_ids = [by_stack[w_owner[by_stack] == r] for r in range(world_size)]
    else:
        stack_lists = [list(range(num_wstacks))] * world_size
        w_owner = ch_owner
        w_ids = [ids[torch.argsort(stack_idx[ids], stable=True)] for ids in ch_ids]

    w_pos = torch.empty(n_vis, dtype=torch.long, device=device)
    for r in range(world_size):
        w_pos[w_ids[r]] = torch.arange(w_ids[r].numel(), device=device)


    # all-to-all ordering: each chunk src -> dst is sorted by the w position on dst
    send_idx, recv_idx, stack_offsets = [], [], []
    counts = torch.zeros(world_size, world_size, dtype=torch.long)
    for r in range(world_size):
        dst = w_owner[ch_ids[r]]
        send_idx.append(torch.argsort(dst * n_vis + w_pos[ch_ids[r]]))
        counts[r] = torch.bincount(dst, minlength=world_size).cpu()

        recv_idx.append(torch.argsort(ch_owner[w_ids[r]], stable=True))

        n_local = torch.bincount(stack_idx[w_ids[r]], minlength=num_wstacks).cpu()[stack_lists[r]]
        stack_offsets.append([0] + torch.cumsum(n_local, 0).tolist())

    return dict(
        ch_ids=ch_ids, w_ids=[ids.cpu() for ids in w_ids], send_idx=send_idx, recv_idx=recv_idx,
        counts=counts, stack_lists=stack_lists, stack_offsets=stack_offsets,
    )


class VisExchange:
    """
    Moves visibility vectors between ch-layout (MROP) and w-layout (NUFFT).
        ch_to_w: send = y_ch[send_idx] -> all-to-all -> y_w[recv_idx] = recv
        w_to_ch: send = y_w[recv_idx]  -> all-to-all -> y_ch[send_idx] = recv
    Both are permutations, so w_to_ch is the adjoint (and inverse) of ch_to_w.
    """

    def __init__(self, send_idx, recv_idx, counts, stack_list, stack_offsets, rank, w_ids=None):
        self.send_idx = send_idx
        self.recv_idx = recv_idx
        self.send_counts = counts[rank].tolist()
        self.recv_counts = counts[:, rank].tolist()
        self.N_vis_ch = sum(self.send_counts)
        self.N_vis_w = sum(self.recv_counts)
        self.is_local = bool(counts.sum() == counts.diag().sum())   # same on every rank
        self.stack_list = stack_list
        self.stack_offsets = stack_offsets
        self.w_ids = w_ids   # root only: needed to scatter u, v, nW, nWimag

        self.send_inv = torch.empty_like(send_idx)
        self.send_inv[send_idx] = torch.arange(send_idx.numel(), dtype=send_idx.dtype, device=send_idx.device)

    def stack_slice(self, i):
        return slice(self.stack_offsets[i], self.stack_offsets[i + 1])

    def _all_to_all(self, send, n_out, out_counts, in_counts):
        if self.is_local:
            return send
        recv = torch.empty(n_out, dtype=send.dtype, device=send.device)
        dist.all_to_all_single(
            torch.view_as_real(recv) if recv.is_complex() else recv,
            torch.view_as_real(send) if send.is_complex() else send,
            output_split_sizes=out_counts, input_split_sizes=in_counts,
        )
        return recv

    def ch_to_w(self, y_ch):
        send = y_ch.view(-1)[self.send_idx]
        recv = self._all_to_all(send, self.N_vis_w, self.recv_counts, self.send_counts)
        y_w = torch.empty_like(recv)
        y_w[self.recv_idx] = recv
        return y_w.view(1, 1, -1)

    def w_to_ch(self, y_w):
        send = y_w.view(-1)[self.recv_idx]
        recv = self._all_to_all(send, self.N_vis_ch, self.send_counts, self.recv_counts)
        return recv[self.send_inv].view(1, 1, -1)

def build_vis_exchange(stack_idx, chan_offsets, channel_lists, num_wstacks, device,
                       wstack=True, stack_cost=None, src=0):
    """
    Called on every rank. Root (holding the full CPU stack_idx) computes the plan
    and sends each rank its indices, like _init_rop does for the MROP indices.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    plan = None
    meta = None
    if rank == src:
        plan = plan_vis_exchange(stack_idx, chan_offsets, channel_lists, num_wstacks, wstack, stack_cost, device)
        meta = dict(counts=plan["counts"], stack_lists=plan["stack_lists"], stack_offsets=plan["stack_offsets"])
    meta = broadcast_object(meta, src=src)
    counts = meta["counts"]

    if rank == src:
        for r in range(world_size):
            if r == src:
                continue
            if counts[r].sum() > 0:
                send_tensor(plan["send_idx"][r].to(device), dst=r)
            if counts[:, r].sum() > 0:
                send_tensor(plan["recv_idx"][r].to(device), dst=r)
        send_idx = plan["send_idx"][src].to(device)
        recv_idx = plan["recv_idx"][src].to(device)
    else:
        n_ch, n_w = int(counts[rank].sum()), int(counts[:, rank].sum())
        send_idx = recv_tensor(n_ch, torch.long, device, src=src) if n_ch else torch.empty(0, dtype=torch.long, device=device)
        recv_idx = recv_tensor(n_w, torch.long, device, src=src) if n_w else torch.empty(0, dtype=torch.long, device=device)

    if rank == src:
        n_w_all = counts.sum(0).tolist()
        total = sum(n_w_all)
        moved = (counts.sum() - counts.diag().sum()).item()
        print(f"INFO: {'w-stack' if wstack else 'channel'} partition, per-rank NUFFT workload:", flush=True)
        for r in range(world_size):
            print(f"  rank={r}: stacks={len(meta['stack_lists'][r]):>3} N_vis={n_w_all[r]:>10} "
                  f"({100 * n_w_all[r] / total:5.1f}%)", flush=True)
        print(f"INFO: all-to-all moves {moved} visibilities ({100 * moved / total:.1f}%) per direction", flush=True)

    return VisExchange(
        send_idx, recv_idx, counts,
        meta["stack_lists"][rank], meta["stack_offsets"][rank], rank,
        w_ids=plan["w_ids"] if rank == src else None,
    )