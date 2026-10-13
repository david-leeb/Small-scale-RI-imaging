"""
CPU test of the w-stack partition (gloo backend, no GPU needed).
    PYTHONPATH=. torchrun --nproc_per_node=4 tests/test_vis_exchange.py
"""
import torch
import torch.distributed as dist

from gpu_utils import scatter_wstack_data
from vis_exchange import build_vis_exchange, plan_vis_exchange


def toy_data(num_chs=9, num_wstacks=13):
    g = torch.Generator().manual_seed(0)
    n_per_ch = torch.randint(200, 2000, (num_chs,), generator=g)
    n_per_ch[1] = 0                                     # empty channel, like in the log
    chan_offsets = [0] + torch.cumsum(n_per_ch, 0).tolist()
    p = torch.exp(-0.5 * ((torch.arange(num_wstacks) - num_wstacks / 2) / 3) ** 2)
    stack_idx = torch.multinomial(p, chan_offsets[-1], replacement=True, generator=g)
    return chan_offsets, stack_idx


def check(wstack):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cpu")
    num_chs, num_wstacks = 9, 13
    chan_offsets, stack_idx = toy_data(num_chs, num_wstacks)
    channel_lists = [list(range(i, num_chs, world_size)) for i in range(world_size)]

    ex = build_vis_exchange(stack_idx if rank == 0 else None, chan_offsets, channel_lists,
                            num_wstacks, device, wstack=wstack)
    expected = plan_vis_exchange(stack_idx, chan_offsets, channel_lists, num_wstacks, wstack)

    x = torch.randn(chan_offsets[-1], dtype=torch.complex64, generator=torch.Generator().manual_seed(1))
    y_ch = x[expected["ch_ids"][rank]].view(1, 1, -1)

    # ch -> w gives the w-layout, w -> ch undoes it
    y_w = ex.ch_to_w(y_ch)
    assert torch.equal(y_w.view(-1), x[expected["w_ids"][rank]])
    assert torch.equal(ex.w_to_ch(y_w), y_ch)

    # every local stack is one contiguous slice
    labels = stack_idx[expected["w_ids"][rank]]
    for i, k in enumerate(ex.stack_list):
        assert (labels[ex.stack_slice(i)] == k).all()

    # in w-stack mode every stack lives on exactly one rank
    if wstack:
        owned = sorted(sum(expected["stack_lists"], []))
        assert owned == list(range(num_wstacks))

    # scattered trajectory data matches the exchange
    data = {"u": x.real.view(1, 1, -1).clone()} if rank == 0 else {}
    data = scatter_wstack_data(data, ["u"], ex, device, {"u": torch.float32})
    assert torch.equal(data["u_dev"], y_w.real)

    # adjoint test: <P a, b> == <a, P^H b>, summed over ranks
    b_w = torch.randn(ex.N_vis_w, dtype=torch.complex64, generator=torch.Generator().manual_seed(10 + rank))
    dots = torch.stack([
        torch.view_as_real(torch.vdot(y_w.view(-1), b_w)),
        torch.view_as_real(torch.vdot(y_ch.view(-1), ex.w_to_ch(b_w).view(-1))),
    ]).double()
    dist.all_reduce(dots)
    assert torch.allclose(dots[0], dots[1], rtol=1e-5)

    if rank == 0:
        print(f"OK: wstack={wstack}, world_size={world_size}", flush=True)


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    check(wstack=True)
    check(wstack=False)
    dist.destroy_process_group()