import torch

def assign_channels_striped(num_chs: int, n_dev: int) -> list[list[int]]:
    return [list(range(i, num_chs, n_dev)) for i in range(n_dev)]

def send_to_devices(data, devices):
    """
    Split the fully-loaded, fully-weight-corrected CPU data across devices,
    channel-aligned with how compute_w_stacks and the operator partition work.
    Call this only after load_real_data_to_tensor + weighting_correction have
    both run on the CPU dict.
    """
    n_dev = len(devices)
    num_chs = data["nFreqs"]
    chan_offsets = data["chan_offsets"]
    channel_lists = assign_channels_striped(num_chs, n_dev)
    data["channel_lists"] = channel_lists

    for key in ["u", "v", "w", "nW", "y", "nWimag"]:
        full = data[key]
        dev_list = []
        for i in range(n_dev):
            if full.numel() == 1:
                chunk = full
            else:
                pieces = [
                    full[:, :, int(chan_offsets[c]):int(chan_offsets[c + 1])]
                    for c in channel_lists[i]
                ]
                chunk = torch.cat(pieces, dim=-1)
            dev_list.append(chunk.to(devices[i], non_blocking=True))
        data[f"{key}_dev"] = dev_list

    data["N_vis_dev"] = [t.numel() for t in data["y_dev"]]
    
    for key in ["u","v","w","nW","y"]:
        del data[key]
    
    return data

def _cross_device_copy(op, tensor: torch.Tensor, dst_device: torch.device) -> torch.Tensor:
    
    src_device = tensor.device
    if src_device == dst_device:
        return tensor

    src_idx = op.devices.index(src_device)
    src_stream = op._transfer_stream_dev[src_idx]

    src_stream.wait_stream(torch.cuda.current_stream(src_device))

    with torch.cuda.device(src_device), torch.cuda.stream(src_stream):
        out = tensor.to(dst_device, non_blocking=True)
    tensor.record_stream(src_stream)

    torch.cuda.current_stream(dst_device).wait_stream(src_stream)
    out.record_stream(torch.cuda.current_stream(dst_device))

    return out

def broadcast_to(op, tensor, dst_device):
    return _cross_device_copy(op, tensor, dst_device)

def gather_to_dev0(op, tensor):
    return _cross_device_copy(op, tensor, op.devices[0])

def mem(label, devices):
    for idx, dev in enumerate(devices):
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        free, total = torch.cuda.mem_get_info(dev)
        driver = (total - free) / 1024**3
        print(f"[MEM] {label:<45} dev={idx} torch={alloc:.2f} GB  peak={peak:.2f} GB  driver={driver:.2f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats(dev)