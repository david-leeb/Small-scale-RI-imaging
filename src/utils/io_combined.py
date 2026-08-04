import torch
import numpy as np
import gc
import os
import re
import glob
from scipy.io import loadmat
from scipy.constants import speed_of_light
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
    
def _derive_ant_batch_from_flag(flag, Q):
    bpb = Q * (Q - 1) // 2
    triu_r, triu_c = np.triu_indices(Q, k=1)
    dense_idx = np.where(flag)[0]
    batch_local = dense_idx // bpb
    k_local = dense_idx % bpb
    ant1 = triu_r[k_local]
    ant2 = triu_c[k_local]
    return ant1.astype(int), ant2.astype(int), batch_local.astype(int)

def _load_channel(args):
    ch_idx, wavelength, channel_file, data_path, Q = args
    ch_data = loadmat(channel_file)
    ch_flag = ch_data["flag"].astype(bool).squeeze()
    
    uvw = loadmat(os.path.join(data_path, "msSpecs.mat"))["uvw"]
    u_full, v_full, w_full = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    
    if ch_data["data_I"].size == 0:
        print(f"WARNING: Channel {ch_idx + 1} data is empty. Skipping.", flush=True)
        return None
    
    if "ant1_flagged" in ch_data:
        ant1 = ch_data["ant1_flagged"].squeeze().astype(int)
        ant2 = ch_data["ant2_flagged"].squeeze().astype(int)
        batches = ch_data["batches_flagged"].squeeze().astype(int)
    else:
        ant1, ant2, batches = _derive_ant_batch_from_flag(ch_flag, Q)

    return {
        "u": u_full[ch_flag] / wavelength,
        "v": v_full[ch_flag] / wavelength,
        "w": w_full[ch_flag] / wavelength,
        "data": ch_data["data_I"].squeeze(),
        "nW": ch_data["weightsNat"].squeeze(),
        "ant1": ant1, "ant2": ant2, "batches": batches,
        "n_vis": int(ch_flag.sum()),
    }

def _find_channel_files(data_path):
    channel_files = {}

    pattern = os.path.join(data_path, "*_data_ch_*.mat")
    for path in glob.glob(pattern):
        m = re.search(r"_data_ch_(\d+)\.mat$", os.path.basename(path))
        if m:
            channel = int(m.group(1)) - 1
            channel_files[channel] = path

    return channel_files

def load_dataset(
    data_path: str,
    Q: int,
    super_resolution: float = None,
    image_pixel_size: float = None,
    img_size: tuple[int, int] = None,
    data_weighting: bool = True,
    weight_type: str = "briggs",
    weight_robustness: float = 0.0,
    nfreqs: int = None,
    freq_num: int = None,
    vis_remove: float = 0.0,
    dl_shift: float = 0.0,
    dm_shift: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):

    msSpecs = loadmat(os.path.join(data_path, "msSpecs.mat"))
    freqs = msSpecs["freqs"].squeeze()[freq_num : freq_num + nfreqs]
    num_channels = freqs.size
    
    print(
        f"INFO: Frequency range: {freqs.min()/1e6:.4f} MHz to {freqs.max()/1e6:.4f} MHz "
        f"({num_channels} channels).",
        flush=True,
    )
    
    channel_files = _find_channel_files(data_path)
    
    workers = num_workers or min(cpu_count(), num_channels)
    channel_args = [
        (freq_num + i, speed_of_light / freqs[i], channel_files[freq_num + i], data_path, Q)
        for i in range(num_channels)
    ]
    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(_load_channel, channel_args),
            total=num_channels, desc="Loading channels",
        ))
    
    # Filter out None results and concatenate
    channel_results = [r for r in results if r is not None]
    
    # Update channel count after filtering
    num_channels = len(channel_results)
    
    if len(channel_results) == 0:
        raise ValueError("No valid data found in any channel")
    
    u_cat = np.concatenate([r["u"] for r in channel_results])
    v_cat = np.concatenate([r["v"] for r in channel_results])
    w_cat = np.concatenate([r["w"] for r in channel_results])
    nW_cat = np.concatenate([r["nW"] for r in channel_results])
    y_cat = np.concatenate([r["data"] for r in channel_results])
    B_per_ch = int(max(r["batches"].max() for r in channel_results if r["batches"].size) + 1)
    batches_cat = np.concatenate([r["batches"] + B_per_ch * i for i, r in enumerate(channel_results)])
    ant1_cat = np.concatenate([r["ant1"] for r in channel_results])
    ant2_cat = np.concatenate([r["ant2"] for r in channel_results])
    
    chan_offsets = np.concatenate([[0], np.cumsum([r["n_vis"] for r in channel_results])])
    
    print(
        f"INFO: Total number of visibilities: {y_cat.size}, with {num_channels} frequency channels ({freq_num} to {freq_num + num_channels - 1}).",
        flush=True,
    )
    
    del channel_results

    max_proj_baseline = np.max(np.sqrt(u_cat**2 + v_cat**2))
    data_dict = {}
    data_dict["max_proj_baseline"] = max_proj_baseline
    spatial_bandwidth = 2 * max_proj_baseline
    if image_pixel_size is not None:
        print(f"INFO: user specified pixelsize: {image_pixel_size:.4e} arcsec.", flush=True)
    else:
        if "nominal_pixelsize" in data:
            image_pixel_size = data["nominal_pixelsize"].item() / super_resolution
            print(
                f"INFO: user-specified pixel size: {image_pixel_size:.4e} arcsec (i.e. super resolution factor: {super_resolution:.4f})",
                flush=True,
            )
        else:
            image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
            print(
                f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
                flush=True,
            )

    data_dict["image_pixel_size"] = image_pixel_size
    super_resolution = (180.0 / np.pi) * 3600.0 / (image_pixel_size * spatial_bandwidth)
    print(f"INFO: super resolution factor: {super_resolution:.4f}", flush=True)
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    u_cat = u_cat * np.pi / halfSpatialBandwidth
    v_cat = v_cat * np.pi / halfSpatialBandwidth

    c_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    
    data_dict["u"] = torch.tensor(u_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["v"] = -torch.tensor(v_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["w"] = -torch.tensor(w_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["nW"] = torch.tensor(nW_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["y"] = torch.tensor(y_cat, dtype=c_dtype, device=device).view(1, 1, -1)
    data_dict["ant1"] = torch.tensor(ant1_cat, device=device)
    data_dict["ant2"] = torch.tensor(ant2_cat, device=device)
    data_dict["batches"] = torch.tensor(batches_cat, device=device)
    data_dict["Q"] = Q
    data_dict["B"] = B_per_ch * num_channels
    data_dict["B_per_ch"] = B_per_ch
    data_dict["nFreqs"] = num_channels
    data_dict["chan_offsets"] = chan_offsets
    data_dict["image_pixel_size"] = image_pixel_size

    del u_cat, v_cat, w_cat, y_cat, nW_cat
    gc.collect()
    
    if vis_remove > 0:
        data_dict["y"] = data_dict["y"] - vis_remove
        print(f"INFO: Removed constant visibility offset of {vis_remove}", flush=True)

    if dl_shift != 0 or dm_shift != 0:
        dl = dl_shift * image_pixel_size * np.pi
        dm = dm_shift * image_pixel_size * np.pi
        phase = torch.exp(
            1j * 2 * np.pi * (data_dict["u"] * dl + data_dict["v"] * dm)
        ).to(c_dtype)
        data_dict["y"] = data_dict["y"] * phase
        print(f"INFO: Applied phase-center shift dl={dl_shift}, dm={dm_shift}", flush=True)

    if data_weighting and weight_type in ["uniform", "briggs"]:
        from src.ri_measurement_operator.pysrc.utils.gen_imaging_weights import gen_imaging_weights

        # compute imaging weights accordingly to the specified weighting scheme
        print("INFO: computing imaging weights...", flush=True)
        data_dict["nWimag"] = gen_imaging_weights(
            data_dict["u"].clone(),
            data_dict["v"].clone(),
            data_dict["nW"],
            img_size,
            weight_type=weight_type,
            weight_robustness=weight_robustness,
        ).numpy(force=True)
    else:
        print("INFO: imaging weights will not be applied.", flush=True)
        data_dict["nWimag"] = [1.0]
    
    data_dict["nWimag"] = torch.tensor(data_dict["nWimag"], dtype=dtype, device=device).view(1, 1, -1)

    return data_dict