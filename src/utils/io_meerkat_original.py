import torch
import numpy as np
import gc

def _load_single_channel(args):
    """Helper function to load a single channel in parallel."""
    import os
    from scipy.io import loadmat
    from scipy.constants import speed_of_light
    
    data_path, ch_idx, start_ch, u, v, w, wavelength = args
    
    ch_data = loadmat(os.path.join(data_path, f"_data_ch_{ch_idx+1}.mat"))
    ch_flag = ch_data["flag"].astype(bool).squeeze()
    
    if ch_data["data_I"].size > 0:
        return {
            "u": u[ch_flag] / wavelength,
            "v": v[ch_flag] / wavelength,
            "w": w[ch_flag] / wavelength,
            "data": ch_data["data_I"].squeeze(),
            "nW": ch_data["weightsNat"].squeeze(),
        }
    else:
        return None

def load_real_data_to_tensor(
    data_path: str,
    start_ch: int = 0,
    end_ch: int = -1,
    super_resolution: float = None,
    image_pixel_size: float = None,
    img_size: tuple[int, int] = (4096, 4096),
    data_weighting: bool = True,
    weight_type: str = "briggs",
    weight_robustness: float = 0.0,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):
    """
    Load real data from .mat files into tensors with parallel channel loading.
    
    Args:
        num_workers: Number of parallel workers for loading channels. 
                    If None, uses min(cpu_count(), num_channels).
    """
    import os
    from scipy.io import loadmat
    from tqdm import tqdm
    from scipy.constants import speed_of_light
    from multiprocessing import Pool, cpu_count

    dtype = torch.float64
    c_dtype = torch.complex128

    msSpecs = loadmat(os.path.join(data_path, "msSpecs.mat"))
    u = msSpecs["uvw"][:, 0]
    v = msSpecs["uvw"][:, 1]
    w = msSpecs["uvw"][:, 2]
    
    if end_ch == -1:
        freqs = msSpecs["freqs"].squeeze()[start_ch:]
    else:
        freqs = msSpecs["freqs"].squeeze()[start_ch : end_ch+1]
    
    num_channels = freqs.size
    if num_workers is None:
        num_workers = min(cpu_count(), num_channels)
    
    print(f"INFO: Loading {num_channels} channels using {num_workers} parallel workers...", flush=True)
    
    # Prepare arguments for parallel loading
    channel_args = [
        (data_path, i, start_ch, u, v, w, speed_of_light / freqs.squeeze()[i - start_ch])
        for i in range(start_ch, start_ch + num_channels)
    ]
    
    # Load channels in parallel
    with Pool(num_workers) as pool:
        channel_results = list(tqdm(
            pool.imap(_load_single_channel, channel_args),
            total=num_channels,
            desc="Loading channels"
        ))
    
    # Filter out None results and concatenate
    channel_results = [r for r in channel_results if r is not None]
    
    if len(channel_results) == 0:
        raise ValueError("No valid data found in any channel")
    
    u_cat = np.concatenate([r["u"] for r in channel_results])
    v_cat = np.concatenate([r["v"] for r in channel_results])
    w_cat = np.concatenate([r["w"] for r in channel_results])
    data = np.concatenate([r["data"] for r in channel_results])
    nW = np.concatenate([r["nW"] for r in channel_results])
    
    data_size = data.size
    print(
        f"INFO: Total number of visibilities: {data_size}, with {num_channels} frequency channels ({start_ch} to {start_ch + num_channels - 1}).",
        flush=True,
    )
    
    del u, v, w, channel_results

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
    # image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
    super_resolution = (180.0 / np.pi) * 3600.0 / (image_pixel_size * spatial_bandwidth)
    print(f"INFO: super resolution factor: {super_resolution:.4f}", flush=True)
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    u_cat = u_cat * np.pi / halfSpatialBandwidth
    v_cat = v_cat * np.pi / halfSpatialBandwidth

    data_dict["u"] = torch.tensor(u_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["v"] = -torch.tensor(v_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["w"] = -torch.tensor(w_cat, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["nW"] = torch.tensor(nW, dtype=dtype, device=device).view(1, 1, -1)
    data_dict["y"] = torch.tensor(data, dtype=c_dtype, device=device).view(1, 1, -1)

    del u_cat, v_cat, w_cat, data, nW
    gc.collect()

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
        data_dict["nWimag"] = [
            1.0,
        ]
    data_dict["nWimag"] = torch.tensor(data_dict["nWimag"], dtype=dtype, device=device).view(1, 1, -1)

    return data_dict