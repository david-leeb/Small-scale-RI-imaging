import torch
import numpy as np
import gc
import os
import re
import glob
import json
import shutil
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

_CACHE_VERSION = 1
_CACHE_KEYS = ["u", "v", "w", "nW", "y", "ant1", "ant2", "batches"]

def _source_files(data_path, freq_num, nfreqs):
    """msSpecs.mat and the channel files of the selected range (used for cache validation)."""
    channel_files = _find_channel_files(data_path)
    files = [os.path.join(data_path, "msSpecs.mat")]
    files += [channel_files[c] for c in range(freq_num, freq_num + nfreqs) if c in channel_files]
    return files
 
 
def _fingerprint(files):
    """Staleness check: size and modification time of every source file."""
    return {os.path.basename(f): [os.path.getsize(f), os.path.getmtime(f)] for f in files}


def _load_raw(data_path, Q, nfreqs, freq_num, num_workers=None):
    """
    Reads the .mat files of the selected channels and concatenates them.
    Everything returned here only depends on (data_path, freq_num, nfreqs, Q).
    """
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
    num_channels = len(channel_results)
    if num_channels == 0:
        raise ValueError("No valid data found in any channel")
 
    B_per_ch = int(max(r["batches"].max() for r in channel_results if r["batches"].size) + 1)
    raw = {
        "u": np.concatenate([r["u"] for r in channel_results]),
        "v": np.concatenate([r["v"] for r in channel_results]),
        "w": np.concatenate([r["w"] for r in channel_results]),
        "nW": np.concatenate([r["nW"] for r in channel_results]),
        "y": np.concatenate([r["data"] for r in channel_results]),
        # antenna and batch indices are small integers: store them compactly (lossless)
        "ant1": np.concatenate([r["ant1"] for r in channel_results]).astype(np.int16),
        "ant2": np.concatenate([r["ant2"] for r in channel_results]).astype(np.int16),
        "batches": np.concatenate(
            [r["batches"] + B_per_ch * i for i, r in enumerate(channel_results)]
        ).astype(np.int32),
    }
    meta = {
        "num_channels": num_channels,
        "B_per_ch": B_per_ch,
        "chan_offsets": np.concatenate([[0], np.cumsum([r["n_vis"] for r in channel_results])]).tolist(),
    }
    return raw, meta
 
 
def data_cache_path(data_path, freq_num, nfreqs, Q, cache_dir="data_cache"):
    dataset = os.path.basename(os.path.normpath(data_path))
    return os.path.join(cache_dir, f"{dataset}_ch{freq_num}-{freq_num + nfreqs - 1}_Q{Q}")
 
 
def _load_raw_cached(data_path, Q, nfreqs, freq_num, num_workers=None, cache_dir="data_cache"):
    """_load_raw with an on-disk cache: one .npy file per array plus meta.json."""
    if cache_dir is None:
        return _load_raw(data_path, Q, nfreqs, freq_num, num_workers)
 
    path = data_cache_path(data_path, freq_num, nfreqs, Q, cache_dir)
    fingerprint = _fingerprint(_source_files(data_path, freq_num, nfreqs))
 
    meta_file = os.path.join(path, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        if meta.get("version") == _CACHE_VERSION and meta.get("sources") == fingerprint:
            print(f"INFO: Loading cached data from {path}.", flush=True)
            raw = {k: np.load(os.path.join(path, f"{k}.npy")) for k in _CACHE_KEYS}
            return raw, meta
        print(f"WARNING: data cache {path} is outdated, reloading from .mat files.", flush=True)
 
    raw, meta = _load_raw(data_path, Q, nfreqs, freq_num, num_workers)
    meta.update(version=_CACHE_VERSION, sources=fingerprint)
 
    # write into a temporary directory and rename it: a crashed run never leaves a broken cache
    tmp_path = f"{path}.tmp{os.getpid()}"
    os.makedirs(tmp_path, exist_ok=True)
    for k in _CACHE_KEYS:
        np.save(os.path.join(tmp_path, f"{k}.npy"), raw[k])
    with open(os.path.join(tmp_path, "meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)
    print(f"INFO: Saved data cache to {path}.", flush=True)
    return raw, meta


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
    cache_dir: str = "data_cache",
):

    raw, meta = _load_raw_cached(data_path, Q, nfreqs, freq_num, num_workers, cache_dir)
    u_cat, v_cat, w_cat, nW_cat, y_cat = raw["u"], raw["v"], raw["w"], raw["nW"], raw["y"]
    ant1_cat, ant2_cat, batches_cat = raw["ant1"], raw["ant2"], raw["batches"]
    num_channels = meta["num_channels"]
    B_per_ch = meta["B_per_ch"]
    chan_offsets = np.asarray(meta["chan_offsets"])
    del raw
 
    print(
        f"INFO: Total number of visibilities: {y_cat.size}, with {num_channels} frequency channels ({freq_num} to {freq_num + num_channels - 1}).",
        flush=True,
    )

    max_proj_baseline = np.max(np.sqrt(u_cat**2 + v_cat**2))
    data_dict = {}
    data_dict["max_proj_baseline"] = max_proj_baseline
    spatial_bandwidth = 2 * max_proj_baseline
    if image_pixel_size is not None:
        print(f"INFO: user specified pixelsize: {image_pixel_size:.4e} arcsec.", flush=True)
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
    data_dict["ant1"] = torch.tensor(ant1_cat, dtype=torch.long, device=device)
    data_dict["ant2"] = torch.tensor(ant2_cat, dtype=torch.long, device=device)
    data_dict["batches"] = torch.tensor(batches_cat, dtype=torch.long, device=device)
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