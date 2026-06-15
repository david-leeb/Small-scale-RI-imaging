from scipy.io import loadmat
from scipy.io.matlab import matfile_version
import h5py
import numpy as np
import torch
from scipy.constants import speed_of_light
import os
from .imaging_weight import gen_imaging_weight
# import matplotlib.pyplot as plt


def load_data_to_tensor(
    main_data_file: str,
    data_path: str,
    super_resolution: float = 1.0,
    data_weighting: bool = True,
    load_weight: bool = False,
    img_size: tuple[int, int] = None,
    weight_type: str = "briggs",
    weight_robustness: float = 0.0,
    nfreqs: int = None,
    freq_num: int = None,
    use_ROP: bool = False,
    vis_remove: float = 0.0,
    dl_shift: float = 0.0,
    dm_shift: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
):
    data = {}
    # super_resolution_target = super_resolution
    # if img_size == (1024, 1024):
    #     super_resolution = 1.87
    data_holo = {}
    mat_version, _ = matfile_version(main_data_file)
    if mat_version == 2:
        with h5py.File(main_data_file, "r") as h5File:
            for key, h5obj in h5File.items():
                if isinstance(h5obj, h5py.Dataset):
                    data_holo[key] = np.array(h5obj)
                    if data_holo[key].dtype.names and "imag" in data_holo[key].dtype.names:
                        data_holo[key] = data_holo[key]["real"] + 1j * data_holo[key]["imag"]
                elif isinstance(h5obj, h5py.Group):
                    data_holo[key] = {}
                    for key2, h5obj2 in h5obj.items():
                        data_holo[key][key2] = np.array(h5obj2)
                        if data_holo[key][key2].dtype.names and "imag" in data_holo[key][key2].dtype.names:
                            data_holo[key][key2] = (
                                data_holo[key][key2]["real"] + 1j * data_holo[key][key2]["imag"]
                            )
                else:
                    print("Type not implemented to be read here", h5obj)
    else:
        loadmat(main_data_file, mdict=data_holo)

    if freq_num is not None:
        if nfreqs is None:
            freqs = [data_holo["freqs"].squeeze()[freq_num - 1]]
            print(f"INFO: Using frequency channel {freq_num}: {freqs[0]} Hz", flush=True)
        else:
            freqs = data_holo["freqs"].squeeze()[freq_num - 1 : freq_num - 1 + nfreqs]
            print(f"INFO: Using {nfreqs} frequency channels.", flush=True)
            print(
                f"INFO: Using frequency channels {freq_num} to {freq_num + nfreqs - 1}: {freqs}",
                flush=True,
            )
    else:
        if nfreqs is not None:
            freqs = data_holo["freqs"].squeeze()[:nfreqs]
            print(f"INFO: Using {nfreqs} frequency channels.", flush=True)
        else:
            freqs = data_holo["freqs"].squeeze()
    data["nFreqs"] = len(freqs)

    if use_ROP:
        data["Q"] = 27
        num_data = 0

        for i_f, f in enumerate(freqs):
            data_tmp = loadmat(
                os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
            )
            num_data += data_tmp["data_I"].size
        data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
        data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)

        data["batches"] = np.zeros((num_data), dtype=int)
        data["ant1"] = np.zeros((num_data), dtype=int)
        data["ant2"] = np.zeros((num_data), dtype=int)
        counter = 0
        for i_f, f in enumerate(freqs):
            data_tmp = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
            if i_f == 0:
                data["B_per_ch"] = len(np.unique(data_tmp["batches_flagged"]))
            new_counter = counter + data_tmp["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["y"][0, 0, counter:new_counter] = data_tmp["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_tmp["weightsNat"].squeeze()

            data["batches"][counter:new_counter] = (
                data_tmp["batches_flagged"].squeeze().astype(int) + data["B_per_ch"] * i_f
            )
            data["ant1"][counter:new_counter] = data_tmp["ant1_flagged"].squeeze().astype(int)
            data["ant2"][counter:new_counter] = data_tmp["ant2_flagged"].squeeze().astype(int)

            counter = new_counter

        data["B"] = data["B_per_ch"] * len(freqs)

    else:
        # prepare data for MROP
        num_data = 0
        for i_f, f in enumerate(freqs):
            data_tmp = loadmat(
                os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
            )
            num_data += data_tmp["data_I"].size

        data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
        data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)

        data["batches"] = None
        data["ant1"] = None
        data["ant2"] = None

        counter = 0
        for i_f, f in enumerate(freqs):
            data_ch_i_f = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
            new_counter = counter + data_ch_i_f["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][
                data_ch_i_f["flag"].squeeze() == 1
            ] / (speed_of_light / f)
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][
                data_ch_i_f["flag"].squeeze() == 1
            ] / (speed_of_light / f)
            data["y"][0, 0, counter:new_counter] = data_ch_i_f["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_ch_i_f["weightsNat"].squeeze()
            counter = new_counter
    max_proj_baseline = np.max(np.sqrt(data["u"] ** 2 + data["v"] ** 2))
    data["max_proj_baseline"] = max_proj_baseline
    spatial_bandwidth = 2 * max_proj_baseline
    image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
    print(
        f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
        flush=True,
    )
    data["super_resolution"] = super_resolution

    # Cast to correct datatype given in config file
    dtype_complex = torch.complex64 if dtype == torch.float32 else torch.complex128
    data["u"] = torch.tensor(data["u"], dtype=dtype, device=device).view(1, 1, -1)
    data["v"] = -torch.tensor(data["v"], dtype=dtype, device=device).view(1, 1, -1)
    data["y"] = torch.tensor(data["y"], dtype=dtype_complex, device=device).view(1, 1, -1)
    data["nW"] = torch.tensor(data["nW"], dtype=dtype_complex, device=device).view(1, 1, -1)
    halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

    data["u"] = data["u"] * np.pi / halfSpatialBandwidth
    data["v"] = data["v"] * np.pi / halfSpatialBandwidth

    if vis_remove > 0:
        data["y"] -= vis_remove
        
    if dl_shift != 0 or dm_shift != 0:
        dl = dl_shift * image_pixel_size * np.pi
        dm = dm_shift * image_pixel_size * np.pi
        phase = torch.exp(1j * 2 * np.pi * (data["u"] * dl + data["v"] * dm))
        data["y"] *= phase
        
        
    if data_weighting:
        if load_weight:
            # load imaging weights if available
            data["nWimag"] = data["nWimag"].squeeze()
            if data["nWimag"].size == 0:
                print("INFO: imaging weight is empty and will not be applied.", flush=True)
                data["nWimag"] = [
                    1.0,
                ]
        else:
            # compute imaging weights accordingly to the specified weighting scheme
            print("INFO: computing imaging weights...", flush=True)
            if "weight_robustness" in data:
                weight_robustness = data["weight_robustness"].item()
                print(f"INFO: load weight_robustness from data file {weight_robustness}", flush=True)
            else:
                print(f"INFO: weight_robustness {weight_robustness}", flush=True)
            data["nWimag"] = gen_imaging_weight(
                data["u"].clone(),
                data["v"].clone(),
                data["nW"],
                img_size,
                weight_type=weight_type,
                weight_robustness=weight_robustness,
            ).numpy(force=True)
    else:
        print("INFO: imaging weights will not be applied.", flush=True)
        data["nWimag"] = [
            1.0,
        ]
    data["nWimag"] = torch.tensor(data["nWimag"], dtype=dtype, device=device).view(1, 1, -1)
    
    data["y"] *= data["nW"] * data["nWimag"]
    
    return data