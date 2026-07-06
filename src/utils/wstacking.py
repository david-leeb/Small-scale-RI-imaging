from typing import Tuple, Union
import torch
import gc, os
import numpy as np
import datetime
import math
from sklearn.cluster import KMeans

from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft

def get_n_term(
    img_size: Tuple[int, int],
    fov_radians: Union[Tuple[float, float], float],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Builds the n coordinate of the celestial sky.

    Args:
        img_size (Tuple[int, int]): The size of the image.
        fov_radians (Union[Tuple[int, int], int]): The field of view in radians.

    Returns:
        int: The number of terms in the Fourier domain.
    """
    if isinstance(fov_radians, float):
        fov_radians = (fov_radians, fov_radians)
    l_grid, m_grid = torch.meshgrid(
        torch.arange(-img_size[1] * 0.5, img_size[1] * 0.5, device=device, dtype=dtype),
        torch.arange(-img_size[0] * 0.5, img_size[0] * 0.5, device=device, dtype=dtype),
        indexing="xy",
    )
    dl_grid = 2 * math.sin(fov_radians[0] * 0.5) / img_size[1]
    dm_grid = 2 * math.sin(fov_radians[1] * 0.5) / img_size[0]
    l_grid = l_grid * dl_grid
    m_grid = m_grid * dm_grid
    return torch.sqrt(1 - l_grid**2 - m_grid**2).reshape(1, 1, *img_size)

def compute_w_stacks(data, param_measop, devices):
    
    # Set up multi GPU parameters
    n_dev = len(devices)
    num_chs = data["nFreqs"]
    assert num_chs % n_dev == 0, (
        f"num_chs ({num_chs}) must be divisible by n_dev ({n_dev}) for channel-aligned sharding"
    )
    chs_per_dev = num_chs // n_dev
    chan_offsets = data["chan_offsets"]
    
    # Compute field of view
    fov_radians = (
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )
    
    w_stack_data_list = []
    for d, dev in enumerate(devices):
        
        # Slice data according to channels per device
        ch_lo = d * chs_per_dev
        ch_hi = (d + 1) * chs_per_dev
        b_lo = int(chan_offsets[ch_lo])
        b_hi = int(chan_offsets[ch_hi])

        data_i = {
            "u": data["u"][:, :, b_lo:b_hi].to(dev),
            "v": data["v"][:, :, b_lo:b_hi].to(dev),
            "w": data["w"][:, :, b_lo:b_hi].to(dev),
        }
        print(f"INFO: Running w-stacking for channel subset {d} "
              f"(channels [{ch_lo}:{ch_hi}), baselines [{b_lo}:{b_hi})) on {dev}", flush=True)
        
        w = data_i["w"]
        w_np = w.numpy(force=True).reshape(-1, 1)
        
        num_wstacks = int(np.ceil(
            w_np.max() * 2 * np.pi * (1 - np.sqrt(1 - 2 * np.sin(fov_radians[0] / 2) ** 2)))
        )
        print(f"INFO: FOV in radians: {fov_radians}, max w value: {w_np.max():.4f}, number of w-stacks determined to be {num_wstacks} based on the FOV and max w value.", flush=True)
        
        # search for centres of w planes
        # parameters for k-means clustering, hard-coded for now
        kmeans_frac_pts = 0.01
        kmeans_max_pts = int(1e6)
        kmeans_max_iter = 1000

        # run k-means on a subset of w values
        kmeans_num_pts = min(int(kmeans_frac_pts * w_np.size), kmeans_max_pts)

        np.random.seed(42)

        idx = np.random.choice(w_np.size, kmeans_num_pts, replace=False)

        w_sampled = w_np[idx]

        w_kmeans = w_sampled.reshape(-1, 1)

        kmeans = KMeans(
            n_clusters=num_wstacks,
            random_state=0,
            max_iter=kmeans_max_iter,
            n_init=30,
            tol=1e-6,
        )
        kmeans.fit(w_kmeans)
        centers = np.sort(kmeans.cluster_centers_, axis=0)

        labels = kmeans.predict(w_np)
        
        # move results to device
        w_center = torch.as_tensor(centers, dtype=param_measop["dtype"], device=dev).view(-1)
        w_stack_idx = torch.as_tensor(labels, dtype=torch.int32, device=dev).view(-1)

        del w_kmeans, kmeans, labels, centers
        gc.collect()

        print("INFO: w-stacking centers: ", end="")
        w_center_np = w_center.numpy(force=True).ravel()
        for i in range(num_wstacks):
            print(f"{w_center_np[i]:.7f}, ", end="")
        print("", flush=True)

        # create w-stacking correction term
        n_term = get_n_term(param_measop["img_size"], fov_radians, dev, param_measop["dtype"])

        # Precompute per-stack w-corrections
        w_stack_correct = [
            torch.exp(-2j * torch.pi * w_center[i] * (n_term - 1)) / n_term
            for i in range(num_wstacks)
        ]
        
        meas_op = [None] * num_wstacks
        for i in range(num_wstacks):
            meas_op[i] = MeasOpPytorchFinufft(
                u=data_i["u"][:, :, w_stack_idx == i],
                v=data_i["v"][:, :, w_stack_idx == i],
                img_size=param_measop["img_size"],
                real_flag=True,
                dtype=param_measop["dtype"],
                device=dev,
            )
    
        w_stack_data = {
            "w_center": w_center,
            "corrections": w_stack_correct,
            "stack_idx": w_stack_idx,
            "meas_op": meas_op,
        }
        w_stack_data_list.append(w_stack_data)
    
    return w_stack_data_list