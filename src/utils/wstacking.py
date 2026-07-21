from asyncio import futures
from typing import Tuple, Union
import torch
import gc, os
import numpy as np
import datetime
import math
from sklearn.cluster import KMeans

from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from src.utils.gpu_utils import assign_channels_striped

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

def process_device(d, dev, data, param_measop, fov_radians, channel_lists, n_dev):
    
    chs = channel_lists[d]
    print(f"INFO: Device {d} ({dev}) will process channels {chs}", flush=True)

    data_i = {
        "u": data["u_dev"][d],
        "v": data["v_dev"][d],
        "w": data["w_dev"][d],
    }
    
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
    labels = kmeans.predict(w_np)
    centers = np.sort(kmeans.cluster_centers_, axis=0)
    
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
    
    return w_stack_data
    
def compute_w_stacks(data, param_measop, devices):
    
    # Set up multi GPU parameters
    n_dev = len(devices)
    num_chs = data["nFreqs"]
    channel_lists = assign_channels_striped(num_chs, n_dev)
    
    # Compute field of view
    fov_radians = (
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )
    
    w_stack_data_list = []
    for d, dev in enumerate(devices):
        result = process_device(d, dev, data, param_measop, fov_radians, channel_lists, n_dev)
        w_stack_data_list.append(result)
    
    return w_stack_data_list

def compute_single_stack(data, param_measop, devices):
    """
    Build the same w_stack_data structure as compute_w_stacks, but as one trivial
    "stack" per device covering ALL of that device's baselines — for datasets with
    no w coordinate (or where w-correction/w-stacking isn't required). No k-means,
    no w-plane clustering; each device gets exactly one NUFFT plan and an identity
    correction term.
    """
    complex_dtype = torch.complex128 if param_measop["dtype"] == torch.float64 else torch.complex64

    w_stack_data_list = []
    for d, dev in enumerate(devices):
        u_i = data["u_dev"][d]
        v_i = data["v_dev"][d]
        n_vis_i = u_i.shape[-1]

        meas_op = MeasOpPytorchFinufft(
            u=u_i, v=v_i, img_size=param_measop["img_size"],
            real_flag=True, dtype=param_measop["dtype"], device=dev,
        )

        identity_correction = torch.ones(1, 1, *param_measop["img_size"], dtype=complex_dtype, device=dev)
        stack_idx = torch.zeros(n_vis_i, dtype=torch.int32, device=dev)  # every baseline in "stack 0"

        w_stack_data_list.append({
            "w_center": torch.zeros(1, device=dev),
            "corrections": [identity_correction],
            "stack_idx": stack_idx,
            "meas_op": [meas_op],
        })

    return w_stack_data_list