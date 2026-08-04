from typing import Tuple, Union
import torch
import numpy as np
import math
from sklearn.cluster import KMeans

from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft, SharedNUFFTPlanPair

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

def process_device_global(d, dev, data, param_measop, fov_radians, num_wstacks, w_center):

    data_i = {
        "u": data["u_dev"][d],
        "v": data["v_dev"][d],
    }
    w_stack_idx = data["stack_idx_dev"][d].to(dev).view(-1)
    w_center = w_center.to(dev)
    
    # create w-stacking correction term
    n_term = get_n_term(param_measop["img_size"], fov_radians, dev, param_measop["dtype"])

    # Precompute per-stack w-corrections
    w_stack_correct = [
        torch.exp(-2j * torch.pi * w_center[i] * (n_term - 1)) / n_term
        for i in range(num_wstacks)
    ]
    
    plan_pair = None
    if param_measop["reduce_memory_usage"]:
        plan_pair = SharedNUFFTPlanPair(param_measop["img_size"], param_measop["dtype"], dev)

    meas_op = [None] * num_wstacks
    for i in range(num_wstacks):
        meas_op[i] = MeasOpPytorchFinufft(
            u=data_i["u"][:, :, w_stack_idx == i],
            v=data_i["v"][:, :, w_stack_idx == i],
            img_size=param_measop["img_size"],
            real_flag=True,
            dtype=param_measop["dtype"],
            device=dev,
            shared_plan_pair=plan_pair,
        )

    w_stack_data = {
        "w_center": w_center,
        "corrections": w_stack_correct,
        "stack_idx": w_stack_idx,
        "meas_op": meas_op,
        "plan_pair": plan_pair,
    }
    
    return w_stack_data

def compute_global_w_stacking(data, param_measop):
    
    # Compute field of view
    fov_radians = (
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )
    
    w = data["w"].numpy(force=True).reshape(-1, 1)
    
    num_wstacks = int(np.ceil(
        w.max() * 2 * np.pi * (1 - np.sqrt(1 - 2 * np.sin(fov_radians[0] / 2) ** 2)))
    )
    
    print(f"INFO: FOV in radians: {fov_radians}, max w value: {w.max():.4f}, number of w-stacks determined to be {num_wstacks} based on the FOV and max w value.", flush=True)
    
    # search for centres of w planes
    # parameters for k-means clustering, hard-coded for now
    kmeans_frac_pts = 0.01
    kmeans_max_pts = int(1e6)
    kmeans_max_iter = 1000

    # run k-means on a subset of w values
    kmeans_num_pts = min(int(kmeans_frac_pts * w.size), kmeans_max_pts)

    np.random.seed(42)
    idx = np.random.choice(w.size, kmeans_num_pts, replace=False)
    w_kmeans = w[idx].reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=num_wstacks,
        random_state=0,
        max_iter=kmeans_max_iter,
        n_init=30,
        tol=1e-6,
    )
    
    kmeans.fit(w_kmeans)
    centers = np.sort(kmeans.cluster_centers_, axis=0).ravel()
    
    w_flat = w.ravel()
    idx_right = np.clip(np.searchsorted(centers, w_flat), 0, len(centers) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(centers) - 1)
    choose_right = np.abs(w_flat - centers[idx_right]) < np.abs(w_flat - centers[idx_left])
    labels = np.where(choose_right, idx_right, idx_left)

    print("INFO: w-stacking centers: ", end="")
    for i in range(num_wstacks):
        print(f"{centers[i].item():.7f}, ", end="")
    print("", flush=True)
    
    data["w_center"] = torch.tensor(centers, dtype=param_measop["dtype"], device=torch.device("cpu")).view(-1)
    data["stack_idx"] = torch.as_tensor(labels, dtype=torch.int32, device=torch.device("cpu")).view(1, 1, -1)
    data["num_wstacks"] = num_wstacks
    data["fov_radians"] = fov_radians
    return data

def compute_w_stacks(data, param_measop, devices):
    w_stack_data_list = [
        process_device_global(
            d, dev, data, param_measop,
            data["fov_radians"], data["num_wstacks"], data["w_center"]
        )
        for d, dev in enumerate(devices)
    ]
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