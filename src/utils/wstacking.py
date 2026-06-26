from typing import Tuple, Union
import torch
import gc, os
import numpy as np
import datetime
import math

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

def compute_w_stacks(w, num_wstacks, param_measop, data):
    from sklearn.cluster import KMeans
    
    w_np = w.numpy(force=True).reshape(-1, 1)
    # search for centres of w planes
    # parameters for k-means clustering, hard-coded for now
    kmeans_frac_pts = 0.01
    kmeans_max_pts = int(1e6)
    # kmeans_max_pts = int(1e6)
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
    # move results back to torch
    w_center = torch.as_tensor(
        centers,
        dtype=param_measop["dtype"],
        device=torch.device("cpu"),
    ).view(-1)

    w_stack_idx = torch.as_tensor(
        labels,
        dtype=torch.int32,
        device=torch.device("cpu"),
    ).view(-1)

    del w_kmeans, kmeans, labels, centers
    gc.collect()

    w_stack_sizes = [(w_stack_idx == i).sum().item() for i in range(num_wstacks)]

    # if self._verbose:
    print("INFO: w-stacking centers: ", end="")
    w_center_np = w_center.numpy(force=True).ravel()
    for i in range(num_wstacks):
        print(f"{w_center_np[i]:.7f}, ", end="")
    print("", flush=True)

    # check if w-correction is needed
    fov_radians = (
        (param_measop["im_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (param_measop["im_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )

    # create w-stacking correction term
    n_term = get_n_term(param_measop["img_size"], fov_radians, torch.device("cpu"), param_measop["dtype"])
    n_term_np = n_term.numpy(force=True)

    # Precompute per-stack w-corrections as CPU tensors (complex64 = float32 precision).
    # Keeping them as CPU tensors avoids a large numpy recompute on every forward/adjoint
    # call. Using complex64 halves RAM vs complex128: 66 × (4096^2 × 8 B) ≈ 8.6 GB → 4.3 GB.
    # They are moved to the target device inside forward_op / adjoint_op.
    w_center_np = w_center.numpy(force=True)
    w_stack_correct_cpu = [
        torch.as_tensor(
            (np.exp(-2j * np.pi * float(w_center_np[i]) * (n_term_np - 1)) / n_term_np).astype(
                np.complex64
                # np.complex128
            ),
        )  # shape (1,1,H,W), complex128, on CPU
        for i in range(num_wstacks)
    ]
    # self._w_stack_correct = (
    #     (torch.exp(-2 * 1j * np.pi * self._w_center.view(1, -1, 1, 1) * (n_term - 1)) / n_term)
    #     .pin_memory()
    #     .to(self._device[0], non_blocking=True)
    # )
    
    print("INFO: Computing measurement operators")
    meas_op = [None] * num_wstacks
    for i in range(num_wstacks):
        meas_op[i] = MeasOpPytorchFinufft(
            u=data["u"][:, :, w_stack_idx == i],
            v=data["v"][:, :, w_stack_idx == i],
            img_size=param_measop["img_size"],
            real_flag=True,
            dtype=param_measop["dtype"],
            device=param_measop["device"],
        )
        
    w_center = w_center.to(param_measop["device"])
    w_stack_idx = w_stack_idx.to(param_measop["device"])

    w_stack_correct = [
        corr.to(param_measop["device"])
        for corr in w_stack_correct_cpu
    ]
    
    return w_center, w_stack_correct, w_stack_idx, meas_op