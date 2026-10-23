from typing import Tuple, Union
import torch
import numpy as np
import math
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import os

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

def process_device_global(dev, data, param_measop, fov_radians, w_center, vis_exchange, rank=None):

    stack_list = vis_exchange.stack_list
    num_stacks = len(stack_list)
    w_center = w_center.to(dev)
    
    # Create w-stacking correction term
    n_term = get_n_term(param_measop["img_size"], fov_radians, dev, torch.float64)
    n_minus_1 = -(1 - n_term ** 2) / (1 + n_term)

    # Precompute per-stack w-corrections if w-stacking is needed
    complex_dtype = torch.complex128 if param_measop["dtype"] == torch.float64 else torch.complex64
    if w_center.numel() == 1:
        w_stack_correct = [torch.ones(1, 1, *param_measop["img_size"], dtype=complex_dtype, device=dev) for _ in stack_list]
    else:
        # w_stack_correct = [
        #     torch.exp(-2j * torch.pi * w_center[k] * (n_term - 1)) / n_term
        #     for k in stack_list
        # ]
        w_stack_correct = [
            (torch.exp(-2j * torch.pi * w_center[k].double() * n_minus_1) / n_term).to(complex_dtype)
            for k in stack_list
        ]
    
    plan_pair = None
    num_plans = param_measop["nufft_num_plans"]
    if num_plans is None or num_plans >= num_stacks:
        num_plans = num_stacks
        if rank == 0:
            print("INFO: No plan sharing. Each stack gets a dedicated plan.", flush=True)
            print(f"INFO: Created {num_plans} individual NUFFT plans.", flush=True)
    else:
        plan_pair = SharedNUFFTPlanPair(param_measop["img_size"], param_measop["dtype"], dev)
        if rank == 0:
            print(f"INFO: Created {num_plans} individual and one shared NUFFT plan.", flush=True)
        
    meas_op = [None] * num_stacks
    if num_plans == 0:
        individual_plan_indices = set()
    else:
        stride = num_stacks / num_plans
        individual_plan_indices = {int(round(i * stride)) for i in range(num_plans)}
    
    def _stack_slice(key, i):
        x = data[key]
        return x if x.numel() == 1 else x[:, :, vis_exchange.stack_slice(i)]
    
    for i in range(num_stacks):
        meas_op[i] = MeasOpPytorchFinufft(
            u=_stack_slice("u_dev", i),
            v=_stack_slice("v_dev", i),
            natural_weight=_stack_slice("nW_dev", i),
            image_weight=_stack_slice("nWimag_dev", i),
            img_size=param_measop["img_size"],
            real_flag=True,
            dtype=param_measop["dtype"],
            device=dev,
            shared_plan_pair=None if i in individual_plan_indices else plan_pair,
        )

    w_stack_data = {
        "w_center": w_center,
        "corrections": w_stack_correct,
        "stack_list": stack_list,
        "meas_op": meas_op,
        "plan_pair": plan_pair,
    }
    
    return w_stack_data

def wstack_cache_path(data_file, param_measop, image_pixel_size, cache_dir="wstacks"):
    """Unique, readable file name per w-stacking config."""
    dataset = os.path.basename(os.path.normpath(data_file))
    first_ch = param_measop["freq_num"]
    last_ch = first_ch + param_measop["nfreqs"] - 1
    name = (
        f"{dataset}_ch{first_ch}-{last_ch}"
        f"_img{param_measop['img_size'][0]}x{param_measop['img_size'][1]}"
        f"_px{image_pixel_size:.6g}.npz"
    )
    return os.path.join(cache_dir, name)


def _load_wstack_cache(cache_path, w, num_wstacks):
    """Returns the cached w-centres, or None if there is no valid cache file."""
    if cache_path is None or not os.path.exists(cache_path):
        return None
    cache = np.load(cache_path)
    # fingerprint of the w data: protects against stale files (e.g. changed flagging)
    valid = (
        int(cache["num_wstacks"]) == num_wstacks
        and int(cache["n_vis"]) == w.size
        and np.isclose(float(cache["w_min"]), float(w.min()))
        and np.isclose(float(cache["w_max"]), float(w.max()))
    )
    if not valid:
        print(f"WARNING: w-stacking cache {cache_path} does not match the data, recomputing.", flush=True)
        return None
    print(f"INFO: Loaded w-stacking centres from {cache_path}.", flush=True)
    return cache["centers"]


def _save_wstack_cache(cache_path, centers, w, num_wstacks):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez(f, centers=centers, num_wstacks=num_wstacks, n_vis=w.size,
                 w_min=float(w.min()), w_max=float(w.max()))
    os.replace(tmp_path, cache_path)   # atomic: a crashed run never leaves a half-written file
    print(f"INFO: Saved w-stacking centres to {cache_path}.", flush=True)

def compute_global_w_stacking(data, param_measop, cache_path=None):
    
    # Compute field of view
    fov_radians = (
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )
    data["fov_radians"] = fov_radians
    
    w = data["w"].numpy(force=True).reshape(-1, 1)
    
    num_wstacks = int(np.ceil(
        w.max() * 2 * np.pi * (1 - np.sqrt(1 - 2 * np.sin(fov_radians[0] / 2) ** 2)))
    )
    data["num_wstacks"] = num_wstacks
    
    if num_wstacks == 1:
        print(f"INFO: Number of w-stacks determined to be 1. Skipping w-stacking.")
        n_vis = data["u"].shape[-1]
        data["w_center"] = torch.zeros(1, dtype=param_measop["dtype"], device=torch.device("cpu"))
        data["stack_idx"] = torch.zeros((1, 1, n_vis), dtype=torch.int32, device=torch.device("cpu"))
        return data
    
    print(f"INFO: FOV in radians: {fov_radians}, max w value: {w.max():.4f}, number of w-stacks determined to be {num_wstacks} based on the FOV and max w value.", flush=True)

    centers = _load_wstack_cache(cache_path, w, num_wstacks)
    if centers is None:
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
        
        with threadpool_limits(limits=64):
            kmeans.fit(w_kmeans)

        centers = np.sort(kmeans.cluster_centers_, axis=0).ravel()

        if cache_path is not None:
            _save_wstack_cache(cache_path, centers, w, num_wstacks)

    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    w_t = torch.from_numpy(w.ravel()).to(dev, torch.float64)
    c_t = torch.from_numpy(np.asarray(centers, dtype=np.float64)).to(dev)
    idx_right = torch.searchsorted(c_t, w_t).clamp_(0, len(centers) - 1)
    idx_left = (idx_right - 1).clamp_(min=0)
    choose_right = (w_t - c_t[idx_right]).abs() < (w_t - c_t[idx_left]).abs()
    labels = torch.where(choose_right, idx_right, idx_left).to(torch.int32).cpu()
    del w_t, c_t, idx_right, idx_left, choose_right
    torch.cuda.empty_cache()

    print("INFO: w-stacking centers: ", end="")
    for i in range(num_wstacks):
        print(f"{centers[i].item():.7f}, ", end="")
    print("", flush=True)
    
    data["w_center"] = torch.tensor(centers, dtype=param_measop["dtype"], device=torch.device("cpu")).view(-1)
    data["stack_idx"] = torch.as_tensor(labels, dtype=torch.int32, device=torch.device("cpu")).view(1, 1, -1)
    
    return data