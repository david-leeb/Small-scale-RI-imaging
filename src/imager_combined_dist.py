"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

import gc
from typing import Dict
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .prox_operator import ProxOpSARAPos
from .optimiser import FBSARADist
from .utils.io_combined import load_dataset
from .utils.wstacking import compute_global_w_stacking, process_device_global
from .mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from .mrop_ri_measurement_operator import weighting_correction

from .utils.gpu_utils_dist import (
    mem, broadcast_object,
    scatter_channel_data, assign_channels_striped
)

#! Crucial to avoid underflow for single precision
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

def imager(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict, rank: int, world_size: int, device: torch.device) -> None:
    """
    Imager for small scale RI imaging task.

    This function prepares the measurement operator, prior, and algorithm for the imaging task.
    It supports different algorithms such as 'airi', 'usara', and 'cairi'. The function also
    handles the imaging process if the 'flag_imaging' is set in the 'param_optimiser'.

    Args:
        param_optimiser (dict): A dictionary containing the parameters for the optimiser.
            It includes parameters like 'algorithm', 'im_min_itr', 'im_max_itr', 'im_var_tol',
            'im_peak_est', 'heu_noise_scale', 'dnn_adaptive_peak', 'dnn_adaptive_peak_tol_min',
            'dnn_adaptive_peak_tol_max', 'dnn_adaptive_peak_tol_step', 'result_path', 'itr_save',
            'verbose', and 'flag_imaging'.
        param_measop (dict): A dictionary containing the parameters for the measurement operator.
            It includes parameters like 'superresolution', 'im_pixel_size', 'flag_data_weighting',
            'weight_load', 'img_size', 'weight_type', 'weight_gridsize', 'weight_robustness',
            'dtype', 'device', 'nufft_grid_size', 'nufft_kb_kernel_dim', and 'nufft_mode'.
        param_proxop (dict): A dictionary containing the parameters for the proximal operator.
            It includes parameters like 'dnn_shelf_path', 'dnn_apply_transform', 'device', 'dtype',
            and 'verbose'.
    """
    
    is_root = rank == 0
    if param_measop["nfreqs"] < world_size:
        print("Warning: Number of channels has to be larger than number of GPUs")  # Optional
        return
    
    # Add reconstruction method to file prefix
    use_ROP = param_measop["use_ROP"]
    param_optimiser["file_prefix"] = (str(param_measop["ROP_param"]["ROP_type"]) if use_ROP else "classical") + "_" + param_optimiser["file_prefix"]

    data = None
    metadata = None
    weight_corr = None
    if is_root:
        # Load dataset on the CPU
        data = load_dataset(
            data_path=param_optimiser["data_file"],
            Q=param_measop["Q"],
            super_resolution=param_measop["superresolution"],
            image_pixel_size=param_measop["im_pixel_size"],
            img_size=param_measop["img_size"],
            nfreqs=param_measop["nfreqs"],
            freq_num=param_measop["freq_num"],
            data_weighting=param_measop["flag_data_weighting"],
            weight_type=param_measop["weight_type"],
            weight_robustness=param_measop["weight_robustness"],
            vis_remove=param_measop["vis_remove"],
            dl_shift=param_measop["dl_shift"],
            dm_shift=param_measop["dm_shift"],
            dtype=param_measop["dtype"],
            device=torch.device("cpu"),
        )
    
        # Default to no weight correction, only applied for ROP and if flag is set
        if use_ROP:
            if param_measop["ROP_param"]["Q"] is None:
                assert "Q" in data, "number of anntennas Q is not in data and not provided"
                param_measop["ROP_param"]["Q"] = int(data["Q"])

            N = int(np.prod(param_measop["img_size"]))
            K = int(data["nFreqs"])
            V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
                
            if param_measop["ROP_param"]["B"] is None:
                if "flag" in data and data["flag"] is not None and "B" not in data:
                    data["B"] = data["flag"].shape[-1] / V
                assert "B" in data, "number of snapshots B is not in data and not provided"
                param_measop["ROP_param"]["B"] = int(data["B"])
                print("INFO: B set to ", int(data["B"]))
            
            B = int(data["B"] / data["nFreqs"])
            Q = int(param_measop["ROP_param"]["Q"])
            
            print(f"INFO: Original dimensions: N = {N}, Q = {Q}, K = {K}, B = {B}, N_ratio = {param_measop["ROP_param"]["N_ratio"]}.")
            epsilon, P_Q, M_B, M_K = solve_epsilon_same_aa(N, param_measop["ROP_param"]["Q"], B, K, N_ratio=param_measop["ROP_param"]["N_ratio"], n=param_measop["ROP_param"]["epsilon_n"], verbose=True)
            print(f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f} (epsilon = (N / Q^2VK)^(1/4)).")
            
            param_measop["ROP_param"]["M_K"] = M_K
            param_measop["ROP_param"]["M_B"] = M_B
            param_measop["ROP_param"]["P"] = P_Q 
            param_measop["ROP_param"]["M"] = M_K * M_B
            print(f"INFO: MROP set with P = {param_measop["ROP_param"]["P"]}, M_K = {param_measop["ROP_param"]["M_K"]}, M_B = {param_measop["ROP_param"]["M_B"]}, M = {param_measop["ROP_param"]["M"]}.")
            print(f"INFO: PM / N = {param_measop["ROP_param"]["P"] * param_measop["ROP_param"]["M"] / N:.4f}", flush=True) #! FIX THAT, WHICH P TO USE?
                    
            # Add ROP specific parameters to file prefix
            param_optimiser["file_prefix"] = param_optimiser["file_prefix"] + (
                "N_ratio_" + str(param_measop["ROP_param"]["N_ratio"]) + 
                "_epsilon_n_" + str(param_measop["ROP_param"]["epsilon_n"]) + 
                "_P_" + str(P_Q) + "_MB_" + str(M_B) + "_MK_" + str(M_K) + "_" 
            )
            
            data, weight_corr = weighting_correction(data, param_measop["ROP_param"], rapha=True)
            torch.cuda.empty_cache()
    
        data["y"] = data["y"] * data["nW"] * data["nWimag"]    
        data = compute_global_w_stacking(data, param_measop)
        
        # Needed on every device
        metadata = dict(
            nFreqs=int(data["nFreqs"]),
            Q=int(data["Q"]),
            B=int(data.get("B", 0)),
            B_per_ch=int(data.get("B_per_ch", 0)),
            image_pixel_size=float(data["image_pixel_size"]),
            fov_radians=data["fov_radians"],
            num_wstacks=int(data["num_wstacks"]),
            n_vis_total=int(data["u"].shape[-1]),
            chan_offsets=[int(x) for x in data["chan_offsets"]],
            w_center=data["w_center"].tolist(),
        )
    
    metadata = broadcast_object(metadata, src=0)
    if use_ROP:
        param_measop["ROP_param"] = broadcast_object(
            param_measop["ROP_param"] if is_root else None, src=0
        )
        weight_corr = broadcast_object(weight_corr if is_root else None, src=0)
        weight_corr = weight_corr.to(device=device, dtype=param_measop["dtype"])
    
    channel_lists = assign_channels_striped(metadata["nFreqs"], world_size)
    
    w_center = torch.tensor(metadata["w_center"], dtype=param_measop["dtype"], device=device)
 
    complex_dtype = torch.complex128 if param_measop["dtype"] == torch.float64 else torch.complex64
    dtypes = {
        "u": param_measop["dtype"], "v": param_measop["dtype"], "w": param_measop["dtype"],
        "nW": param_measop["dtype"], "nWimag": param_measop["dtype"],
        "y": complex_dtype, "stack_idx": torch.int32,
    }
    scatter_keys = ["u", "v", "w", "nW", "y", "nWimag", "stack_idx"]
    data = data if is_root else {}
    data = scatter_channel_data(
        data, scatter_keys, channel_lists, metadata["chan_offsets"], device, dtypes, src=0
    )
    torch.cuda.empty_cache()
 
    if is_root:
        del data["u"], data["v"], data["w"], data["y"]
        gc.collect()
 
    local_data_for_wstack = {
        "u_dev": [data["u_dev"]], "v_dev": [data["v_dev"]],
        "nW_dev": [data["nW_dev"]], "nWimag_dev": [data["nWimag_dev"]],
        "stack_idx_dev": [data["stack_idx_dev"]],
    }
    w_stack_data = process_device_global(
        0, device, local_data_for_wstack, param_measop,
        metadata["fov_radians"], metadata["num_wstacks"], w_center, rank=rank
    )
    gc.collect()
    torch.cuda.empty_cache()
    
    from .mrop_ri_measurement_operator.src.mrop_dist import create_meas_op_ROP_dist
    nufft_op = create_meas_op_ROP_dist(MeasOpPytorchFinufft)
    meas_op = nufft_op(
        img_size=param_measop["img_size"],
        w_stack_data=w_stack_data,
        num_chs=metadata["nFreqs"],
        rank=rank,
        world_size=world_size,
        use_ROP=use_ROP,
        ROP_param=param_measop["ROP_param"] if use_ROP else None,
        ant1=data["ant1"] if is_root else None,
        ant2=data["ant2"] if is_root else None,
        batches=data["batches"] if is_root else None,
        device=device,
        dtype=param_measop["dtype"],
        real_flag=True,
        verbose=is_root
    )
    torch.cuda.empty_cache()
    
    # Gather compressed data on GPU 0 for Optimiser
    y_local = data["y_dev"].to(device=device, dtype=meas_op._dtype_meas)
    y_compressed = None
    if use_ROP:
        y_compressed, compression_ratio = meas_op.compress_data(y_local)
        if is_root:
            param_optimiser["file_prefix"] += f"cr_{compression_ratio:.2f}_"
            
    del data["u_dev"], data["v_dev"], data["w_dev"], data["nW_dev"], data["nWimag_dev"], data["y_dev"]

    gc.collect()
    torch.cuda.empty_cache()

    prox_op_sara = None
    if is_root:
        if param_optimiser["algorithm"] == "usara":
            prox_op_sara = ProxOpSARAPos(
                param_measop["img_size"],
                device=param_proxop["device"],
                dtype=param_proxop["dtype"],
                verbose=param_proxop["verbose"],
            )
        else:
            print("For now only uSARA is implemented.")
            return
    torch.cuda.empty_cache()
    
    optimiser = FBSARADist(
        y_compressed if use_ROP else y_local.view(1, 1, -1),
        meas_op,
        prox_op_sara,
        rank=rank,
        use_ROP=use_ROP,
        y_uncompressed=y_local.view(1, 1, -1) if use_ROP else None,
        weight_correction=weight_corr if use_ROP else None,
        im_min_itr=param_optimiser["im_min_itr"],
        im_max_itr=param_optimiser["im_max_itr"],
        im_var_tol=param_optimiser["im_var_tol"],
        heu_reg_scale=param_optimiser["heu_reg_param_scale"],
        new_heu=param_optimiser["new_heu"],
        im_max_itr_outer=param_optimiser["im_max_outer_itr"],
        im_var_tol_outer=param_optimiser["im_var_outer_tol"],
        save_pth=param_optimiser["result_path"],
        file_prefix=param_optimiser["file_prefix"],
        reweight_save=param_optimiser["reweighting_save"],
        verbose=param_optimiser["verbose"] and is_root,
    )
    torch.cuda.empty_cache()

    # imaging
    if param_optimiser["flag_imaging"]:
        # initialisation
        optimiser.initialisation()

        #! DEBUG: run measurement operator and adjoint to check correctness
        # from src.mrop_ri_measurement_operator.test_meas_op import test_adjoint_op
        # test_adjoint_op(meas_op, param_measop["img_size"], param_measop["dtype"])

        # run imaging loop
        optimiser.run()

        # finalisation
        optimiser.finalisation()

        # calculate final metrics
        img_residual = None
        if param_optimiser["verbose"]:
            img_residual = optimiser.get_residual_image()
            
        if param_optimiser["verbose"] and is_root:
            img_model = optimiser.get_model_image()
            img_dirty = optimiser.get_dirty_image()
            psf = optimiser.get_psf()

            img_residual_std = np.std(img_residual).item()
            img_residual_std_noramalised = img_residual_std / psf.max().item()
            img_residual_ratio = np.linalg.norm(img_residual.flatten()) / np.linalg.norm(img_dirty.flatten())
            print(
                "INFO: The standard deviation of the final",
                f"residual dirty image is {img_residual_std}",
            )
            print(
                "INFO: The standard deviation of the normalised",
                f"final residual dirty image is {img_residual_std_noramalised}",
            )
            print(
                "INFO: The ratio between the norm of the residual",
                f"and the dirty image: ||residual|| / || dirty || = {img_residual_ratio}",
            )

            if param_optimiser["groundtruth"]:
                img_gdth = fits.getdata(param_optimiser["groundtruth"]).astype(np.double)
                rsnr = 20 * np.log10(
                    np.linalg.norm(img_gdth.flatten())
                    / np.linalg.norm(img_gdth.flatten() - img_model.flatten())
                )
                print(
                    "INFO: The signal-to-noise ratio of the final",
                    f"reconstructed image is {rsnr} dB",
                )
        
        free, total = torch.cuda.mem_get_info(device)
        driver_used = (total - free) / 1024**3
        torch_reserved = torch.cuda.memory_reserved(device) / 1024**3
        dist.barrier()
        print(
            f"rank={rank} driver_used={driver_used:.2f} GB torch_reserved={torch_reserved:.2f} GB "
            f"non-torch={driver_used - torch_reserved:.2f} GB",
            flush=True,
        )