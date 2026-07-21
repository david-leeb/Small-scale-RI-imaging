"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits

import gc
import os

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight
from .utils.io_combined import load_dataset
from .utils.wstacking import compute_w_stacks, compute_single_stack
from src.mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from .utils.gpu_utils import mem, send_to_devices

torch.set_float32_matmul_precision('high')

def imager(param_optimiser: Dict, param_measop: Dict, param_proxop: Dict) -> None:
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
    
    torch.cuda.reset_peak_memory_stats()
    
    device = param_measop["device"]
    devices = None
    if device == torch.device("cuda"):
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        print("INFO: Detected", len(devices), "GPUs")

    # initialisation
    CACHE_PATH = f"./cache/{param_optimiser["src_name"]}_{param_measop["nfreqs"]}chs.pt"
    if os.path.exists(CACHE_PATH):
        print(f"INFO: Loading cached data from {CACHE_PATH}", flush=True)
        data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    else:
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
            # vis_remove=17.7,
            dl_shift=param_measop["dl_shift"],
            dm_shift=param_measop["dm_shift"],
            dtype=param_measop["dtype"],
            device=torch.device("cpu"),
        )
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        print(f"INFO: Saving to cache at {CACHE_PATH}", flush=True)
        torch.save(data, CACHE_PATH)
    
    weight_corr = None
    if param_measop["use_ROP"]:
        from .mrop_ri_measurement_operator import weighting_correction
        data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
        print(
            f"INFO: Correction has been applied to the weighting for {param_measop['ROP_param']['ROP_type']}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        
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
            
        param_measop["im_pixel_size"] = data["image_pixel_size"]
        
        B = int(data["B"] / data["nFreqs"])
        Q = int(param_measop["ROP_param"]["Q"])
        
        print(f"INFO: Original dimensions: N = {N}, Q = {Q}, K = {K}, B = {B}, N_ratio = {param_measop["ROP_param"]["N_ratio"]}.")
        epsilon, P_Q, M_B, M_K = solve_epsilon_same_aa(N, param_measop["ROP_param"]["Q"], B, K, N_ratio=param_measop["ROP_param"]["N_ratio"], n=param_measop["ROP_param"]["epsilon_n"], verbose=True)
        print(f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f} (epsilon = (N / Q^2VK)^(1/4)).")
        #! DEBUG
        M_K = 50
        M_B = 721
        P_Q = 59
        
        param_measop["ROP_param"]["M_K"] = M_K
        param_measop["ROP_param"]["M_B"] = M_B
        param_measop["ROP_param"]["P"] = P_Q 
        param_measop["ROP_param"]["M"] = M_K * M_B
        print(f"INFO: MROP set with P = {param_measop["ROP_param"]["P"]}, M_K = {param_measop["ROP_param"]["M_K"]}, M_B = {param_measop["ROP_param"]["M_B"]}, M = {param_measop["ROP_param"]["M"]}.")
        print(f"INFO: PM / N = {param_measop["ROP_param"]["P"] * param_measop["ROP_param"]["M"] / N:.4f}", flush=True)
    
    data = send_to_devices(data, devices)
    gc.collect()
    
    # for i in range(len(devices)):
    #     data["y_dev"][i] = data["y_dev"][i] * data["nW_dev"][i] * data["nWimag_dev"][i]
    
    param_measop["w_stacking"] = True
    if param_measop["w_stacking"]:
        w_stack_data_list = compute_w_stacks(data, param_measop, devices)
    else:
        w_stack_data_list = compute_single_stack(data, param_measop, devices)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    from .mrop_ri_measurement_operator import create_meas_op_ROP
    nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)
        
    meas_op = nufft_op(
        img_size=param_measop["img_size"],
        w_stack_data=w_stack_data_list,
        num_chs=data["nFreqs"],
        use_ROP=param_measop["use_ROP"],
        devices=devices,
        ROP_param=param_measop["ROP_param"] if param_measop["use_ROP"] else None,
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        image_weight=data["nWimag"],
        device=devices[0],
        dtype=param_measop["dtype"],
        real_flag=True,
    )
    
    data["y"] = meas_op.prepare_or_compress_data(data["y_dev"], weight=weight_corr)
    
    del data["u_dev"], data["v_dev"], data["w_dev"], data["nW_dev"], data["y_dev"]
    gc.collect()
    torch.cuda.empty_cache()
    
    meas_op_approx = None
    if param_optimiser["approx_meas_op"]:
        from .ri_measurement_operator.pysrc.measOperator.meas_op_PSF import MeasOpPSF

        meas_op_approx = MeasOpPSF(
            data["u"],
            data["v"],
            param_measop["img_size"],
            natural_weight=data["nW"],
            image_weight=data["nWimag"],
            real_flag=True,
            normalise_psf=False,
            device=param_measop["device"],
            dtype=param_measop["dtype"],
        )

    optimiser = None
    if param_optimiser["algorithm"] == "airi":
        prox_op_airi = ProxOpAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        optimiser = FBAIRI(
            y,
            meas_op,
            prox_op_airi,
            meas_op_approx=meas_op_approx,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            im_peak_est=param_optimiser["im_peak_est"],
            heu_noise_scale=param_optimiser["heu_noise_scale"],
            new_heu=param_optimiser["new_heu"],
            adapt_net_select=param_optimiser["dnn_adaptive_peak"],
            peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
            peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
            peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            iter_save=param_optimiser["itr_save"],
            verbose=param_optimiser["verbose"],
        )

    elif param_optimiser["algorithm"] == "cairi":
        prox_op_airi = ProxOpAIRI(
            param_proxop["dnn_shelf_path"],
            rand_trans=param_proxop["dnn_apply_transform"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )

        # preconditioning weight
        if param_optimiser["precond_flag"]:
            precond_weight = (
                torch.from_numpy(
                    gen_imaging_weight(
                        data["u"].cpu().numpy(),
                        data["v"].cpu().numpy(),
                        param_measop["img_size"],
                        weight_type="uniform",
                        grid_size=2,
                    ).reshape(1, 1, -1)
                )
                ** 2
            )
        else:
            precond_weight = torch.ones(1, 1)

        # Theoretical l2 error bound, assume chi-square distribution, tau=1
        l2_bound = np.sqrt(torch.numel(y) + 2.0 * np.sqrt(torch.numel(y)))
        if param_optimiser["verbose"]:
            print(
                "INFO: The theoretical l2 error bound is",
                f"{l2_bound}",
            )

        prox_op_dual_data = ProxOpElipse(
            center=y,
            precond_weight=precond_weight,
            radius=l2_bound,
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            y,
            meas_op,
            prox_op_airi,
            prox_op_dual_data,
            im_min_itr=param_optimiser["im_min_itr"],
            im_max_itr=param_optimiser["im_max_itr"],
            im_var_tol=param_optimiser["im_var_tol"],
            im_peak_est=param_optimiser["im_peak_est"],
            heu_noise_scale=param_optimiser["heu_noise_scale"],
            adapt_net_select=param_optimiser["dnn_adaptive_peak"],
            peak_tol_min=param_optimiser["dnn_adaptive_peak_tol_min"],
            peak_tol_max=param_optimiser["dnn_adaptive_peak_tol_max"],
            peak_tol_step=param_optimiser["dnn_adaptive_peak_tol_step"],
            save_pth=param_optimiser["result_path"],
            file_prefix=param_optimiser["file_prefix"],
            iter_save=param_optimiser["itr_save"],
            verbose=param_optimiser["verbose"],
        )

    elif param_optimiser["algorithm"] == "usara":
        prox_op_sara = ProxOpSARAPos(
            param_measop["img_size"],
            device=param_proxop["device"],
            dtype=param_proxop["dtype"],
            verbose=param_proxop["verbose"],
        )
        gc.collect()
        torch.cuda.empty_cache()

        optimiser = FBSARA(
            data["y"],
            meas_op,
            prox_op_sara,
            use_ROP=param_measop["use_ROP"],
            meas_op_approx=meas_op_approx,
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
            verbose=param_optimiser["verbose"],
        )
        gc.collect()
        torch.cuda.empty_cache()

    # imaging
    if param_optimiser["flag_imaging"]:
        # initialisation
        optimiser.initialisation()

        #! DEBUG: run measurement operator and adjoint to check correctness
        from src.mrop_ri_measurement_operator.test_meas_op import test_adjoint_op
        test_adjoint_op(meas_op, param_measop["img_size"], param_measop["dtype"])
        
        # run imaging loop
        optimiser.run()
        # finalisation
        optimiser.finalisation()

        # calculate final metrics
        if param_optimiser["verbose"]:
            img_model = optimiser.get_model_image()
            img_residual = optimiser.get_residual_image()
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
        
        for idx, dev in enumerate(devices):
            free, total = torch.cuda.mem_get_info(dev)
            driver_used = (total - free) / 1024**3
            torch_reserved = torch.cuda.memory_reserved(dev) / 1024**3
            print(f"dev={idx} driver_used={driver_used:.2f} GB torch_reserved={torch_reserved:.2f} GB non-torch={driver_used - torch_reserved:.2f} GB")