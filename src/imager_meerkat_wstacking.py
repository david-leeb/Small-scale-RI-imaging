"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits

import ctypes
import gc

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight
# from .ri_measurement_operator.pysrc.utils.io import load_data_to_tensor
# from .ri_measurement_operator.pysrc.utils.io_new import load_data_to_tensor
from .utils.io_meerkat import load_real_data_to_tensor
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft_wstacking import MeasOpPytorchFinufftWStacking
from .utils.wstacking import compute_w_stacks
from src.mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa

torch.set_float32_matmul_precision('high')

def _mem(label, devices):
    for idx, dev in enumerate(devices):
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        peak = torch.cuda.max_memory_allocated(dev) / 1024**3
        free, total = torch.cuda.mem_get_info(dev)
        driver = (total - free) / 1024**3
        print(f"[MEM] {label:<45} dev={idx} torch={alloc:.2f} GB  peak={peak:.2f} GB  driver={driver:.2f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats(dev)

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
    _mem("START", [param_measop["device"]])

    # initialisation
    data = load_real_data_to_tensor(
        data_path=param_optimiser["data_file"],
        super_resolution=param_measop["superresolution"],
        image_pixel_size=param_measop["im_pixel_size"],
        img_size=param_measop["img_size"],
        nfreqs=param_measop["nfreqs"],
        freq_num=param_measop["freq_num"],
        data_weighting=param_measop["flag_data_weighting"],
        weight_type=param_measop["weight_type"],
        weight_robustness=param_measop["weight_robustness"],
        device=param_measop["device"],
    )
    
    _mem("after load_real_data_to_tensor", [param_measop["device"]])
    
    from .mrop_ri_measurement_operator import weighting_correction
    data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
    print(
        f"INFO: Correction has been applied to the weighting for {param_measop['ROP_param']['ROP_type']}",
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after weighting_correction", [param_measop["device"]])
    
    if param_measop["ROP_param"]["Q"] is None:
        assert "Q" in data, "number of anntennas Q is not in data and not provided"
        param_measop["ROP_param"]["Q"] = int(data["Q"])

    N = int(np.prod(param_measop["img_size"]))
    K = int(data["nFreqs"])
    V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
        
    if param_measop["ROP_param"]["B"] is None:
        if "flag" in data and data["flag"] is not None and "B" not in data:
            data["B"] = data["flag"].shape[-1] / V #! VERIFY 
        assert "B" in data, "number of snapshots B is not in data and not provided"
        param_measop["ROP_param"]["B"] = int(data["B"])
        print("INFO: B set to ", int(data["B"]))
        
    param_measop["im_pixel_size"] = data["image_pixel_size"]
    
    # B = int(data["B_per_ch"]) 
    B = int(data["B"] / data["nFreqs"]) #! CHECK B ALLOCATION
    Q = int(param_measop["ROP_param"]["Q"])
    
    print(f"INFO: Original dimensions: N = {N}, Q = {Q}, K = {K}, B = {B}, N_ratio = {param_measop["ROP_param"]["N_ratio"]}.")
    epsilon, P_Q, M_B, M_K = solve_epsilon_same_aa(N, param_measop["ROP_param"]["Q"], B, K, param_measop["ROP_param"]["N_ratio"], param_measop["ROP_param"]["epsilon_n"])
    print(f"INFO: Calculated epsilon for MROP modulation dimensions: {epsilon:.4f} (epsilon = (N / Q^2VK)^(1/4)).")
    param_measop["ROP_param"]["M_K"] = M_K
    param_measop["ROP_param"]["M_B"] = M_B
    param_measop["ROP_param"]["P"] = P_Q #* (P_Q - 1) // 2
    param_measop["ROP_param"]["M"] = M_K * M_B
    print(f"INFO: MROP set with P = {param_measop["ROP_param"]["P"]}, M_K = {param_measop["ROP_param"]["M_K"]}, M_B = {param_measop["ROP_param"]["M_B"]}, M = {param_measop["ROP_param"]["M"]}.")
    print(f"INFO: PM / N = {param_measop["ROP_param"]["P"] * param_measop["ROP_param"]["M"] / N:.4f}", flush=True)
    
    device = param_measop["device"]
    devices = None
    if device == torch.device("cuda"):
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        print("INFO: Detected", len(devices), "GPUs")
    #! consider multi GPU setup here to make sure all functions have the same params
    
    w_stack_data_list = compute_w_stacks(data, param_measop, devices)
    del data["u"], data["v"], data["w"], data["nW"]
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after compute_w_stacks  (NUFFTs + corrections)", devices)
    
    from .mrop_ri_measurement_operator import create_meas_op_ROP as create_meas_op_ROP
    from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
    nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)

    meas_op = nufft_op(
        # u=data["u"],
        # v=data["v"],
        flag=data["flag"],
        img_size=param_measop["img_size"],
        # natural_weight=data["nW"],
        image_weight=data["nWimag"],
        device=devices[0],
        dtype=param_measop["dtype"],
        num_chs=data["nFreqs"],
        ROP_param=param_measop["ROP_param"],
        real_flag=True,
        w_stack_data=w_stack_data_list,
        devices=devices
    )
    
    _mem("after nufft_op.__init__  (alpha,C_buf,flat_sym,wstack bufs)", devices)
    
    del data["flag"]
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after deleting raw flag tensor", devices)
    
    if param_measop["use_ROP"]:
        original_data_size = data["y"].numel()
        print(f"INFO: data size before {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}", flush=True)
        if param_measop["ROP_param"]["ROP_type"] in ["MROP", "MROP_gaussian"]:
            # data["y"] = meas_op.MD(data["y"] * weight_corr)
            data["y"] = meas_op.compress_data(data["y"], weight=weight_corr)
        elif param_measop["ROP_param"]["ROP_type"] == "CROP":
            data["y"] = meas_op.D(data["y"] * weight_corr)
        print(f"INFO: data size after {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}", flush=True)
        print(f"INFO: compression ratio is {original_data_size / data['y'].numel()}", flush=True)
    
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after MD() compression  (temporaries Z,Y,y_triu peak here)", devices)
    
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
        _mem("after ProxOpSARAPos.__init__  (dual/weights/psit 1D bufs)", devices)

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
        _mem("after FBSARA.__init__  (dual/weights/psit 1D bufs)", devices)

    # imaging
    if param_optimiser["flag_imaging"]:
        # initialisation
        torch.cuda.empty_cache()
        _mem("before initialisation() after cache clear", devices)
        optimiser.initialisation()
        _mem("after initialisation()  (op_norm power iteration)", devices)

        #! DEBUG: run measurement operator and adjoint to check correctness
        from src.mrop_ri_measurement_operator.test_meas_op import test_adjoint_op
        test_adjoint_op(meas_op, param_measop["img_size"], param_measop["dtype"])
        
        # run imaging loop
        _mem("before imaging loop", devices)
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