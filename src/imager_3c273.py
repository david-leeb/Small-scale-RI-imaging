"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits
import nvtx

torch.set_float32_matmul_precision('high')
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos, ProxOpSARAPos_original
from .optimiser import FBAIRI, PDAIRI, FBSARA
from .utils import gen_imaging_weight
# from .ri_measurement_operator.pysrc.utils.io import load_data_to_tensor
from .utils.io_3c273 import load_data_to_tensor

def solve_epsilon_same_aa(N, Q, B, K, N_ratio=1.0, n=1.0, verbose=False):
    from scipy.optimize import fsolve

    # Polynomial from P_Q = eps*Q, M_B = eps/n*B, M_K = eps/n*K and D = N*N_ratio:
    # 0 = eps^4 - eps^3/Q - 2n^2*N*N_ratio/(Q^2*B*K)
    c = 2 * n**2 * N * N_ratio / (Q**2 * B * K)
    fun = lambda eps: eps**4 - eps**3 / Q - c
    epsilon = float(np.clip(fsolve(fun, c ** (1 / 4))[0], 0.0, 1.0))

    P_Q = min(max(2, int(np.round(epsilon * Q))), Q)
    M_K = min(max(1, int(np.round(epsilon / n * K))), K)

    # Back-solve M_B from P_Q*(P_Q-1)/2 * M_B * M_K = N * N_ratio
    mb_target = N * N_ratio / (P_Q * (P_Q - 1) / 2 * M_K)
    M_B_lo = min(max(1, int(np.floor(mb_target))), B)
    M_B_hi = min(max(1, int(np.ceil(mb_target))), B)
    D_lo = P_Q * (P_Q - 1) / 2 * M_B_lo * M_K
    D_hi = P_Q * (P_Q - 1) / 2 * M_B_hi * M_K
    M_B = M_B_lo if abs(D_lo - N * N_ratio) <= abs(D_hi - N * N_ratio) else M_B_hi

    D = P_Q * (P_Q - 1) / 2 * M_B * M_K
    if verbose:
        print(f"Q -> P_Q: {Q} -> {P_Q}; B -> M_B: {B} -> {M_B}; K -> M_K: {K} -> {M_K}")
        print(f"D = {D}; D/N = {D/N:.6f}; N_ratio = {N_ratio}")
    return epsilon, P_Q, M_B, M_K

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
    # initialisation
    
    data = load_data_to_tensor(
        main_data_file=param_optimiser["data_file"],
        data_path="/".join(param_optimiser["data_file"].split("/")[:-1]),
        super_resolution=param_measop["superresolution"],
        data_weighting=param_measop["flag_data_weighting"],
        load_weight=param_measop["weight_load"],
        img_size=param_measop["img_size"],
        weight_type=param_measop["weight_type"],
        weight_robustness=param_measop["weight_robustness"],
        nfreqs=param_measop["nfreqs"],
        freq_num=param_measop["freq_num"],
        use_ROP=param_measop["use_ROP"],
        vis_remove=17.7,
        dl_shift=param_measop["dl_shift"], #128 
        dm_shift=param_measop["dm_shift"], #-128
        dtype=param_measop["dtype"],
        device=param_measop["device"],
        # verbose=param_optimiser["verbose"],
    )
    
    # if data["nFreqs"] == 1:
    #     data["flag"] = data["flag"][:, 0, :].unsqueeze(1)

    if param_measop["ROP_param"]["Q"] is None:
        assert "Q" in data, "number of anntennas Q is not in data and not provided"
        param_measop["ROP_param"]["Q"] = int(data["Q"])

    N = int(np.prod(param_measop["img_size"]))
    K = int(data["nFreqs"])
    V = int(param_measop["ROP_param"]["Q"] * (param_measop["ROP_param"]["Q"] - 1) // 2)
    B = int(data["B_per_ch"])
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

    if param_measop["ROP_param"]["B"] is None:
        if "flag" in data and data["flag"] is not None and "B" not in data:
            data["B"] = data["flag"].shape[-1] / V * K
        assert "B" in data, "number of snapshots B is not in data and not provided"
        param_measop["ROP_param"]["B"] = int(data["B"])
    
    from .mrop_ri_measurement_operator import weighting_correction
    data, weight_corr = weighting_correction(data, param_measop["ROP_param"])
    print(
        f"INFO: Correction has been applied to the weighting for {param_measop['ROP_param']['ROP_type']}",
        flush=True,
    )
    
    meas_op = None

    if not param_measop["use_ROP"]:
        from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
        nufft_op = MeasOpPytorchFinufft
    else:
        # if param_measop["ROP_param"]["ROP_batchwise"]:
        #     if param_optimiser.get("nfreqs", data["nFreqs"]) in [None, 1]:
        #         from .mrop_ri_measurement_operator import create_meas_op_ROP_batchwise as create_meas_op_ROP
        #     else:
        #         from .mrop_ri_measurement_operator import create_meas_op_ROP_batchwise_mf as create_meas_op_ROP
        # elif param_measop["ROP_param"]["ROP_vmap"]:
        #     if param_optimiser.get("nfreqs", data["nFreqs"]) in [None, 1]:
        #         from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap as create_meas_op_ROP

        #         print("INFO: Using vmap ROP for single frequency data", flush=True)
        #     else:
        #         if param_measop["ROP_param"]["freq_mod"]:
        #             from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap_mf_bf_mod as create_meas_op_ROP
        #             print("INFO: Using vmap ROP for multi-frequency data, treating frequency dimension as batches", flush=True)
        #         else:
        #             from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap_mf as create_meas_op_ROP
        #             print("INFO: Using vmap ROP for multi-frequency data", flush=True)
        # else:
        #     from .mrop_ri_measurement_operator import create_meas_op_ROP

        ORIGINAL = False
        if param_measop["ROP_param"]["ROP_vmap"]:
            from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft_original import MeasOpPytorchFinufft
            from .mrop_ri_measurement_operator import create_meas_op_ROP_vmap as create_meas_op_ROP
        elif ORIGINAL:
            from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
            from .mrop_ri_measurement_operator import create_meas_op_ROP_new_taylor as create_meas_op_ROP
        else:
            from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
            from .mrop_ri_measurement_operator import create_meas_op_ROP as create_meas_op_ROP
        nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)

    meas_op = nufft_op(
        u=data["u"],
        v=data["v"],
        img_size=param_measop["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        device=param_measop["device"],
        dtype=param_measop["dtype"],
        num_chs=data["nFreqs"],
        ROP_param=param_measop["ROP_param"],
        ant1=data.get("ant1", None),
        ant2=data.get("ant2", None),
        batches=data.get("batches", None),
    )

    with torch.cuda.nvtx.range("Preprocessing_ROP"):
        if param_measop["use_ROP"]:
            print(f"INFO: data size before {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}", flush=True)
            if param_measop["ROP_param"]["ROP_type"] in ["MROP", "MROP_gaussian"]:
                data["y"] = meas_op.MD(data["y"] * weight_corr)
            elif param_measop["ROP_param"]["ROP_type"] == "CROP":
                data["y"] = meas_op.D(data["y"] * weight_corr)
            print(f"INFO: data size after {param_measop['ROP_param']['ROP_type']} is {data['y'].numel()}", flush=True)
            
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
            data["y"],
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
        l2_bound = np.sqrt(torch.numel(data["y"]) + 2.0 * np.sqrt(torch.numel(data["y"])))
        if param_optimiser["verbose"]:
            print(
                "INFO: The theoretical l2 error bound is",
                f"{l2_bound}",
            )

        prox_op_dual_data = ProxOpElipse(
            center=data["y"],
            precond_weight=precond_weight,
            radius=l2_bound,
            device=meas_op.get_device(),
            dtype=meas_op.get_data_type_meas(),
        )

        optimiser = PDAIRI(
            data["y"],
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
        with torch.cuda.nvtx.range("prox_op_sara_initialisation"):
            prox_op_class = ProxOpSARAPos if param_proxop["use_optimized"] else ProxOpSARAPos_original
            prox_op_sara = prox_op_class(
                param_measop["img_size"],
                device=param_proxop["device"],
                dtype=param_proxop["dtype"],
                verbose=param_proxop["verbose"],
            )

        with torch.cuda.nvtx.range("FBSARA_initialisation"):
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

    # imaging
    if param_optimiser["flag_imaging"]:
        
        # initialisation
        with torch.cuda.nvtx.range("Initialisation"):
            optimiser.initialisation()
        
        #! DEBUG: run measurement operator and adjoint to check correctness
        with torch.cuda.nvtx.range("Adjoint_Operator_Test"):
            from src.mrop_ri_measurement_operator.test_meas_op import test_adjoint_op
            test_adjoint_op(meas_op, param_measop["img_size"], param_measop["dtype"])
        
        # run imaging loop
        with torch.cuda.nvtx.range("Run_Optimiser"):
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

        # Get peak memory active (tensors currently in memory)
        max_allocated = torch.cuda.max_memory_allocated()

        # Get peak memory reserved (total cache memory allocated from the driver)
        max_reserved = torch.cuda.max_memory_reserved()

        # Convert bytes to Megabytes (MB) where \(1\text{ MB} = 1024^2\text{ bytes}\)
        print(f"Max GPU memory allocated: {max_allocated / (1024 ** 2):.2f} MB")
        print(f"Max GPU memory reserved:  {max_reserved / (1024 ** 2):.2f} MB")