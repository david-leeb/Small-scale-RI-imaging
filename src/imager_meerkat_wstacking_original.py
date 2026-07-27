"""
Prepare proper measurement operator, prior and algorithm for imaging task
"""

from typing import Dict
import torch
import numpy as np
from astropy.io import fits
import gc

from .prox_operator import ProxOpAIRI, ProxOpElipse, ProxOpSARAPos
from .optimiser import FBAIRI, PDAIRI, FBSARAMEERKAT
from .utils import gen_imaging_weight

# from .ri_measurement_operator.pysrc.utils.io import load_data_to_tensor
# from .ri_measurement_operator.pysrc.utils.io_new import load_data_to_tensor
from .utils.io_meerkat_original import load_real_data_to_tensor
from .ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft_wstacking import (
    MeasOpPytorchFinufftWStacking,
)


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

    # data = load_data_to_tensor(
    #     param_optimiser["data_file"],
    #     super_resolution=param_measop["superresolution"],
    #     image_pixel_size=param_measop["im_pixel_size"],
    #     data_weighting=param_measop["flag_data_weighting"],
    #     load_weight=param_measop["weight_load"],
    #     img_size=param_measop["img_size"],
    #     uv_unit="radians",
    #     weight_type=param_measop["weight_type"],
    #     # weight_gridsize=param_measop["weight_gridsize"],
    #     weight_robustness=param_measop["weight_robustness"],
    #     dtype=param_measop["dtype"],
    #     device=param_measop["device"],
    #     verbose=param_optimiser["verbose"],
    # )

    data = load_real_data_to_tensor(
        data_path=param_optimiser["data_file"],
        # super_resolution=args.super_resolution,
        image_pixel_size=param_measop["im_pixel_size"],
        img_size=param_measop["img_size"],
        start_ch=95,
        end_ch=125,
        data_weighting=param_measop["flag_data_weighting"],
        weight_type=param_measop["weight_type"],
        weight_robustness=param_measop["weight_robustness"],
    )

    fov_radians = (
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][0] * np.pi / 180,
        (data["image_pixel_size"] / 3600) * param_measop["img_size"][1] * np.pi / 180,
    )

    num_wstacks = np.ceil(
        data["w"].numpy(force=True).max() * 2 * np.pi * (1 - np.sqrt(1 - 2 * np.sin(fov_radians[0] / 2) ** 2))
    )
    w_max = data["w"].numpy(force=True).max()
    num_wstacks = int(max(num_wstacks, torch.cuda.device_count()))
    print(
        f"INFO: FOV in radians: {fov_radians}, max w value: {w_max:.4f}, number of w-stacks determined to be {num_wstacks} based on the FOV and max w value.",
        flush=True,
    )
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    # meas_op = None
    meas_op = MeasOpPytorchFinufftWStacking(
        u=data["u"],
        v=data["v"],
        w=data["w"],
        image_pixel_size=param_measop["im_pixel_size"],
        wcentres_mat_file=param_measop["wcentres_mat_file"],
        num_wstacks=num_wstacks,
        img_size=param_measop["img_size"],
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        real_flag=True,
        device=param_measop["device"],
        device_list=[param_measop["device"]],
        dtype=param_measop["dtype"],
        kmeans_pkg="sklearn",
    )
    # .pin_memory().to(device=device, non_blocking=True).view(1, 1, -1)
    num_wstacks = meas_op._num_wstacks
    y = [
        data["y"][..., meas_op._w_stack_idx == i]
        .pin_memory()
        .to(meas_op._device[i], non_blocking=True)
        .view(1, 1, -1)
        for i in range(num_wstacks)
    ]

    nW = [
        data["nW"][..., meas_op._w_stack_idx == i]
        .pin_memory()
        # .to(meas_op._device[i], non_blocking=True)
        .view(1, 1, -1)
        for i in range(num_wstacks)
    ]

    nWimag = [
        data["nWimag"][..., meas_op._w_stack_idx == i]
        .pin_memory()
        # .to(meas_op._device[i], non_blocking=True)
        .view(1, 1, -1)
        for i in range(num_wstacks)
    ]

    torch.cuda.synchronize()
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

    del data["u"], data["v"], data["nW"], data["nWimag"], data["w"], data["y"]
    gc.collect()
    torch.cuda.empty_cache()

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
            nW=nW,
            nWimag=nWimag,
            heu_corr_factor=param_measop["heu_corr_factor"],
            meas_op_norm=param_measop["meas_op_norm"],
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

        optimiser = FBSARAMEERKAT(
            y,
            meas_op,
            prox_op_sara,
            nW=nW,
            nWimag=nWimag,
            heu_corr_factor=param_measop["heu_corr_factor"],
            meas_op_norm=param_measop["meas_op_norm"],
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
        optimiser.initialisation()
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
