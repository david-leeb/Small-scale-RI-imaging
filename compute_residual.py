import numpy as np
import torch
from astropy.io import fits
import json
import sys

sys.path.insert(0, "/mnt/pvc/diss/Small-scale-RI-imaging-mrop")

from src.utils.io_combined import load_dataset
from src.utils import set_imaging_params_ri
from run_imager_combined import parsing_arguments, parsing_parameters
from src.utils.wstacking import compute_w_stacks, compute_single_stack, compute_global_w_stacking
from src.ri_measurement_operator.pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft
from src.utils.gpu_utils import mem, send_to_devices
from src.mrop_ri_measurement_operator.src.utils.solve_epsilon_new import solve_epsilon_same_aa
from src.mrop_ri_measurement_operator import weighting_correction

@torch.no_grad()
def compute_classical_residual(meas_op_classical, y, model_image_path, save_path, normalised_save_path = None):
    device = meas_op_classical.get_device()
    op_dtype = torch.complex128
    img_dtype = torch.float64
    
    x_mrop = torch.from_numpy(fits.getdata(model_image_path).astype(np.float64))
    x_mrop = x_mrop.to(device=device, dtype=img_dtype).unsqueeze(0).unsqueeze(0)

    dirty = meas_op_classical.adjoint_op(y.to(device=device, dtype=op_dtype))
    model_bp = meas_op_classical.adjoint_op(meas_op_classical.forward_op(x_mrop))
    residual = (dirty - model_bp).squeeze().cpu().to(img_dtype).numpy()

    fits.writeto(save_path, residual, overwrite=True)

    if normalised_save_path is not None:
        psf_peak = meas_op_classical.get_psf().max().item()
        fits.writeto(normalised_save_path, residual / psf_peak, overwrite=True)

    return residual


if __name__ == "__main__":

    input_args = parsing_arguments()
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    if param_general.get("verbose", True):
        print("Input parameters", flush=True)
        print(json.dumps(param_general, indent=4), flush=True)
    
    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)
    
    device = param_measop["device"]
    devices = None
    if device == torch.device("cuda"):
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        print("INFO: Detected", len(devices), "GPUs")
    
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
    
    data["y"] = data["y"] * data["nW"] * data["nWimag"]
    
    data = compute_global_w_stacking(data, param_measop)
    data = send_to_devices(data, devices)
    
    param_measop["w_stacking"] = True
    param_measop["reduce_memory_usage"] = False
    if param_measop["w_stacking"]:
        w_stack_data_list = compute_w_stacks(data, param_measop, devices)
    else:
        w_stack_data_list = compute_single_stack(data, param_measop, devices)
    
    from src.mrop_ri_measurement_operator import create_meas_op_ROP
    nufft_op = create_meas_op_ROP(MeasOpPytorchFinufft)
        
    meas_op_classical = nufft_op(
        img_size=param_measop["img_size"],
        w_stack_data=w_stack_data_list,
        num_chs=data["nFreqs"],
        use_ROP=False,
        devices=devices,
        ROP_param=None,
        ant1=data["ant1"],
        ant2=data["ant2"],
        batches=data["batches"],
        natural_weight_dev=data["nW_dev"],
        image_weight_dev=data["nWimag_dev"],
        device=devices[0],
        dtype=param_measop["dtype"],
        real_flag=True,
    )
    
    y_full = meas_op_classical.prepare_or_compress_data(data["y_dev"])
        
    residual = compute_classical_residual(
        meas_op_classical=meas_op_classical,
        y=y_full,
        model_image_path="results/meerkat_debug_new/meerkat/uSARA_heuRegScale_0.65_chs_220_N_ratio_1.0_epsilon_n_2.0_model_image.fits",
        save_path="results/uSARA_heuRegScale_0.65_chs_220_N_ratio_1.0_epsilon_n_2.0_residual_dirty_image.fits",
        normalised_save_path="results/uSARA_heuRegScale_0.65_chs_220_N_ratio_1.0_epsilon_n_2.0_normalised_residual_dirty_image.fits",
    )