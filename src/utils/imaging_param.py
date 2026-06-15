"""
Generate parameter dictionaries for measuemt operator, prox operator and optimiser
from input parameter dictionary.
"""

from typing import Any, Dict, Tuple
import os
import platform
import torch
import psutil


def set_imaging_params_ri(
    param_general: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Set parameters of measuemt operator, prox operator and optimiser for
    small scale RI imaging task.

    Args:
        param_general (dict): A dictionary containing general parameters.

    Returns:
        tuple: A tuple containing three dictionaries:
            - param_measop (dict): A dictionary containing parameters for the measurement operator.
            - param_proxop (dict): A dictionary containing parameters for the prox operator.
            - param_optimiser (dict): A dictionary containing parameters for the optimiser.
    """
    # initialisation
    param_measop = {}
    param_proxop = {}
    param_optimiser = {}

    # parameters shared by all algorithms
    param_optimiser["algorithm"] = param_general["algorithm"]
    param_optimiser["data_file"] = param_general["data_file"]
    param_optimiser["use_s3"] = param_general.get("use_s3", False)
    param_optimiser["s3_bucket_name"] = param_general.get("s3_bucket_name", None)
    param_optimiser["tmp_dir"] = param_general.get("tmp_dir", None)
    param_optimiser["src_name"] = param_general.get("src_name", None)
    param_optimiser["regen_obs_nfreqs"] = param_general.get("regen_obs_nfreqs", False)
    # set default values
    param_optimiser["flag_imaging"] = param_general.get("flag_imaging", True)
    param_proxop["verbose"] = param_general.get("verbose", True)
    param_optimiser["verbose"] = param_general.get("verbose", True)

    # image size
    if (
        param_general.get("im_dim_x", None)
        and param_general["im_dim_y"] >= 1
        and param_general.get("im_dim_y", None)
        and param_general["im_dim_y"] >= 1
    ):
        param_measop["img_size"] = (
            int(param_general["im_dim_x"]),
            int(param_general["im_dim_y"]),
        )
    else:
        param_measop["img_size"] = (512, 512)
    
    if param_general.get("dl_shift", None):
        param_measop["dl_shift"] = int(param_general["dl_shift"])
    else:
        param_measop["dl_shift"] = 0
    
    if param_general.get("dm_shift", None):
        param_measop["dm_shift"] = int(param_general["dm_shift"])
    else:
        param_measop["dm_shift"] = 0
        
    param_measop["nfreqs"] = param_general.get("nfreqs", None)
    param_measop["freq_num"] = param_general.get("freq_num", None)

    # image pixel size
    if param_general.get("im_pixel_size", None) and param_general["im_pixel_size"] > 0.0:
        param_measop["im_pixel_size"] = param_general["im_pixel_size"]
    else:
        param_measop["im_pixel_size"] = None
    # super-resolution factor
    if param_general.get("superresolution", None) and param_general["superresolution"] >= 1.0:
        param_measop["superresolution"] = float(param_general["superresolution"])
    else:
        param_measop["superresolution"] = 1.0

    # weighting
    param_measop["flag_data_weighting"] = param_general.get("flag_data_weighting", True)
    param_measop["weight_load"] = param_general.get("weight_load", True)
    if param_general.get("weight_type", None) and param_general["weight_type"] in [
        "briggs",
        "uniform",
    ]:
        param_measop["weight_type"] = param_general["weight_type"]
    else:
        param_measop["weight_type"] = "briggs"
    if param_general.get("weight_robustness", None):
        param_measop["weight_robustness"] = float(param_general["weight_robustness"])
    else:
        param_measop["weight_robustness"] = 0.0
    if param_general.get("weight_gridsize", None) and param_general.get("weight_gridsize", None) > 0.0:
        param_measop["weight_gridsize"] = float(param_general["weight_gridsize"])
    else:
        param_measop["weight_gridsize"] = 2

    # nufft settings
    # package
    if param_general.get("nufft_package", None) and param_general["nufft_package"] in [
        "finufft",
        "pynufft",
        "tkbnufft",
    ]:
        param_measop["nufft_package"] = param_general["nufft_package"]
    else:
        param_measop["nufft_package"] = "finufft"

    # oversampling factor
    if (
        param_general.get("nufft_oversampling_factor", None)
        and param_general["nufft_oversampling_factor"][0] >= 1.0
        and param_general["nufft_oversampling_factor"][1] >= 1.0
    ):
        param_measop["nufft_grid_size"] = (
            int(param_general["nufft_oversampling_factor"][0] * param_measop["img_size"][0]),
            int(param_general["nufft_oversampling_factor"][1] * param_measop["img_size"][1]),
        )
    else:
        param_measop["nufft_grid_size"] = (
            int(2.0 * param_measop["img_size"][0]),
            int(2.0 * param_measop["img_size"][1]),
        )
    # KB kernel dimension
    if param_general.get("nufft_kb_kernel_dim", None) and param_general["nufft_kb_kernel_dim"] >= 1.0:
        param_measop["nufft_kb_kernel_dim"] = int(param_general["nufft_kb_kernel_dim"])
    else:
        param_measop["nufft_kb_kernel_dim"] = 7
    # nufft mode
    if param_general.get("nufft_mode", None) and param_general["nufft_mode"] in [
        "table",
        "matrix",
    ]:
        param_measop["nufft_mode"] = param_general["nufft_mode"]
    else:
        param_measop["nufft_mode"] = "table"

    # BDA
    param_measop["use_BDA"] = param_general.get("use_BDA", False)
    param_measop["max_avg_time"] = param_general.get("BDA_max_avg_time", None)
    param_measop["max_avg_freq"] = param_general.get("BDA_max_avg_freq", None)
    param_measop["smearing_limit"] = param_general.get("BDA_smearing_limit", None)

    # MROP
    param_measop["ROP_type"] = param_general.get("ROP_type", None)
    param_measop["use_ROP"] = False
    if param_measop["ROP_type"] in ["none", None]:
        param_measop["ROP_type"] = None
        if param_measop["use_BDA"]:
            param_measop["ROP_param"] = {"Q": param_general.get("ROP_Q", None)}
        else:
            param_measop["ROP_param"] = None
    elif param_measop["ROP_type"] in ["MROP", "CROP", "MROP_gaussian"]:
        param_measop["use_ROP"] = True
        if "ROP_seed" not in param_general:
            try:
                ROP_seed = int(param_optimiser["data_file"].split("_id_")[1].split("_")[0])
            except:
                ROP_seed = 1
        param_measop["ROP_param"] = {
            "ROP_type": param_measop["ROP_type"],
            "P": param_general["ROP_P"],
            "M": param_general["ROP_M"],
            "N_ratio": param_general.get("ROP_N_ratio", 1.0),
            "epsilon_n": param_general.get("ROP_epsilon_n", 1.0),
            "Q": param_general.get("ROP_Q", None),
            "B": param_general.get("ROP_B", None),
            "rv_type": param_general["ROP_rv_type"],
            "ROP_seed": ROP_seed,
            "ROP_batchwise": param_general.get("ROP_batchwise", False),
            "ROP_batch_step": param_general.get("ROP_batch_step", None),
            "weight_type": param_general.get("weight_type", None),
            "ROP_vmap": param_general.get("ROP_vmap", False),
            "ROP_vmap_chunk_size": param_general.get("ROP_vmap_chunk_size", None),
            "freq_mod": param_general.get("freq_mod", None),
            "same_ab": param_general.get("same_ab", False),
            "same_ab_all": param_general.get("same_ab_all", False),
            "same_ab_B": param_general.get("same_ab_B", False),
            "same_seed": param_general.get("ROP_same_seed", 0),
        }

        assert not (
            param_measop["ROP_param"]["same_ab_all"] and param_measop["ROP_param"]["same_ab_B"]
        ), "same_ab_all and same_ab_B cannot be both True."

        if param_measop["ROP_param"]["same_seed"] != 0:
            param_measop["ROP_param"]["ROP_seed"] = param_measop["ROP_param"]["same_seed"]
        if param_measop["use_ROP"]:
            assert not param_general[
                "approx_meas_op"
            ], "approximate measurement operator is currently not supported for MROP/CROP."
            assert param_optimiser["algorithm"] in [
                "usara",
                "airi",
            ], "MROP/CROP is currently only supported for uSARA and AIRI."
            # assert (
            #     param_measop["weight_type"] == "uniform"
            # ), "MROP/CROP is currently only supported for uniform weighting."
    else:
        raise ValueError(
            f"argument ROP_type {param_measop['ROP_type']} not supported. " "Please use 'MROP' or 'CROP'."
        )

    # computing resources
    # number of threads
    if platform.system() != "Darwin":  # not on macOS
        avail_cpus = len(psutil.Process().cpu_affinity())
        if param_general.get("ncpus", None) and param_general["ncpus"] >= 1:
            request_cpus = min(avail_cpus, int(param_general["ncpus"]))
            torch.set_num_threads(request_cpus)
            if param_optimiser["verbose"]:
                print(f"INFO: avaiable cpus {avail_cpus}, request cpus {request_cpus}")
        else:
            torch.set_num_threads(avail_cpus)
            if param_optimiser["verbose"]:
                print(f"INFO: avaiable cpus {avail_cpus}")
    # devices
    list_devices = []
    if torch.cuda.is_available():
        list_devices.append("cuda")
    if torch.backends.mps.is_available():
        list_devices.append("mps")
    list_devices.append("cpu")
    if param_general.get("meas_device", None) and param_general["meas_device"] in list_devices:
        param_measop["device"] = torch.device(param_general["meas_device"])
    else:
        param_measop["device"] = (
            torch.device(list_devices[0]) if list_devices[0] != "mps" else torch.device(list_devices[1])
        )
    if param_general.get("prox_device", None) and param_general["prox_device"] in list_devices:
        param_proxop["device"] = torch.device(param_general["prox_device"])
    else:
        param_proxop["device"] = torch.device(list_devices[0])

    # max number of iterations
    if param_general.get("im_max_itr", None) and param_general["im_max_itr"] > 0:
        param_optimiser["im_max_itr"] = int(param_general["im_max_itr"])
    else:
        param_optimiser["im_max_itr"] = 2000
    # min number of iterations
    if param_general.get("im_min_itr", None) and param_general["im_min_itr"] > 0:
        param_optimiser["im_min_itr"] = int(param_general["im_min_itr"])
    else:
        param_optimiser["im_min_itr"] = 200
    # image variation tolerance
    if param_general.get("im_var_tol", None) and param_general["im_var_tol"] > 0:
        param_optimiser["im_var_tol"] = float(param_general["im_var_tol"])
    else:
        param_optimiser["im_var_tol"] = 1e-5

    # data type
    if param_general.get("meas_dtype", None) and param_general.get("meas_dtype") in [
        "float",
        "float32",
        "single",
    ]:
        param_measop["dtype"] = torch.float
    else:
        param_measop["dtype"] = torch.double
    if param_general.get("prox_dtype", None) and param_general.get("prox_dtype") in [
        "double",
        "float64",
    ]:
        param_proxop["dtype"] = torch.double
    else:
        param_proxop["dtype"] = torch.float

    # interval for saveing intermediate results
    if param_general.get("itr_save", None) and param_general["itr_save"] >= 1:
        param_optimiser["itr_save"] = int(param_general["itr_save"])
    else:
        param_optimiser["itr_save"] = param_optimiser["im_max_itr"] + 1

    if param_general.get("groundtruth", None):
        if not param_optimiser["use_s3"] and os.path.isfile(param_general["groundtruth"]):
            param_optimiser["groundtruth"] = param_general["groundtruth"]
        elif param_optimiser["use_s3"]:
            param_optimiser["groundtruth"] = param_general["groundtruth"]
    else:
        param_optimiser["groundtruth"] = None

    # using approximate measurement operator
    param_optimiser["approx_meas_op"] = False

    # parameters shared by AIRI algorithms
    if param_optimiser["algorithm"] in ["airi", "cairi"]:
        # heuristic noise scale
        if param_general.get("heu_noise_scale", None) and param_general["heu_noise_scale"] > 0:
            param_optimiser["heu_noise_scale"] = float(param_general["heu_noise_scale"])
        else:
            param_optimiser["heu_noise_scale"] = 1.0

        param_optimiser["new_heu"] = param_general.get("new_heu", False)

        # AIRI shelf path
        if param_general.get("dnn_shelf_path", None) and isinstance(param_general["dnn_shelf_path"], str):
            param_proxop["dnn_shelf_path"] = param_general["dnn_shelf_path"]

        # Adaptive network selection scheme
        # estimated image peak value
        if param_general.get("im_peak_est", None) and param_general["im_peak_est"] > 0:
            param_optimiser["im_peak_est"] = float(param_general["im_peak_est"])
        else:
            param_optimiser["im_peak_est"] = None
        param_optimiser["dnn_adaptive_peak"] = param_general.get("dnn_adaptive_peak", True)
        if (
            param_general.get("dnn_adaptive_peak_tol_max", None)
            and param_general["dnn_adaptive_peak_tol_max"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_max"] = float(param_general["dnn_adaptive_peak_tol_max"])
        else:
            param_optimiser["dnn_adaptive_peak_tol_max"] = 0.1
        if (
            param_general.get("dnn_adaptive_peak_tol_min", None)
            and param_general["dnn_adaptive_peak_tol_min"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_min"] = float(param_general["dnn_adaptive_peak_tol_min"])
        else:
            param_optimiser["dnn_adaptive_peak_tol_min"] = 1e-3
        if (
            param_general.get("dnn_adaptive_peak_tol_step", None)
            and param_general["dnn_adaptive_peak_tol_step"] > 0
        ):
            param_optimiser["dnn_adaptive_peak_tol_step"] = float(param_general["dnn_adaptive_peak_tol_step"])
        else:
            param_optimiser["dnn_adaptive_peak_tol_step"] = 0.1

        # random input image flip & 90-degree rotation
        param_proxop["dnn_apply_transform"] = param_general.get("dnn_apply_transform", True)

        # specific parameters for AIRI
        if param_optimiser["algorithm"] == "airi":
            param_optimiser["approx_meas_op"] = param_general.get("approx_meas_op", False)

    # parameters shared by SARA algorithms
    elif param_optimiser["algorithm"] in ["usara"]:
        # heuristic regularisation parameter scale
        if param_general.get("heu_reg_param_scale", None) and param_general["heu_reg_param_scale"] > 0:
            param_optimiser["heu_reg_param_scale"] = float(param_general["heu_reg_param_scale"])
        else:
            param_optimiser["heu_reg_param_scale"] = 1.0

        param_optimiser["new_heu"] = param_general.get("new_heu", False)

        param_optimiser["reweighting_save"] = param_general.get("reweighting_save", False)
        if param_general.get("im_max_outer_itr", None) and param_general["im_max_outer_itr"] >= 1:
            param_optimiser["im_max_outer_itr"] = int(param_general["im_max_outer_itr"])
        else:
            param_optimiser["im_max_outer_itr"] = 20
        if param_general.get("im_var_outer_tol", None) and param_general["im_var_outer_tol"] > 0:
            param_optimiser["im_var_outer_tol"] = float(param_general["im_var_outer_tol"])
        else:
            param_optimiser["im_var_outer_tol"] = 1e-3

        # specific parameters for uSARA
        if param_optimiser["algorithm"] == "usara":
            param_optimiser["approx_meas_op"] = param_general.get("approx_meas_op", False)
            param_proxop["use_optimized"] = param_general.get("use_optimized", True)
            
    param_measop["meas_op_norm"] = param_general.get("meas_op_norm", None)
    param_measop["heu_corr_factor"] = param_general.get("heu_corr_factor", None)
    param_measop["wcentres_mat_file"] = param_general.get("wcentres_mat_file", None)   

    # parameters shared by constrained algorithms
    if param_optimiser["algorithm"] in ["cairi"]:
        param_optimiser["precond_flag"] = param_general.get("precond_flag", True)

    # set path for saving results
    if not param_general.get("src_name", None):
        param_general["src_name"] = os.path.splitext(os.path.basename(param_general["data_file"]))[0]
    if not param_general.get("result_path", None):
        param_general["result_path"] = os.path.join(".", "results")
    param_optimiser["result_path"] = os.path.join(param_general["result_path"], param_general["src_name"])
    os.makedirs(param_optimiser["result_path"], exist_ok=True)

    file_prefix = ""
    if param_optimiser["algorithm"] == "airi":
        file_prefix = "AIRI_heuScale_" + str(param_optimiser["heu_noise_scale"]) + "_"
    elif param_optimiser["algorithm"] == "cairi":
        file_prefix = "cAIRI_heuScale_" + str(param_optimiser["heu_noise_scale"]) + "_"
    elif param_optimiser["algorithm"] == "usara":
        file_prefix = "uSARA_heuRegScale_" + str(param_optimiser["heu_reg_param_scale"]) + "_"
    if param_general.get("run_id", None):
        file_prefix += "runID_" + str(param_general["run_id"]) + "_"
    param_optimiser["file_prefix"] = file_prefix

    return (
        dict(sorted(param_measop.items())),
        dict(sorted(param_proxop.items())),
        dict(sorted(param_optimiser.items())),
    )
