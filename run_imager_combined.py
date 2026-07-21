"""
Prepare input variables for imager
"""

from typing import Any, Dict, Optional
import argparse
import json

from src.imager_combined import imager
from src.utils import set_imaging_params_ri


def parsing_arguments() -> argparse.Namespace:
    """
    Parses input arguments from the command line.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="imaging main parameters")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/example.json",
        help="path of the configuration file",
    )
    parser.add_argument(
        "--src_name", type=str, default=None, help="Initial reg parameter"
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="path of the datafile"
    )
    parser.add_argument(
        "--result_path", type=str, default=None, help="path of the result folder"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["airi", "cairi", "usara"],
        help="imaging algorithm",
    )
    parser.add_argument(
        "--im_dim_x", type=int, default=None, help="result horizental image size"
    )
    parser.add_argument(
        "--im_dim_y", type=int, default=None, help="result vertical image size"
    )
    parser.add_argument(
        "--dnn_shelf_path",
        type=str,
        default=None,
        help="path of the denoiser shelf configuration file",
    )
    parser.add_argument(
        "--im_pixel_size",
        type=float,
        default=None,
        help="pixel size of the estimated image in the unit of arcsec",
    )
    parser.add_argument(
        "--superresolution",
        type=float,
        default=None,
        help="imaging super-resolution factor",
    )
    parser.add_argument(
        "--groundtruth",
        type=str,
        default=None,
        help="path of the groundtruth image in fits format",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="identification number of the current task",
    )
    return parser.parse_args()


def parsing_parameters(
    config_file: str, input_param: Optional[argparse.Namespace] = None
) -> Dict[str, Any]:
    """
    Mixes input parameters from the command line and config file.

    Args:
        config_file (str): Path of the configuration file.
        input_param (argparse.Namespace, optional): Parsed command line arguments.

    Returns:
        Dict[str, Any]: Dictionary of parameters.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    json_dict = {}
    for json_data_sub in json_data:
        for k, v in json_data_sub.items():
            if k != "__comment":
                json_dict[k] = v

    # load main parameters
    param = json_dict["main"]
    if input_param is not None:
        dict_input_param = vars(input_param)
        for k, v in dict_input_param.items():
            if k in param.keys() and v is not None:
                param[k] = v

    # load general parameters
    list_keys = ["flag", "weighting", "computing", "nufft", "ROP"]
    if param["algorithm"] == "airi":
        list_keys.extend(["airi", "airi_default"])
    elif param["algorithm"] == "cairi":
        list_keys.extend(["cairi", "cairi_default"])
    elif param["algorithm"] == "usara":
        list_keys.extend(["usara", "usara_default"])
    else:
        raise NotImplementedError(f"Algorithm {param['algorithm']} not found\n")
    for key in list_keys:
        if key == "ROP" and key not in json_dict:
            continue
        for k, v in json_dict[key].items():
            if "_comment_" not in k:
                param[k] = v
    if input_param is not None and input_param.dnn_shelf_path is not None:
        param["dnn_shelf_path"] = input_param.dnn_shelf_path

    return param


def print_dict(curr_dict: Dict[str, Any], flush: bool = True) -> None:
    """
    Prints items in a dictionary.

    Args:
        curr_dict (Dict[str, Any]): Dictionary to print.
        flush (bool, optional): Whether to flush the output buffer. Defaults to True.
    """
    for k, v in curr_dict.items():
        print(f"    {k}: {v}", flush=flush)


if __name__ == "__main__":
    input_args = parsing_arguments()
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    if param_general.get("verbose", True):
        print("Input parameters", flush=True)
        print(json.dumps(param_general, indent=4), flush=True)

    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)
    if param_optimiser["verbose"]:
        print(
            "________________________________________________________________\n",
            flush=True,
        )
        print("Imaging parameters:", flush=True)
        print("  param_optimiser:", flush=True)
        print_dict(param_optimiser, flush=True)
        print("  param_measop:", flush=True)
        print_dict(param_measop, flush=True)
        print("  param_proxop:", flush=True)
        print_dict(param_proxop, flush=True)
        if param_measop["use_ROP"]:
            print("  ROP_param:", flush=True)
            print_dict(param_measop["ROP_param"], flush=True)
        print(
            "________________________________________________________________\n",
            flush=True,
        )
        

    imager(param_optimiser, param_measop, param_proxop)
