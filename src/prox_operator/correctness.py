import torch
import numpy as np
import sys
import copy
import ptwt

sys.path.insert(0, "/mnt/pvc/diss/Small-scale-RI-imaging-mrop")

from src.utils.imaging_param import set_imaging_params_ri
from run_imager_3c273 import parsing_arguments, parsing_parameters
from src.prox_operator import ProxOpSARAPos, ProxOpSARAPos_original
from src.prox_operator.db_wavelets import CompiledWaveletBank

def run_verification():    
    input_args = parsing_arguments()
    
    input_args.config = "../../config/3C273_central_64chs_512_profile.json"
    
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    param_general["data_file"] = "../../../data/273-X08-dmog/msSpecs.mat"
    param_general["results_path"] = "../../../results/3C273_64ch_profile"
    
    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)
    
    prox_op_og = ProxOpSARAPos_original(
                param_measop["img_size"],
                device=param_proxop["device"],
                dtype=param_proxop["dtype"],
                verbose=True,
                max_iter=20
            )
    
    prox_op_opt = ProxOpSARAPos(
                param_measop["img_size"],
                device=param_proxop["device"],
                dtype=param_proxop["dtype"],
                verbose=True,
                max_iter=20
            )
        
    print("\n--- Test ---")
    torch.manual_seed(1337)
    
    img_og = torch.randn(param_measop["img_size"], dtype=torch.double, device=param_measop["device"]).unsqueeze(0)
    img_opt = img_og.clone()
    
    prox_op_og.update(img_og, initialisation=True)
    prox_op_opt.update(img_opt, initialisation=True)
    
    out_og = prox_op_og(img_og)
    out_opt = prox_op_opt(img_opt)
    
    prox_op_og.update(out_og)
    prox_op_opt.update(out_opt)
    
    diff_max = torch.max(torch.abs(out_og - out_opt)).item()
    diff_rel = torch.norm(out_og - out_opt) / torch.norm(out_og)
    
    print(f"Max Absolute Error: {diff_max:.10e}")
    print(f"Relative Error:     {diff_rel.item():.10e}")
    if diff_rel < 1e-5:
        print("✅ prox matches.")
    else:
        print("❌ prox differs.")

if __name__ == "__main__":
    run_verification()