import torch
import sys

sys.path.insert(0, "/mnt/pvc/diss/Small-scale-RI-imaging-mrop")

from src.prox_operator.prox_op_sara_original import ProxOpSARAPos_original as ProxOpSARAPos_original
from src.prox_operator.prox_op_sara import ProxOpSARAPos as ProxOpSARAPos
from src.prox_operator.prox_op_sara_optimized_ptwt import ProxOpSARAPos as ProxOpSARAPos_optimized

def run_verification():        
    img_size = (2048, 2048)
    device = torch.device("cuda")
    dtype = torch.float32
    
    obj_tol = 0
    
    prox_op_og = ProxOpSARAPos_original(
                img_size,
                device=device,
                dtype=dtype,
                verbose=True,
                max_iter=20,
                obj_tol=obj_tol
            )
    
    prox_op_opt = ProxOpSARAPos(
                img_size,
                device=device,
                dtype=dtype,
                verbose=True,
                max_iter=20,
                obj_tol=obj_tol
            )
    
    prox_op_opt_new = ProxOpSARAPos_optimized(
                img_size,
                device=device,
                dtype=dtype,
                verbose=True,
                max_iter=20,
                obj_tol=obj_tol
            )
        
    print("\n--- Test ---")
    torch.manual_seed(1337)
    
    img_og = torch.randn(img_size, dtype=dtype, device=device).unsqueeze(0)
    img_opt = img_og.clone()
    img_opt_new = img_og.clone()
    
    prox_op_og.update(img_og, initialisation=True)
    prox_op_opt.update(img_opt, initialisation=True)
    prox_op_opt_new.update(img_opt_new, initialisation=True)
    
    for i in range(5):    
        out_og = prox_op_og(img_og)
        out_opt = prox_op_opt(img_opt)
        out_opt_new = prox_op_opt_new(img_opt_new)
        
        prox_op_og.update(out_og)
        prox_op_opt.update(out_opt)
        prox_op_opt_new.update(out_opt_new)
    
        diff_max_opt = torch.max(torch.abs(out_og - out_opt)).item()
        diff_rel_opt = torch.norm(out_og - out_opt) / torch.norm(out_og)
        
        diff_max_new = torch.max(torch.abs(out_og - out_opt_new)).item()
        diff_rel_new = torch.norm(out_og - out_opt_new) / torch.norm(out_og)
        
        print(f"OPT CUSTOM")
        print(f"Max Absolute Error: {diff_max_opt:.10e}")
        print(f"Relative Error:     {diff_rel_opt.item():.10e}")
        if diff_rel_opt < 1e-5:
            print("✅ prox matches.")
        else:
            print("❌ prox differs.")
        
        print(f"OPT PTWT")
        print(f"Max Absolute Error: {diff_max_new:.10e}")
        print(f"Relative Error:     {diff_rel_new.item():.10e}")
        if diff_rel_new < 1e-5:
            print("✅ prox matches.")
        else:
            print("❌ prox differs.")

if __name__ == "__main__":
    run_verification()