import torch
import sys

sys.path.insert(0, "/mnt/pvc/diss/Small-scale-RI-imaging-mrop")

from src.prox_operator.prox_op_sara_original import ProxOpSARAPos_original as ProxOpSARAPos_original
from src.prox_operator.prox_op_sara import ProxOpSARAPos as ProxOpSARAPos
from src.prox_operator.prox_op_sara_optimized_ptwt import ProxOpSARAPos as ProxOpSARAPos_optimized

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

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

def _time_per_sub_iteration(prox_op, x: torch.Tensor, n_repeats: int = 20, n_warmup: int = 3):
    """
    Mean/std time (ms) for ONE sub-iteration of prox_op.
 
    Times prox_op(x) end-to-end per call and divides by prox_op._max_iter
    -- exact, not an estimate, because obj_tol=0 means the early-stop
    check (obj_rel_var < obj_tol) can never fire, so every call runs
    exactly max_iter sub-iterations.
 
    Warm-up calls happen first and are not timed -- for the CUDA-graph
    implementations this is where the one-time capture happens (3
    internal warm-up iterations + the capture itself), which would
    otherwise dominate a small-sample mean/std. update() is never called
    between timed repeats: it invalidates the graph (forces recapture on
    the next call), and __call__ without an intervening update() is the
    steady-state cost that matches how often each is actually called in
    practice (update() once per reweighting cycle, __call__ every inner
    iteration).
    """
    for _ in range(n_warmup):
        prox_op(x)
    torch.cuda.synchronize()
 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(n_repeats):
        start.record()
        prox_op(x)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end) / prox_op._max_iter)
 
    times_ms = torch.tensor(times_ms)
    return times_ms.mean().item(), times_ms.std().item()
 
 
def run_benchmark(n_repeats: int = 20, n_warmup: int = 5):
    img_size = (512, 512)
    device = torch.device("cuda")
    dtype = torch.float32
    obj_tol = 0  # every __call__ runs exactly max_iter sub-iterations
 
    implementations = {
        "ORIGINAL": ProxOpSARAPos_original,
        "OPT CUSTOM": ProxOpSARAPos,
        "OPT PTWT": ProxOpSARAPos_optimized,
    }
 
    print("\n--- Benchmark: time per sub-iteration ---")
    torch.manual_seed(1337)
    x = torch.randn(img_size, dtype=dtype, device=device).unsqueeze(0)
 
    results = {}
    for name, cls in implementations.items():
        prox_op = cls(
            img_size,
            device=device,
            dtype=dtype,
            verbose=False,  # avoid print() overhead polluting timing
            max_iter=20,
            obj_tol=obj_tol,
        )
        prox_op.update(x, initialisation=True)
 
        mean_ms, std_ms = _time_per_sub_iteration(
            prox_op, x, n_repeats=n_repeats, n_warmup=n_warmup
        )
        results[name] = mean_ms
        print(f"{name:11s}: {mean_ms:8.4f} ms/iter  (std {std_ms:.4f} ms, n={n_repeats})")
 
        # isolate each implementation's memory/timing from the next
        del prox_op
        torch.cuda.empty_cache()
 
    base_name = "ORIGINAL"
    print(f"\nSpeedup vs {base_name}:")
    for name, mean_ms in results.items():
        if name == base_name:
            continue
        print(f"  {name:11s}: {results[base_name] / mean_ms:.2f}x")
 
    return results

if __name__ == "__main__":
    run_verification()
    run_benchmark()