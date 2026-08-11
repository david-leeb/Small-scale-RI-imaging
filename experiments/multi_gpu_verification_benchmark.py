"""
Usage:
    python validate_multi_gpu.py --config path/to/your_config.yaml
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from astropy.io import fits

RESULT_ROOT = "results/gpu_validation"


def run_worker() -> None:
    """Runs inside the subprocess -- identical to your real pipeline;
    result_path/file_prefix are overridden via env vars set by
    run_config() below so the two runs don't overwrite each other."""
    from src.utils import set_imaging_params_ri
    from run_imager_combined import parsing_arguments, parsing_parameters
    from src.imager_combined import imager

    input_args = parsing_arguments()  # sees only "--config <path>", same as a normal run
    param_general = parsing_parameters(input_args.config, input_param=input_args)
    param_measop, param_proxop, param_optimiser = set_imaging_params_ri(param_general)

    param_optimiser["result_path"] = os.environ["VALIDATE_RESULT_PATH"]
    param_optimiser["file_prefix"] = os.environ["VALIDATE_FILE_PREFIX"]
    os.makedirs(param_optimiser["result_path"], exist_ok=True)

    imager(param_optimiser, param_measop, param_proxop)


def clear_directory(path: str) -> None:
    """Removes `path` (and everything in it) if it exists, then recreates
    it empty. Run before each subprocess so find_output_file()'s
    pattern-based search can't accidentally pick up a stale file left
    over from a previous run in the same directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_output_file(result_path: str, tag: str, keyword: str) -> str:
    """Finds the file containing `keyword` (e.g. "model_image") whose
    name starts with `tag`, instead of assuming an exact filename --
    imager() can append extra text to file_prefix at runtime (e.g. the
    "P_..._MB_..._MK_..._" ROP-parameter suffix), so the final filename
    isn't fully predictable ahead of time."""
    pattern = os.path.join(result_path, f"{tag}*{keyword}*.fits")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file found matching {pattern!r} -- check the run actually "
            f"completed and wrote into {result_path!r}."
        )
    if len(matches) > 1:
        print(f"WARNING: multiple matches for {pattern!r}, using the first: {matches}")
    return matches[0]


def run_config(tag: str, config_path: str, cuda_visible_devices: str = None):
    """Launches run_worker() in a fresh subprocess with the given GPU
    visibility. cuda_visible_devices=None leaves it unset, i.e. whatever
    GPUs are normally visible -- your real multi-GPU configuration.

    Clears result_path first -- this deletes prior contents of
    results/gpu_validation/<tag> specifically (not your main results
    tree), so change RESULT_ROOT if that path means something else on
    your machine."""
    result_path = os.path.join(RESULT_ROOT, tag)
    file_prefix = f"{tag}_"

    clear_directory(result_path)

    env = os.environ.copy()
    env["VALIDATE_WORKER"] = "1"
    env["VALIDATE_RESULT_PATH"] = result_path
    env["VALIDATE_FILE_PREFIX"] = file_prefix
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print(f"\n=== Running '{tag}' (CUDA_VISIBLE_DEVICES={cuda_visible_devices or 'default'}) "
          f"-> {result_path} ===")
    subprocess.run([sys.executable, __file__, "--config", config_path], env=env, check=True)
    return result_path, tag


def compare_model_images(path_a: str, tag_a: str, path_b: str, tag_b: str, tol: float = 1e-5):
    """tol is deliberately not tiny: single- vs multi-GPU sums the same
    per-device partial results in a different order (see
    MeasOpNUFFTRMOP.adjoint_op's parts[0] + parts[1:] accumulation), so a
    correct multi-GPU run is expected to differ from single-GPU at the
    level of floating-point non-associativity, not be bit-identical."""
    file_a = find_output_file(path_a, tag_a, "model_image")
    file_b = find_output_file(path_b, tag_b, "model_image")
    print(f"\ncomparing:\n  A: {file_a}\n  B: {file_b}")

    img_a = fits.getdata(file_a).astype(np.float64)
    img_b = fits.getdata(file_b).astype(np.float64)

    diff = img_a - img_b
    max_abs = np.max(np.abs(diff))
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(img_a)

    print("\n=== Single-GPU vs Multi-GPU comparison ===")
    print(f"max|diff|    = {max_abs:.6e}")
    print(f"relative L2  = {rel_l2:.6e}")
    print(f"single std   = {img_a.std():.6e}")
    print(f"multi  std   = {img_b.std():.6e}")
    if rel_l2 < tol:
        print(f"✅ match (relative L2 < {tol:.0e}).")
    else:
        print(f"❌ differ (relative L2 >= {tol:.0e}).")
    return max_abs, rel_l2


if __name__ == "__main__":
    if os.environ.get("VALIDATE_WORKER") == "1":
        run_worker()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        top_args = parser.parse_args()

        single_path, single_tag = run_config("single_gpu", top_args.config, cuda_visible_devices="0")
        multi_path, multi_tag = run_config("multi_gpu", top_args.config, cuda_visible_devices=None)

        compare_model_images(single_path, single_tag, multi_path, multi_tag)