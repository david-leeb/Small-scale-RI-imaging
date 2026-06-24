import time

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

import ptwt


def _to_jit_wavedec_2(data: torch.Tensor, wavelet) -> list[torch.Tensor]:
    """Ensure uniform datatypes in lists for the tracer.
    Going from list[Union[torch.Tensor, list[torch.Tensor]]] to list[torch.Tensor]
    means we have to stack the lists in the output.
    """
    # assert data.shape == (128, 1e3, 1e3), "Changing the chape requires re-tracing."
    coeff = ptwt.wavedec2(data, wavelet, mode="periodic", level=5)
    coeff2 = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            coeff2.append(c)
        else:
            coeff2.append(torch.stack(c))
    return coeff2


if __name__ == "__main__":
    repetitions = 10
    length = 1e3
    img_size = 512

    pywt_time_cpu = []

    ptwt_time_cpu = []
    ptwt_time_gpu = []
    ptwt_time_gpu_jit = []

    # for _ in range(repetitions):
    #     data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    #     start = time.perf_counter()
    #     # pywt_res = pywt.wavedec2(data, "db5", level=5, mode="periodic")
    #     end = time.perf_counter()
    #     pywt_time_cpu.append(end - start)

    # for _ in range(repetitions):
    #     data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    #     data = torch.from_numpy(data)
    #     start = time.perf_counter()
    #     # res = ptwt.wavedec2(data, "db5", mode="periodic", level=5)
    #     end = time.perf_counter()
    #     ptwt_time_cpu.append(end - start)

    # for i in range(repetitions):
    #     data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    #     data = torch.from_numpy(data).cuda()

    #     start = time.perf_counter()
    #     # res = ptwt.wavedec2(data, "db5", mode="periodic", level=5)
    #     # torch.cuda.synchronize()
    #     end = time.perf_counter()
    #     print("rep: ", i)
    #     ptwt_time_gpu.append(end - start)

    # wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    # jit_wavedec = torch.jit.trace(
    #     _to_jit_wavedec_2,
    #     (data.cuda(), wavelet),
    #     strict=False,
    # )

    # for i in range(repetitions):
    #     data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    #     data = torch.from_numpy(data).cuda()

    #     pc_start = time.perf_counter()
    #     res = jit_wavedec(data, wavelet)
    #     torch.cuda.synchronize()
    #     pc_end = time.perf_counter()
    #     print("rep: ", i)
    #     ptwt_time_gpu_jit.append(pc_end - pc_start)
        
    # ptwt_time_gpu_compile = []

    # def _wavedec2_native(x: torch.Tensor, wt) -> list:
    #     return ptwt.wavedec2(x, wt, mode="periodic", level=5)

    # compiled_wavedec = torch.compile(_wavedec2_native)

    # # Warm-up: triggers Triton kernel compilation (paid once, not in the timed loop).
    # # Run twice: first call compiles, second call confirms steady-state execution.
    # for _ in range(2):
    #     _ = compiled_wavedec(data.cuda(), wavelet)
    #     torch.cuda.synchronize()

    # for i in range(repetitions):
    #     data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    #     data = torch.from_numpy(data).cuda()

    #     pc_start = time.perf_counter()
    #     res = compiled_wavedec(data, wavelet)
    #     torch.cuda.synchronize()
    #     pc_end = time.perf_counter()
    #     print("rep: ", i)
    #     ptwt_time_gpu_compile.append(pc_end - pc_start)

    # print("2d fwt results")
    # print(
    #     f"2d-pywt-cpu    :{np.mean(pywt_time_cpu):5.5f} +- {np.std(pywt_time_cpu):5.5f}"
    # )
    # print(
    #     f"2d-ptwt-cpu    :{np.mean(ptwt_time_cpu):5.5f} +- {np.std(ptwt_time_cpu):5.5f}"
    # )
    # print(
    #     f"2d-ptwt-gpu    :{np.mean(ptwt_time_gpu):5.5f} +- {np.std(ptwt_time_gpu):5.5f}"
    # )
    # print(
    #     f"2d-ptwt-gpu-jit:{np.mean(ptwt_time_gpu_jit):5.5f} +- {np.std(ptwt_time_gpu_jit):5.5f}"
    # )
    # print(
    #     f"2d-ptwt-gpu-cmp:{np.mean(ptwt_time_gpu_compile):5.5f}"
    #     f" +- {np.std(ptwt_time_gpu_compile):5.5f}"
    # )
    
    ptwt_time_gpu_compile_rec = []
    ptwt_time_gpu_rec = []

    # Wrap waverec2 so dynamo treats it as opaque.
    _waverec2_opaque = torch.compiler.disable(ptwt.waverec2)

    def _waverec2_native(coeffs: list, wt) -> torch.Tensor:
        return _waverec2_opaque(coeffs, wt)

    compiled_waverec = torch.compile(_waverec2_native)

    # Get a set of coefficients to reconstruct from (reuse from the compile dec run).
    data = np.random.randn(img_size, int(length), int(length)).astype(np.float32)
    data = torch.from_numpy(data).cuda()
    wavelet = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet("db5"), torch.float32)
    
    def _wavedec2_native(x: torch.Tensor, wt) -> list:
        return ptwt.wavedec2(x, wt, mode="periodic", level=5)

    compiled_wavedec = torch.compile(_wavedec2_native)
    dummy_coeffs = compiled_wavedec(data.cuda(), wavelet)
    # dummy_coeffs = ptwt.wavedec2(data.cuda(), wavelet, mode="periodic", level=5)

    # Warm-up: trigger Triton compilation.
    for _ in range(2):
        _ = compiled_waverec(dummy_coeffs, wavelet)
        torch.cuda.synchronize()

    for i in range(repetitions):
        # Re-use the same coefficients each iteration — we're timing reconstruction only.
        pc_start = time.perf_counter()
        res = compiled_waverec(dummy_coeffs, wavelet)
        torch.cuda.synchronize()
        pc_end = time.perf_counter()
        print("rep: ", i)
        ptwt_time_gpu_compile_rec.append(pc_end - pc_start)

    for i in range(repetitions):
        start = time.perf_counter()
        res = ptwt.waverec2(dummy_coeffs, wavelet)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print("rep: ", i)
        ptwt_time_gpu_rec.append(end - start)
        
    print(
        f"2d-ptwt-gpu-cmp-rec:{np.mean(ptwt_time_gpu_compile_rec):5.5f}"
        f" +- {np.std(ptwt_time_gpu_compile_rec):5.5f}"
    )
    print(
        f"2d-ptwt-gpu-rec:{np.mean(ptwt_time_gpu_rec):5.5f}"
        f" +- {np.std(ptwt_time_gpu_rec):5.5f}"
    )