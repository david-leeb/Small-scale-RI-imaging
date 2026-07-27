from typing import Tuple, List, Union
import torch
import numpy as np
import ptwt
import nvtx
import pywt
import warnings

from .prox_op import ProxOp

@torch.compile
def _primal_update_inplace(result, x, recon):
    result.copy_(torch.clamp(x - recon, min=0.0))

@torch.compile
def _compute_norm_l2(val, result, x):
    val.copy_(torch.sum((result - x) ** 2))

@torch.compile
def _fused_1d_dual_update(dual_1d, psit_1d, weights_1d, sfth_val, norm_acc):
    tmp = dual_1d + psit_1d
    threshold = sfth_val * weights_1d
    # Fused soft-thresholding
    dual_1d.copy_(tmp - torch.sign(tmp) * torch.clamp(torch.abs(tmp) - threshold, min=0.0))
    # Fused L1 Norm calculation
    norm_acc.copy_(torch.sum(torch.abs(psit_1d) * weights_1d))
 
def _traceable_wavedec2(data: torch.Tensor, wavelet: tuple, level: int) -> List[torch.Tensor]:
    coeff = ptwt.wavedec2(data, wavelet, level=level, mode="zero")
    res = []
    for c in coeff:
        if isinstance(c, torch.Tensor):
            res.append(c)
        else:
            res.append(torch.stack(c, dim=0))
    return res

def _traceable_waverec2(coeffs: List[torch.Tensor], wavelet: tuple) -> torch.Tensor:
    unstacked_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:
            unstacked_coeffs.append(c)
        else:
            unstacked_coeffs.append(tuple(torch.unbind(c, dim=0)))
    return ptwt.waverec2(unstacked_coeffs, wavelet)

WaveletCoeff = List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
WaveletDictCoeff = List[Union[WaveletCoeff, torch.Tensor]]
        
class ProxOpSARAPos(ProxOp):
    """
    GPU-optimised proximity operator for SARA with positivity constraint.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        sfth_val: float = 1.0e-4,
        wl_dict: Tuple[str, ...] = (
            "dirac", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8",
        ),
        dec_lev: int = 4,
        wl_noise_floor: float = 1.0e-4,
        mode: str = "zero",
        max_iter: int = 20,
        obj_tol: float = 1e-4,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        assert device.type == "cuda", "ProxOpSARAPos requires a CUDA device."
        super().__init__(device=device, dtype=dtype)

        self._img_size = img_size
        self._sfth_val = sfth_val
        self._dec_lev = torch.tensor(dec_lev)
        self._wl_noise_floor = wl_noise_floor
        self._mode = mode
        self._scale_factor = float(np.sqrt(len(wl_dict)))
        self._wl_dict = list(wl_dict)
        self._dirac = "dirac" in self._wl_dict
        self._verbose = verbose
        self._weights: WaveletDictCoeff = []
        self._max_iter = max_iter
        self._obj_tol = obj_tol
        if self._dirac:
            self._wl_dict.remove("dirac")

        self._dec_fns = []
        self._rec_fns = []
        self._wavelets = []
        
        # Dummy data required for tracing
        dummy_x = torch.zeros(1, 1, *img_size, device=device, dtype=dtype)
        
        for b in self._wl_dict:
            # Create wavelet tensor tuple
            wt = ptwt.WaveletTensorTuple.from_wavelet(pywt.Wavelet(b), dtype=dtype)
            wt = ptwt.WaveletTensorTuple(*(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in wt
            ))
            self._wavelets.append(wt)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Trace the decomposition
                jit_dec = torch.jit.trace(
                    _traceable_wavedec2, 
                    (dummy_x, wt, self._dec_lev), 
                    strict=False
                )
                self._dec_fns.append(jit_dec)
                
                # Trace the reconstruction
                dummy_coeffs = jit_dec(dummy_x, wt, self._dec_lev)
                jit_rec = torch.jit.trace(
                    _traceable_waverec2, 
                    (dummy_coeffs, wt), 
                    strict=False
                )
                self._rec_fns.append(jit_rec)

        # Buffers for CUDA Graph capture
        self._x_buf   = torch.zeros(1, 1, *img_size, device=device, dtype=dtype)
        self._result  = torch.zeros(1, 1, *img_size, device=device, dtype=dtype)
        self._recon   = torch.zeros(1, 1, *img_size, device=device, dtype=dtype)
        self._norm_l2 = torch.zeros(1, device=device, dtype=dtype)
        self._norm_l1 = torch.zeros(1, device=device, dtype=dtype)

        # Initialise dual variable
        dummy_psit = self._wavedec2_dict(self._x_buf)

        # Get number of elements for each wavelet
        self._psit_numels = []
        for i in range(len(self._wl_dict)):
            for j in range(len(dummy_psit[i])):
                self._psit_numels.append(dummy_psit[i][j].numel())
        if self._dirac:
            self._psit_numels.append(dummy_psit[-1].numel())

        total_elements = sum(self._psit_numels)

        # Allocate 1D arrays
        self._dual_1d = torch.zeros(total_elements, device=device, dtype=dtype)
        self._weights_1d = torch.zeros(total_elements, device=device, dtype=dtype)
        self._psit_1d = torch.zeros(total_elements, device=device, dtype=dtype)

        self._dual: WaveletDictCoeff = []
        offset = 0
        for i in range(len(self._wl_dict)):
            basis_list = []
            for j in range(len(dummy_psit[i])):
                shape = dummy_psit[i][j].shape
                numel = dummy_psit[i][j].numel()
                basis_list.append(self._dual_1d[offset : offset + numel].view(shape))
                offset += numel
            self._dual.append(basis_list)

        if self._dirac:
            shape = dummy_psit[-1].shape
            numel = dummy_psit[-1].numel()
            self._dual.append(self._dual_1d[offset : offset + numel].view(shape))

        # CUDA graph
        self._graph: torch.cuda.CUDAGraph | None = None

    def _wavedec2_dict(self, x: torch.Tensor) -> WaveletDictCoeff:
        """
        Wavelet decomposition across all bases.
        """
        scaled = x / self._scale_factor
        coeff: WaveletDictCoeff = []
        for i, fn in enumerate(self._dec_fns):
            coeff.append(fn(scaled, self._wavelets[i], self._dec_lev))
        if self._dirac:
            coeff.append(scaled.clone())
        return coeff

    def _waverec2_dict_into(self, y: WaveletDictCoeff) -> None:
        """
        Wavelet reconstruction across all bases
        """
        self._recon.zero_()
        for i, fn in enumerate(self._rec_fns):
            self._recon.add_(fn(y[i], self._wavelets[i]) / self._scale_factor)
        if self._dirac:
            self._recon.add_(y[-1] / self._scale_factor)

    def _sfth_dual(self, psit: WaveletDictCoeff) -> None:
        """
        Gathers all psit tensors into a 1D buffer and executes a single 
        fused vector operation.
        """
        # Flatten the nested psit structure into a list of 1D views
        flat_psit_views = []
        for i in range(len(self._wl_dict)):
            for j in range(len(psit[i])):
                flat_psit_views.append(psit[i][j].flatten())
        if self._dirac:
            flat_psit_views.append(psit[-1].flatten())

        torch.cat(flat_psit_views, dim=0, out=self._psit_1d)

        _fused_1d_dual_update(
            self._dual_1d,
            self._psit_1d,
            self._weights_1d,
            self._sfth_val,
            self._norm_l1,
        )

    def _iteration(self) -> None:
        """
        One prox sub-iteration.
        """
        # Primal update
        self._waverec2_dict_into(self._dual)
        _primal_update_inplace(self._result, self._x_buf, self._recon)

        # Accumulate l2 norm
        _compute_norm_l2(self._norm_l2, self._result, self._x_buf)

        # Dual update
        self._sfth_dual(self._wavedec2_dict(self._result))

    def _capture_graph(self) -> None:
        """
        Warm up then capture one call to _iteration() as a CUDA graph.
        """
        
        dual_backup = self._dual_1d.clone()
        
        # Warm-up
        for _ in range(3):
            self._iteration()

        self._norm_l2.zero_()
        self._norm_l1.zero_()
        torch.cuda.synchronize(self._device)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._iteration()
        
        self._dual_1d.copy_(dual_backup)
        
        if self._verbose:
            print("  CUDA graph captured.", flush=True)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SARA + positivity proximity operator.
        """
        # if not self._weights:
        #     raise RuntimeError("Call update() before __call__() to initialise weights.")

        self._x_buf.copy_(x)

        if self._graph is None:
            self._capture_graph()

        self._norm_l2.zero_()

        obj_val_prev = -1
        for i in range(self._max_iter):
            with nvtx.annotate("Prox_Subiteration"):
                self._graph.replay()
                # self._iteration()
                obj_val = 0.5 * self._norm_l2.item() + self._sfth_val * self._norm_l1.item()
                obj_rel_var = abs(obj_val - obj_val_prev) / obj_val
                obj_val_prev = obj_val
                if obj_rel_var < self._obj_tol:
                    break

        if self._verbose:
            print(
                f"  Prox converged: Iter {i+1}, rel_fval = {obj_rel_var},",
                f"l1norm_w = {self._norm_l1.item()}",
                flush=True,
            )

        return self._result.clone()

    def update(self, x: torch.Tensor, initialisation: bool = False) -> None:
        """Update the weight for l1 norm.."""
        flat_weight_views = []
        
        if initialisation:
            flat_weight_views = [
                torch.ones(numel, device=self._device, dtype=self._dtype) 
                for numel in self._psit_numels
            ]
        else:
            x = x.to(self._dtype).to(self._device)
            scaled = x / self._scale_factor
            for i, fn in enumerate(self._dec_fns):
                curr_coeff = fn(scaled, self._wavelets[i], self._dec_lev)
                
                w_approx = self._wl_noise_floor / (self._wl_noise_floor + torch.abs(curr_coeff[0]))
                flat_weight_views.append(w_approx.flatten())
                for lev in range(1, self._dec_lev + 1):
                    w_det = self._wl_noise_floor / (self._wl_noise_floor + torch.abs(curr_coeff[lev]))
                    flat_weight_views.append(w_det.flatten())
                    
            if self._dirac:
                w_dirac = self._wl_noise_floor / (self._wl_noise_floor + torch.abs(scaled))
                flat_weight_views.append(w_dirac.flatten())

        # Pack all the weights into the 1D buffer
        torch.cat(flat_weight_views, dim=0, out=self._weights_1d)

        self._graph = None  # Graph must be recaptured because weights changed
        
    def get_l1_norm(self) -> float:
        """
        Gets the latest l1 norm.
        """
        return self._norm_l1.item()

    def set_noise_floor_level(self, wl_noise_floor: float) -> None:
        """
        Sets the noise floor level in wavelet coefficient.
        """
        self._wl_noise_floor = wl_noise_floor

    def set_soft_thresholding_value(self, sfth_val: float) -> None:
        """
        Sets the soft thresholding level of the proximity operator.
        """
        self._sfth_val = sfth_val
        self._graph = None