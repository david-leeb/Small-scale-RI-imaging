"""
Proximity operator for SARA and positivity
"""

from typing import Tuple, List, Union
import torch
import numpy as np
import ptwt
import pywt
import nvtx
import warnings
import torch.nn.functional as F

from .prox_op import ProxOp
from .db_wavelets import CompiledWaveletBank

WaveletCoeff = List[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
]
WaveletDictCoeff = List[Union[WaveletCoeff, torch.Tensor]]


class ProxOpSARAPos(ProxOp):
    """
    Proximity operator for SARA with positivity constrained.

    It provides methods for wavelet decomposition and reconstruction with multiple bases,
    proximity operator for the adjoint of l1 norm, soft thresholding, and updating weights
    for l1 norm. We use pytorch wavelet toolbox (ptwt) for wavelet decomposition and
    reconstruction.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        sfth_val: float = 1.0e-4,
        wl_dict: Tuple[str, ...] = (
            "dirac",
            "db1",
            "db2",
            "db3",
            "db4",
            "db5",
            "db6",
            "db7",
            "db8",
        ),
        dec_lev: int = 4,
        wl_noise_floor: float = 1.0e-4,
        mode: str = "zero",
        max_iter: int = 20,
        obj_tol: float = 1e-4,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the SARA positivity proximity operator with the given parameters.

        Args:
            img_size (Tuple[int, int]): The size of the image.
            sfth_val (float, optional): The soft thresholding value. Defaults to 1.0e-4.
            wl_dict (Tuple[str, ...], optional): The wavelet dictionary.
                Defaults to ("dirac", "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8").
            dec_lev (int, optional): The decomposition level. Defaults to 4.
            wl_noise_floor (float, optional): The noise floor level in wavelet coefficient.
                Defaults to 1.0e-4.
            mode (str, optional): The mode for wavelet decomposition and reconstruction.
                Defaults to "zero".
            max_iter (int, optional): The maximum number of iterations. Defaults to 20.
            obj_tol (float, optional): The tolerance for the objective function. Defaults to 1e-4.
            device (torch.device, optional): The device on which the computations are performed.
                Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)

        self._img_size = img_size
        self._sfth_val = sfth_val
        self._dec_lev = dec_lev
        self._wl_noise_floor = wl_noise_floor
        self._mode = mode
        self._scale_factor = np.sqrt(len(wl_dict))
        self._wl_dict = list(wl_dict)
        self._dirac = "dirac" in self._wl_dict
        self._verbose = verbose
        self._weights = []
        self._norm_l1 = 0.0
        self._max_iter = max_iter
        self._obj_tol = obj_tol
        if self._dirac:
            self._wl_dict.remove("dirac")
            
        self._scale_factor_inv = torch.tensor(
            1.0 / self._scale_factor, 
            device=self._device, 
            dtype=self._dtype
        )
        
        self.wavelet_bank = CompiledWaveletBank(self._wl_dict, self._dec_lev, self._img_size, self._device, self._dtype, self._dirac)
        
        self._init_state_buffers()

    def _init_state_buffers(self):
        # We only need one dummy pass to establish the size of the 1D flat vector
        dummy_x = torch.zeros((1, 1, *self._img_size), device=self._device, dtype=self._dtype)
        dummy_flat = self.wavelet_bank.decompose_flat(dummy_x)
        
        self._dual_flat = torch.zeros_like(dummy_flat)
        self._weights_flat = torch.ones_like(dummy_flat)
        self._latest_norm_l1 = 0.0
        self._sfth_val_tensor = torch.tensor(self._sfth_val, device=self._device, dtype=self._dtype)
    
    @nvtx.annotate()
    @torch.compile(mode="max-autotune")
    def _run_compiled_solver(
        self, x: torch.Tensor, dual: torch.Tensor, weights: torch.Tensor, sfth_val_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the entire proximal optimization loop entirely on the GPU.
        Fuses operations across all 20 iterations with zero Python overhead.
        """
        curr_dual = dual
        result = x
        norm_l1 = torch.zeros((), device=x.device, dtype=x.dtype)
        thresh = sfth_val_t * weights
        
        # torch.compile unrolls or loops this natively on the hardware
        for _ in range(self._max_iter):
            # Primal Update
            rec = self.wavelet_bank.reconstruct_from_flat(curr_dual)
            result = F.relu(x - (rec * self._scale_factor_inv))
            
            # Dual Update 
            psit_flat = self.wavelet_bank.decompose_flat(result * self._scale_factor_inv)
            
            # Keep track of the L1 norm purely on the GPU
            norm_l1 = torch.sum(torch.abs(psit_flat * weights))
            
            # Clean dual clipping 
            curr_dual = torch.clamp(curr_dual + psit_flat, min=-thresh, max=thresh)
            
        return result, curr_dual, norm_l1
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the proximity operator to the input tensor using pure eager mode.
        """
        
        # Reset the dual variables for the new solver pass
        self._dual_flat.zero_()
        
        # Execute the entire 20-iteration chain in one shot
        torch.compiler.cudagraph_mark_step_begin()
        result, final_dual, norm_l1 = self._run_compiled_solver(
            x, self._dual_flat, self._weights_flat, self._sfth_val_tensor
        )
        
        # Update your state buffer cleanly without ruining graph tracking
        self._dual_flat.copy_(final_dual)
        
        # Save the final L1 norm for your getter function
        self._latest_norm_l1 = norm_l1.item()

        return result.clone()
    
    @nvtx.annotate()
    def update(self, x: torch.tensor, initialisation: bool = False) -> None:
        """
        Update the weight for l1 norm using 1D contiguous vectors.
        """
        if initialisation:
            self._weights_flat.fill_(1.0)
            return
        
        x = x.to(dtype=self._dtype, device=self._device)
        coeffs_flat = self.wavelet_bank.decompose_flat(x)
        w_flat = self._wl_noise_floor / (self._wl_noise_floor + torch.abs(coeffs_flat))
        
        self._weights_flat.copy_(w_flat)

    def get_l1_norm(self) -> float:
        """
        Gets the latest l1 norm calculated in `_sfth_dual`.

        Returns:
            float: The latest l1 norm.
        """
        return self._latest_norm_l1

    def set_noise_floor_level(self, wl_noise_floor: float) -> None:
        """
        Sets the noise floor level in wavelet coefficient.

        Args:
            wl_noise_floor (float): The noise floor level to set.
        """
        self._wl_noise_floor = wl_noise_floor

    def set_soft_thresholding_value(self, sfth_val: float):
        """
        Sets the soft thresholding level of the proximity operator.

        Args:
            sfth_val (float): The soft thresholding level for the wavelet coefficients.
        """
        self._sfth_val = sfth_val
        if hasattr(self, '_sfth_val_tensor'):
            self._sfth_val_tensor.fill_(sfth_val)
