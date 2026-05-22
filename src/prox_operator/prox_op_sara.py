"""
Proximity operator for SARA and positivity
"""

from typing import Tuple, List, Union
import torch
import numpy as np
import ptwt

from .prox_op import ProxOp

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

        # initialise dual variable
        self._dual = self._wavedec2_dict(
            torch.zeros((1, 1, *self._img_size), device=self._device, dtype=self._dtype)
        )

        # self._coeffShape = []
        # img = torch.zeors(1,1,*img_size)
        # for basis in self._wl_dict:
        #     curr_coeff = ptwt.wavedec2(img, basis, level=self._dec_lev, mode=self._mode)
        #     coeffStart = 0
        #     coeffEnd = coeffStart + torch.numel(curr_coeff[0])
        #     self._coeffShape.append([(coeffStart,coeffEnd), curr_coeff[0].shape])
        #     for i in range(1, self.dec_lev+1):
        #         currShape = []
        #         for j in range(3):
        #             coeffStart = coeffEnd
        #             coeffEnd = coeffStart + torch.numel(curr_coeff[i][0])

    def _wavedec2_dict(self, x: torch.Tensor) -> WaveletDictCoeff:
        """
        Performs wavelet decomposition with multiple bases on the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be decomposed.

        Returns:
            WaveletDictCoeff:
                A list of wavelet coefficients. Each element in the list corresponds to the
                wavelet decomposition with one basis. For each basis, the wavelet coefficients
                are stored in a list where the first element is the approximation coefficients,
                and the following tuple of three tensors corresponding to the horizontal, vertical,
                and diagonal details. If the dirac basis is used, the last element in the list is
                the scaled input tensor.

        """
        coeff: WaveletDictCoeff = []
        for basis in self._wl_dict:
            curr_coeff = list(
                ptwt.wavedec2(
                    x / self._scale_factor, basis, level=self._dec_lev, mode=self._mode
                )
            )
            coeff.append(curr_coeff)
        if self._dirac:
            coeff.append(x / self._scale_factor)
        return coeff

    def _waverec2_dict(self, y: WaveletDictCoeff) -> torch.Tensor:
        """
        Performs wavelet reconstruction with multiple bases on the input tensor.

        Args:
            y (WaveletDictCoeff):
                A list of wavelet coefficients. Each element in the list corresponds to the
                wavelet decomposition with one basis. For each basis, the wavelet coefficients
                are stored in a list where the first element is the approximation coefficients,
                and the following tuple of three tensors corresponding to the horizontal, vertical,
                and diagonal details. If the dirac basis is used, the last element in the list is
                the scaled input tensor.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        result = torch.zeros(
            1, 1, *self._img_size, device=self._device, dtype=self._dtype
        )
        for i, basis in enumerate(self._wl_dict):
            result += ptwt.waverec2(y[i], basis) / self._scale_factor
        if self._dirac:
            result += y[-1] / self._scale_factor
        return result

    def _prox_l1_adj(self, z: torch.tensor, sfth: float) -> torch.tensor:
        """
        Proximity operator for the adjoint of l1 norm.

        Args:
            z (torch.Tensor): The input tensor.
            sfth (float): The soft thresholding level.

        Returns:
            torch.Tensor: The result of applying the proximity operator on `z`.
        """
        return z - (
            torch.sign(z)
            * torch.maximum(
                torch.abs(z) - sfth,
                torch.zeros(1, 1, device=self._device, dtype=self._dtype),
            )
        )

    @torch.compile 
    def _sfth_dual(self, psit_img: WaveletDictCoeff) -> None:
        """
        Applies soft thresholding to `psit_img` and update the dual variable.

        Args:
            psit_img (WaveletDictCoeff):
                The list of wavelet coeeficitents on which soft thresholding will be applied.
        """
        self._norm_l1 = 0.0
        for i in range(len(self._wl_dict)):
            self._dual[i][0] = self._prox_l1_adj(
                self._dual[i][0] + psit_img[i][0], self._sfth_val * self._weights[i][0]
            )
            self._norm_l1 += torch.sum(torch.abs(psit_img[i][0] * self._weights[i][0]))
            for j in range(1, self._dec_lev + 1):
                self._dual[i][j] = tuple(
                    self._prox_l1_adj(
                        self._dual[i][j][k] + psit_img[i][j][k],
                        self._sfth_val * self._weights[i][j][k],
                    )
                    for k in range(3)
                )
                for k in range(3):
                    self._norm_l1 += torch.sum(
                        torch.abs(psit_img[i][j][k] * self._weights[i][j][k])
                    )
        if self._dirac:
            self._dual[-1] = self._prox_l1_adj(
                self._dual[-1] + psit_img[-1], self._sfth_val * self._weights[-1]
            )
            self._norm_l1 += torch.sum(torch.abs(psit_img[-1] * self._weights[-1]))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the proximity operator to the input tensor.
        The SARA prior and positivity constraint are splited with dual forward-backward algorithm.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the proximity operator.
        """
        # dual forward backward
        obj_val_prev = -1
        for i in range(self._max_iter):
            # primal update
            with torch.cuda.nvtx.range("primal_update"):
                result = torch.maximum(
                    x - self._waverec2_dict(self._dual),
                    torch.zeros(
                        1,
                        1,
                        device=self._device,
                        dtype=self._dtype,
                    ),
                )
                norm_l2 = torch.sum((result - x) ** 2)
                
            # dual update
            with torch.cuda.nvtx.range("dual_update"):
                self._sfth_dual(self._wavedec2_dict(result))
            
            # stop criterion
            with torch.cuda.nvtx.range("stop_criterion"):
                obj_val = 0.5 * norm_l2.item() + self._sfth_val * self._norm_l1.item()
                obj_rel_var = abs(obj_val - obj_val_prev) / obj_val
                obj_val_prev = obj_val
                if self._verbose > 1:
                    print(
                        f"  Prox Iter {i+1}, prox_fval = {obj_val},",
                        f"rel_fval = {obj_rel_var}, l1norm_w = {self._norm_l1.item()}",
                        flush=True,
                    )
                if obj_rel_var < self._obj_tol:
                    break
        if self._verbose:
            print(
                f"  Prox converged: Iter {i+1}, rel_fval = {obj_rel_var},",
                f"l1norm_w = {self._norm_l1.item()}",
                flush=True,
            )

        return result

    def update(self, x: torch.tensor, initialisation: bool = False) -> None:
        """
        Update the weight for l1 norm.

        Args:
            x (torch.Tensor): The input tensor.
            initialisation (bool, optional): If True, initialize the weights to 1.
                Defaults to False.
        """
        self._weights = []
        if initialisation:
            torch_one = torch.ones(
                1,
                1,
                device=self._device,
                dtype=self._dtype,
            )
            for basis in self._wl_dict:
                weighting_i = [torch_one.clone()]
                for i in range(1, self._dec_lev + 1):
                    weighting_i.append(tuple(torch_one.clone() for _ in range(3)))
                self._weights.append(weighting_i)
            if self._dirac:
                self._weights.append(torch_one.clone())
        else:
            x = x.to(self._dtype).to(self._device)
            for basis in self._wl_dict:
                curr_coeff = ptwt.wavedec2(
                    x / self._scale_factor, basis, level=self._dec_lev, mode=self._mode
                )
                weighting_i = [
                    self._wl_noise_floor
                    / (self._wl_noise_floor + torch.abs(curr_coeff[0]))
                ]
                for i in range(1, self._dec_lev + 1):
                    weighting_i.append(
                        tuple(
                            self._wl_noise_floor
                            / (self._wl_noise_floor + torch.abs(curr_coeff[i][j]))
                            for j in range(3)
                        )
                    )
                self._weights.append(weighting_i)
            if self._dirac:
                self._weights.append(
                    self._wl_noise_floor
                    / (self._wl_noise_floor + torch.abs(x / self._scale_factor))
                )

    def get_l1_norm(self) -> float:
        """
        Gets the latest l1 norm calculated in `_sfth_dual`.

        Returns:
            float: The latest l1 norm.
        """
        return self._norm_l1.item()

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
