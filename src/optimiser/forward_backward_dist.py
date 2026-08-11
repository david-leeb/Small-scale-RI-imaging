"""
Forward-backward algorithm
"""
import os
from typing import Union
from timeit import default_timer as timer
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

from .optimiser import Optimiser
from ..prox_operator import ProxOp
from ..ri_measurement_operator.pysrc.measOperator import MeasOp


class ForwardBackward(Optimiser):
    """
    This class implements the forward-backward algorithm.

    We here assume the data fidelity term is differentiable and the regularisation term is
    non-differentiable.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOp,
        meas_op_precise: Union[MeasOp, None],
        prox_op: ProxOp,
        im_max_itr: int = 2000,
        algorithm: str = None,
        new_heu: bool = False,
        save_pth: str = "results",
        file_prefix: str = "",
        rank: int = None
    ) -> None:
        """
        Initializes the ForwardBackward optimiser.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOp): The measurement operator.
            meas_op_precise (Union[MeasOp, None]): The precise measurement operator.
            prox_op (ProxOp): The proximal operator.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            save_pth (str, optional): The path where results will be saved. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to None.
        """
        self.rank = rank
        is_root = self.rank == 0
        
        super().__init__(meas, meas_op, save_pth=save_pth, file_prefix=file_prefix)
        self._meas_op_precise = meas_op_precise
        self._prox_op = prox_op
        self._start_iter = 0
        self._im_max_itr = im_max_itr
        self._gd_step_size = 1.0
        self._new_heu = new_heu
        self._algorithm = algorithm

        # timing
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_forward = 0.0
        self._t_backward = 0.0

        self._iter = 0

        # cuda event
        self._forward_cuda_timing = False
        self._backward_cuda_timing = False
        if self._meas_op.get_device().type == "cuda":
            self._forward_cuda_timing = True
        if is_root and self._prox_op.get_device().type == "cuda":
            self._backward_cuda_timing = True

        # save dirty image and psf
        if self._meas_op.use_ROP:
            self._meas_bp = self._meas_op_precise.adjoint_op(self._meas)
        else:
            self._meas_bp = self._meas_op_precise.adjoint_classical(self._meas)
        if is_root: 
            self._meas_bp = self._meas_bp.to(self._meas_op_precise.get_device())
        
        self._psf = self._meas_op_precise.get_psf()
        if is_root:
            self._psf_peak = self._psf.max().item()
            print(f"PSF peak value: {self._psf_peak:.6e}")
        
        if self._new_heu and is_root:
            self._meas_bp_max = self._meas_bp.max().item()
        
        if is_root:
            fits.writeto(
                os.path.join(self._save_pth, "dirty.fits"),
                self.get_dirty_image() / self._psf_peak,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(self._save_pth, "psf.fits"),
                self.get_psf(),
                overwrite=True,
            )

    @torch.no_grad()
    def run(self) -> None:
        """
        Runs the main loop of the forward-backward algorithm.

        This method performs the forward and backward steps of the algorithm
        in a loop until the stop criteria is met.
        """
        
        is_root = self.rank == 0
        
        # timing with cuda events
        if self._forward_cuda_timing:
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
        if self._backward_cuda_timing:
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        
        if is_root:    
            self._t_total = timer()
            
        for self._iter in range(self._start_iter, self._im_max_itr):
            if is_root:
                self._t_iter = timer()
                self._each_iter_begin()
            
            # forward step
            with torch.cuda.nvtx.range("ForwardStep"):
                if self._forward_cuda_timing:
                    forward_start_event.record()
                elif is_root:
                    self._t_forward = timer()
                
                res = self._meas_op.forward_adjoint_op(self._model)
                
                if is_root:
                    res = res - self._meas_bp
                    if self._new_heu:
                        # Compute heuristic using cached _meas_bp_max and single GPU-CPU transfer
                        _cur_heuristic = res.std().item() / self._meas_bp_max
                        if self._algorithm == "usara":
                            _threshold = _cur_heuristic / 3.0
                            self._prox_op.set_noise_floor_level(_threshold) 
                            self._prox_op.set_soft_thresholding_value(_threshold)
                        else:
                            print("WARNING: Not implemented")
                            return

                    x_hat = self._model - self._gd_step_size * res
                    x_hat = x_hat.to(device=self._prox_op.get_device(), dtype=self._prox_op.get_data_type())

                if self._forward_cuda_timing:
                    forward_end_event.record()
                    torch.cuda.synchronize()
                    if is_root:
                        self._t_forward = forward_start_event.elapsed_time(forward_end_event) / 1e3
                elif is_root:
                    self._t_forward = timer() - self._t_forward

            # backward step
            with torch.cuda.nvtx.range("BackwardStep"):
                if is_root:
                    if self._backward_cuda_timing:
                        backward_start_event.record()
                    else:
                        self._t_backward = timer()
                        
                    self._model = self._prox_op(x_hat)
                    
                    self._model = self._model.to(device=self._meas_op.get_device()).to(
                        dtype=self._meas_op.get_data_type()
                    )
                    if self._backward_cuda_timing:
                        backward_end_event.record()
                        torch.cuda.synchronize()
                        self._t_backward = backward_start_event.elapsed_time(backward_end_event) / 1e3
                    else:
                        self._t_backward = timer() - self._t_backward
                    self._t_iter = timer() - self._t_iter

                    stop = self._stop_criteria()
                else:
                    stop = False

                # Sync stop condition across devices
                flags = torch.tensor(
                    [1 if stop else 0, 1 if (is_root and self._reweighting_flag) else 0],
                    device=self._meas_op.device,
                )
                dist.broadcast(flags, src=0)
                stop, reweighting_now = bool(flags[0].item()), bool(flags[1].item())

                residual = self.get_residual_image() if reweighting_now else None
                                
                if stop: 
                    break
                
                if is_root:
                    self._each_iter_end(residual)
                    self._model_prev = self._model

        if is_root:
            self._t_total = timer() - self._t_total

    def get_residual_image(self, dtype=torch.double) -> np.ndarray:
        """
        Get the residual image. Always use the precise measurement operator.

        args:
            dtype (torch.dtype): The data type of the output residual image.

        Returns:
            np.ndarray: The residual image.
        """
        AtAx = self._meas_op_precise.forward_adjoint_op(self._model)
        if self.rank != 0:
            return None
        return (self._meas_bp - AtAx).squeeze().cpu().to(dtype).numpy()
