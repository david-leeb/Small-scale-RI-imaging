"""
uSARA algorithm, built on forward-backward algorithm
"""

import os
from typing import Union
from timeit import default_timer as timer
import torch
import numpy as np
from astropy.io import fits

from .forward_backward import ForwardBackward
from ..prox_operator import ProxOpSARAPos
from ..ri_measurement_operator.pysrc.measOperator import MeasOpNUFFT
from ..ri_measurement_operator.pysrc.measOperator.meas_op_PSF import (
    MeasOpPSF,
)


class FBSARA(ForwardBackward):
    """
    FBSARA implements uSARA algorithm based on forward-backward algorithm.

    It handles data fidelity in the forward step and uses SARA prior and
    positivity constrained in the backward step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOpNUFFT,
        prox_op: ProxOpSARAPos,
        use_ROP: bool = False,
        meas_op_approx: Union[MeasOpPSF, None] = None,
        meas_op_classical: MeasOpNUFFT | None = None,
        y_uncompressed: torch.Tensor = None,
        im_min_itr: int = 30,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-5,
        heu_reg_scale: float = 1.0,
        new_heu: bool = False,
        im_max_itr_outer: int = 20,
        im_var_tol_outer: float = 1e-4,
        save_pth: str = "results",
        file_prefix: str = "",
        reweight_save: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the uSARA optimiser, which is built on forward-backward algorithm.

        Args:
            meas (torch.Tensor): The measurements to be used in the algorithm.
            meas_op (MeasOpTkbnRI): The measurement operator.
            prox_op (ProxOpSARAPos): The proximal operator.
            im_min_itr (int, optional): The minimum number of inner loop iterations.
                Defaults to 30.
            im_max_itr (int, optional): The maximum number of inner loop iterations.
                Defaults to 2000.
            im_var_tol (float, optional): The tolerance for inner loop image variation.
                Defaults to 1e-5.
            heu_reg_scale (float, optional): The heuristic regularization scale. Defaults to 1.0.
            im_max_itr_outer (int, optional): The maximum number of outer loop iterations.
                Defaults to 20.
            im_var_tol_outer (float, optional): The tolerance for outer loop image variation.
                Defaults to 1e-4.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            reweight_save (bool, optional): Whether to save the reweighting results.
            Defaults to True.
            verbose (bool, optional): Whether to print verbose logs. Defaults to True.
        """
        super(FBSARA, self).__init__(
            meas,
            meas_op if meas_op_approx is None else meas_op_approx,
            meas_op,
            prox_op,
            im_max_itr=im_max_itr * im_max_itr_outer,
            new_heu=new_heu,
            save_pth=save_pth,
            file_prefix=file_prefix,
        )

        self._use_ROP = use_ROP
        self._im_max_itr_inner = im_max_itr
        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._heu_reg_scale = heu_reg_scale
        self._im_max_itr_outer = im_max_itr_outer
        self._im_var_tol_outer = im_var_tol_outer
        self._reweight_save = reweight_save
        self._verbose = verbose
        self._new_heu = new_heu

        self._heuristic = 1.0
        self._model_prev_re = self._model
        self._iter_inner = 0
        self._iter_outer = 0
        self._reweighting_flag = False
        self._t_iter_outer = 0.0
        self._im_rel_var = 1.0
        self._im_rel_var_outer = 1.0
        
        self._meas_op_classical = meas_op_classical
        self._y_uncompressed = y_uncompressed
        
    def initialisation(self) -> None:
        """
        Initialises specific parameters of uSARA.

        This method initialises the step size and the uSARA proximity operator,
        sets the heuristic regularization parameter and the noise floor level in
        the wavelet domain. Some information will be printed if `verbose` is True.
        """
        # step size
        self._gd_step_size = 1.98 / self._meas_op_precise.get_op_norm()

        # heuristic noise level
        # self._heuristic = 1 / np.sqrt(2 * self._meas_op_precise.get_op_norm())
        if self._new_heu:
            noise = (torch.randn_like(self._meas, dtype=self._meas.dtype, device=self._meas.device) + 1j * torch.randn_like(self._meas, dtype=self._meas.dtype, device=self._meas.device)) / np.sqrt(2)
            self._heuristic = (self._meas_op.adjoint_op(noise) / self._meas_op.get_psf().max()).std().item() / self._meas_bp.max().item()
            if self._verbose:
                print(
                    f"INFO: using Sally's new heuristic",
                    flush=True,
                )
        else:
            self._heuristic = 1 / np.sqrt(2 * self._meas_op_precise.get_op_norm())
            if self._verbose:
                print(
                    f"INFO: measurement operator norm {self._meas_op_precise.get_op_norm()}",
                    flush=True,
                )
        if self._verbose:
            # print(
            #     f"INFO: measurement operator norm {self._meas_op_precise.get_op_norm()}",
            #     flush=True,
            # )
            print(f"INFO: heuristic noise level: {self._heuristic}", flush=True)
            
        if not self._use_ROP:
            heu_corr_factor = np.sqrt(
                self._meas_op_precise.get_op_norm_prime()
                / self._meas_op_precise.get_op_norm()
            )
        else:
            heu_corr_factor = 1.0
            
        if not np.isclose(heu_corr_factor, 1.0) and not self._use_ROP:
            self._heuristic *= heu_corr_factor
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after correction: {self._heuristic}, ",
                    f"corection factor {heu_corr_factor}",
                    flush=True,
                )
                
        # set noise floor level in wavelet domain
        self._prox_op.set_noise_floor_level(self._heuristic / 3.0)  # 9 wavelet bases
        if self._verbose:
            print(
                f"INFO: estimated noise floor level in wavelet coeeficients: {self._heuristic/3.0}"
            )
            
        if not np.isclose(self._heu_reg_scale, 1.0):
            self._heuristic *= self._heu_reg_scale
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after scaling: {self._heuristic},",
                    f"scaling factor {self._heu_reg_scale}",
                    flush=True,
                )
                
        # set soft thresholding value
        self._prox_op.set_soft_thresholding_value(
            self._heuristic / 3.0
        )  # 9 wavelet bases

        # initialise weighting for the l1 norm
        self._prox_op.update(self._model, initialisation=True)

        # other parameters
        self._model_prev_re = self._model
        self._iter_inner = 0
        self._iter_outer = 0
        self._reweighting_flag = False

        if self._verbose:
            print(f"INFO: forward-backward step size (gamma): {self._gd_step_size}")
            print(
                f"INFO: regularization parameter (lambda): {self._heuristic/3.0/self._gd_step_size}"
            )
            print(f"INFO: heuristic soft threolding level: {self._heuristic/3.0}")
            if isinstance(self._meas_op, MeasOpPSF):
                print(
                    "INFO: use approximate measurement operator for reconstruction",
                    flush=True,
                )
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:   uSARA   *********", flush=True)
            print("*************************************************", flush=True)

    def _each_iter_begin(self) -> None:
        """
        Method to be executed at the beginning of each iteration.

        This method resets the timer at the start of each reweighting loop.
        """
        if self._iter_inner == 0:
            print(
                "\n********************* Reweighting,",
                f"start major cycle {self._iter_outer+1}",
                "*********************\n",
                flush=True,
            )
            self._t_iter_outer = timer()

    @torch.no_grad()
    def _stop_criteria(self) -> bool:
        """
        Checks the stop criteria for the algorithm.

        The imaging loop will be terminated if the stop criteria of both inner loop
        and reweighting loop are met. The stop criteria are: 1) meets the minimum number
        of iterations and 2) the relative variation is smaller than tolerance or the
        maximum number of iterations is reached. The `_reweighting_flag` will be set if
        the inner loop stop criteria is met.

        Returns:
            bool: Whether the stop criteria is met.
        """
        # img relative variation
        self._im_rel_var = (
            torch.linalg.vector_norm(self._model - self._model_prev)
            / (torch.linalg.vector_norm(self._model) + 1e-10)
        ).item()

        # log
        if self._verbose:
            print(
                f"Cumulative itr: {self._iter+1},  re-weighting itr: {self._iter_outer+1},",
                f"forward-backward itr: {self._iter_inner+1},",
                f"relative variation {self._im_rel_var}",
                f"\ntimings: gradient step {self._t_forward} sec,",
                f"denoising step {self._t_backward} sec,",
                f"iteration {self._t_iter} sec.\n",
                flush=True,
            )

        # stop criteria
        if (
            self._iter_inner + 1 >= self._im_min_itr
            and self._im_rel_var < self._im_var_tol
        ) or self._iter_inner + 1 >= self._im_max_itr_inner:
            self._t_iter_outer = timer() - self._t_iter_outer
            self._reweighting_flag = True
            # img relative variation reweighting loop
            self._im_rel_var_outer = (
                torch.linalg.vector_norm(self._model - self._model_prev_re)
                / (torch.linalg.vector_norm(self._model) + 1e-10)
            ).item()
            self._model_prev_re = self._model
            # log
            if self._verbose:
                print(
                    "************************** Major cycle",
                    f"{self._iter_outer+1} finished **************************",
                    flush=True,
                )
                print(
                    f"INFO: Re-weighting iteration {self._iter_outer+1}",
                    f"completed in {self._t_iter_outer} sec.",
                    flush=True,
                )
                print(
                    f"INFO: Image relative variation of the major cycle {self._im_rel_var_outer}",
                    flush=True,
                )

            if (
                self._im_rel_var_outer < self._im_var_tol_outer
                or self._iter_outer + 1 >= self._im_max_itr_outer
            ):
                return True

        return False

    @torch.no_grad()
    def _each_iter_end(self) -> None:
        """
        Method to be executed at the end of each iteration.

        This method saves intermediate results, and updates the SARA wavelet coeeficients
        weights if the reweighting flag is set. The inner and outer iteration counters will
        also be updated accordingly.
        """
        if self._reweighting_flag:
            # update l1 weighting
            self._prox_op.update(self._model)

            if self._verbose:
                print(
                    "INFO: The std of the residual dirty image",
                    f"{np.std(self.get_residual_image()/self._psf_peak).item()}",
                    flush=True,
                )

            # save intermediate results
            if self._reweight_save:
                fits.writeto(
                    os.path.join(
                        self._save_pth,
                        self._file_prefix
                        + "tmp_model_major_itr_"
                        + str(self._iter_outer + 1)
                        + ".fits",
                    ),
                    self.get_model_image(dtype=torch.float32),
                    overwrite=True,
                )
                fits.writeto(
                    os.path.join(
                        self._save_pth,
                        self._file_prefix
                        + "tmp_residual_major_itr_"
                        + str(self._iter_outer + 1)
                        + ".fits",
                    ),
                    self.get_residual_image(dtype=torch.float32) / self._psf_peak,
                    overwrite=True,
                )

            self._iter_inner = 0
            self._iter_outer += 1
            self._reweighting_flag = False
        else:
            self._iter_inner += 1

    @torch.no_grad()
    def finalisation(self) -> None:
        """
        Finalises the imaging process and saves the final images.

        This method calculates the residual image, prints information about
        the process if verbose mode is on, and saves the final model and
        residual images in FITS format.
        """
        if self._verbose:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total} sec,",
                f"total number of iterations {self._iter + 1}",
                flush=True,
            )
            print(
                "INFO: The std of the normalised residual dirty image",
                f"{np.std(self.get_residual_image()/self._psf_peak).item()}",
                flush=True,
            )

        # save final images
        fits.writeto(
            os.path.join(self._save_pth, self._file_prefix + "model_image.fits"),
            self.get_model_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth, self._file_prefix + "residual_dirty_image.fits"
            ),
            self.get_residual_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth,
                self._file_prefix + "normalised_residual_dirty_image.fits",
            ),
            self.get_residual_image() / self._psf_peak,
            overwrite=True,
        )
        
        # If using ROP, save the residual using the classical measurement operator applied to the ROP output
        if self._use_ROP:
            x_out = torch.from_numpy(self.get_model_image()).to(device=self._meas_op_classical.get_device(), dtype=self._meas_op_classical._dtype)
            y_uncompressed = self._y_uncompressed.to(device=self._meas_op_classical.get_device(), dtype=self._meas_op_classical._dtype_meas)
            dirty = self._meas_op_classical.adjoint_op(y_uncompressed)
            model_bp = self._meas_op_classical.adjoint_op(self._meas_op_classical.forward_op(x_out))
            residual = (dirty - model_bp).squeeze().cpu().to(torch.double).numpy()
            
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix + "residual_dirty_image_ROP.fits",
                ),
                residual,
                overwrite=True,
            )
            
            # Save normalised residual
            psf_peak_classical = self._meas_op_classical.get_psf().max().item()
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix + "normalised_residual_dirty_image_ROP.fits",
                ),
                residual / psf_peak_classical,
                overwrite=True,
            )
