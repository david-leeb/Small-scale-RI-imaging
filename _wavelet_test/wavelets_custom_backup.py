
from __future__ import annotations

from typing import Optional, Union, cast, List, Tuple, Union

import pywt
import torch
import torch.nn.functional as F

from ptwt_utils import (
    AxisHint,
    _adjust_padding_at_reconstruction,
    _check_same_device_dtype,
    _construct_2d_filt,
    _get_filter_tensors,
    _get_padding_n,
    _group_for_symmetric,
    _pad_symmetric,
    _postprocess_coeffs,
    _postprocess_tensor,
    _preprocess_coeffs,
    _preprocess_deconstruction,
    _translate_boundary_strings,
)
from ptwt_constants import BoundaryMode, Wavelet, WaveletCoeff2d, WaveletDetailTuple2d

def wavedec2_new_2d(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int] = (-2, -1),
):
  
    data, ds, dec_lo, dec_hi, dec_filt = _preprocess_deconstruction(
        data, wavelet, axes=axes, ndim=2
    )

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    leading_shape = data.shape[:-2]
    result_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    res_ll = data
    for _ in range(level):
        padding = cast(tuple[int, int, int, int], _get_padding_n(res_ll, wavelet, n=2))
        res_ll = torch.nn.functional.pad(res_ll, padding, mode="constant")
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=2)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        result_lst.append((res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)))

    result_lst.reverse()
    res_ll = res_ll.squeeze(1)
    
    flat: List[torch.Tensor] = [res_ll.reshape(*leading_shape, *res_ll.shape[-2:])]
    for lh, hl, hh in result_lst:
        hw = lh.shape[-2:]
        flat.append(lh.reshape(*leading_shape, *hw))
        flat.append(hl.reshape(*leading_shape, *hw))
        flat.append(hh.reshape(*leading_shape, *hw))
    return flat

def wavedec2_new(
    data: torch.Tensor,
    wavelet: Union[Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: tuple[int, int] = (-2, -1),
):
  
    data, ds, dec_lo, dec_hi, dec_filt = _preprocess_deconstruction(
        data, wavelet, axes=axes, ndim=2
    )

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    dec_lo_1d = dec_lo.squeeze()
    dec_hi_1d = dec_hi.squeeze()
    w_dec_width = torch.stack([dec_lo_1d, dec_hi_1d], dim=0).unsqueeze(1).unsqueeze(2)
    v_pair = torch.stack([dec_lo_1d, dec_hi_1d], dim=0)
    w_dec_height = torch.cat([v_pair, v_pair], dim=0).unsqueeze(1).unsqueeze(3)
    
    leading_shape = data.shape[:-2]
    result_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    res_ll = data.reshape(-1, 1, data.shape[-2], data.shape[-1])
    
    for _ in range(level):
        padl, padr, padt, padb = cast(
            tuple[int, int, int, int], _get_padding_n(res_ll, wavelet, n=2)
        )
        
        x_wpad = F.pad(res_ll, (padl, padr, 0, 0), mode="constant")
        stage1 = F.conv2d(x_wpad, w_dec_width, stride=(1, 2))

        stage1_hpad = F.pad(stage1, (0, 0, padt, padb), mode="constant")
        res = F.conv2d(stage1_hpad, w_dec_height, stride=(2, 1), groups=2)
        
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, dim=1)
        res_ll = res_ll.contiguous()
        
        result_lst.append((res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)))

    result_lst.reverse()
    res_ll = res_ll.squeeze(1)
    
    flat: List[torch.Tensor] = [res_ll.reshape(*leading_shape, *res_ll.shape[-2:])]
    for lh, hl, hh in result_lst:
        hw = lh.shape[-2:]
        flat.append(lh.reshape(*leading_shape, *hw))
        flat.append(hl.reshape(*leading_shape, *hw))
        flat.append(hh.reshape(*leading_shape, *hw))
    return flat

def waverec2_new(
    coeffs: List[torch.Tensor],
    wavelet: Union[Wavelet, str],
    *,
    axes: AxisHint = None,
) -> torch.Tensor:
    
    leading_shape = coeffs[0].shape[:-2]
    
    detail_tuples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i in range(1, len(coeffs), 3):
        detail_tuples.append((coeffs[i], coeffs[i + 1], coeffs[i + 2]))
    
    torch_device, torch_dtype = _check_same_device_dtype(coeffs)

    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    rec_filt = _construct_2d_filt(lo=rec_lo, hi=rec_hi)

    res_ll = coeffs[0].reshape(-1, *coeffs[0].shape[-2:])
    
    num_levels = len(detail_tuples)
    
    for c_pos, (res_lh, res_hl, res_hh) in enumerate(detail_tuples):

        hw = res_lh.shape[-2:]
        res_ll = torch.stack([
            res_ll,
            res_lh.reshape(-1, *hw),
            res_hl.reshape(-1, *hw),
            res_hh.reshape(-1, *hw),
        ], 1)
        res_ll = torch.nn.functional.conv_transpose2d(
            res_ll, rec_filt, stride=2
        ).squeeze(1)

        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2

        if c_pos < num_levels - 1:
            next_target_shape = detail_tuples[c_pos + 1][0].shape
            padr, padl = _adjust_padding_at_reconstruction(
                res_ll.shape[-1], next_target_shape[-1], padr, padl
            )
            padb, padt = _adjust_padding_at_reconstruction(
                res_ll.shape[-2], next_target_shape[-2], padb, padt
            )

        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]

    out_hw = res_ll.shape[-2:]
    res_ll = res_ll.reshape(*leading_shape, *out_hw)

    return res_ll