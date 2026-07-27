from __future__ import annotations

from typing import List, Optional, Tuple, Union

import pywt
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from .ptwt_utils import (
    _adjust_padding_at_reconstruction,
    _get_filter_tensors,
)
from .ptwt_constants import Wavelet

Schedule = List[Tuple[int, int, int, int]]  # (padl, padr, padt, padb) per level

def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    padl = padr = (2 * filt_len - 3) // 2
    padr += data_len % 2
    return padl, padr

def _build_schedules(img_size: Tuple[int, int], filt_len: int, level: int) -> Tuple[Schedule, Schedule]:
    h, w = img_size
    shapes = [(h, w)]
    fwd_schedule: Schedule = []
    for _ in range(level):
        padl, padr = _get_pad(w, filt_len)
        padt, padb = _get_pad(h, filt_len)
        fwd_schedule.append((padl, padr, padt, padb))
        h = (h + padt + padb - filt_len) // 2 + 1
        w = (w + padl + padr - filt_len) // 2 + 1
        shapes.append((h, w))

    base = (2 * filt_len - 3) // 2
    crop_schedule: Schedule = []
    for c_pos in range(level):
        entering_h, entering_w = shapes[level - c_pos]
        padl = padr = padt = padb = base
        if c_pos < level - 1:
            raw_w = (entering_w - 1) * 2 + filt_len
            raw_h = (entering_h - 1) * 2 + filt_len
            target_h, target_w = shapes[level - c_pos - 1]
            padr, padl = _adjust_padding_at_reconstruction(raw_w, target_w, padr, padl)
            padb, padt = _adjust_padding_at_reconstruction(raw_h, target_h, padb, padt)
        crop_schedule.append((padl, padr, padt, padb))

    return fwd_schedule, crop_schedule

def _resolve_level(img_size: Tuple[int, int], wavelet: Union[Wavelet, str], level: Optional[int]) -> int:
    if level is not None:
        return level
    return pywt.dwtn_max_level([img_size[0], img_size[1]], wavelet)

def prepare_wavedec2_new(
    wavelet: Union[Wavelet, str],
    img_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    level: Optional[int] = None,
):
    dec_lo, dec_hi, _, _ = _get_filter_tensors(wavelet, flip=True, device=device, dtype=dtype)
    dec_lo_1d = dec_lo.squeeze()
    dec_hi_1d = dec_hi.squeeze()
    w_dec_width = torch.stack([dec_lo_1d, dec_hi_1d], dim=0).unsqueeze(1).unsqueeze(2)
    v_pair = torch.stack([dec_lo_1d, dec_hi_1d], dim=0)
    w_dec_height = torch.cat([v_pair, v_pair], dim=0).unsqueeze(1).unsqueeze(3)
    filt_len = dec_lo_1d.shape[-1]
    level = _resolve_level(img_size, wavelet, level)
    fwd_schedule, _ = _build_schedules(img_size, filt_len, level)
    return w_dec_width, w_dec_height, fwd_schedule

def prepare_waverec2_new(
    wavelet: Union[Wavelet, str],
    img_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    level: int,
):
    _, _, rec_lo, rec_hi = _get_filter_tensors(wavelet, flip=False, device=device, dtype=dtype)
    rec_lo_1d = rec_lo.squeeze()
    rec_hi_1d = rec_hi.squeeze()
    v_pair = torch.stack([rec_lo_1d, rec_hi_1d], dim=0)
    w_rec_height = torch.cat([v_pair, v_pair], dim=0).unsqueeze(1).unsqueeze(3)
    w_rec_width = v_pair.unsqueeze(1).unsqueeze(2)
    filt_len = rec_lo_1d.shape[-1]
    _, crop_schedule = _build_schedules(img_size, filt_len, level)
    return w_rec_height, w_rec_width, crop_schedule

def wavedec2_new(
    data: torch.Tensor,
    w_dec_width: torch.Tensor,
    w_dec_height: torch.Tensor,
    fwd_schedule: Schedule,
) -> List[torch.Tensor]:
    leading_shape = data.shape[:-2]
    result_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    res_ll = data.reshape(-1, 1, data.shape[-2], data.shape[-1])

    for padl, padr, padt, padb in fwd_schedule:
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
    w_rec_height: torch.Tensor,
    w_rec_width: torch.Tensor,
    crop_schedule: Schedule,
) -> torch.Tensor:
    leading_shape = coeffs[0].shape[:-2]
    res_ll = coeffs[0].reshape(-1, *coeffs[0].shape[-2:])
 
    for c_pos, (padl, padr, padt, padb) in enumerate(crop_schedule):
        res_lh, res_hl, res_hh = coeffs[1 + 3 * c_pos], coeffs[2 + 3 * c_pos], coeffs[3 + 3 * c_pos]
        hw = res_lh.shape[-2:]
        stacked = torch.stack([
            res_ll,
            res_lh.reshape(-1, *hw),
            res_hl.reshape(-1, *hw),
            res_hh.reshape(-1, *hw),
        ], 1)
        stage_a = F.conv_transpose2d(stacked, w_rec_height, stride=(2, 1), groups=2)
        if padt > 0:
            stage_a = stage_a[..., padt:, :]
        if padb > 0:
            stage_a = stage_a[..., :-padb, :]
 
        stage_b = F.conv_transpose2d(stage_a, w_rec_width, stride=(1, 2))
        if padl > 0:
            stage_b = stage_b[..., padl:]
        if padr > 0:
            stage_b = stage_b[..., :-padr]
        res_ll = stage_b.squeeze(1)
 
    out_hw = res_ll.shape[-2:]
    return res_ll.reshape(*leading_shape, *out_hw)
