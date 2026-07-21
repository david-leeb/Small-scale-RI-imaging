from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

import contextlib

from ptwt._util import _get_filter_tensors
# from ptwt._util import construct_2d_filt

Schedule = List[Tuple[int, int, int, int]]  # (padl, padr, padt, padb) per level

def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def _construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Construct two-dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        Stacked 2d-filters of dimension

        [2^2, 1, height, width].

        The four filters are ordered ll, lh, hl, hh.

    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    filt = filt.unsqueeze(1)
    return filt

"""
Fast, static-shape 2D wavelet transforms for a fixed (wavelet, img_size,
level), mode="zero" only. Two implementations, both validated against
ptwt.wavedec2/waverec2 (see validate_and_benchmark.py):

  Wavelet2D           dense (O(L^2) per level), bit-exact vs. ptwt.
  SeparableWavelet2D  row-then-column (O(L) per level), faster for
                      longer filters and large images, not bit-exact
                      (different float summation order) but validated
                      within a 1e-4 abs / 1e-3 rel tolerance. Handles
                      db1 (filt_len=2) via a small elementwise/reshape
                      path instead of conv_transpose2d -- see
                      waverec2_separable's docstring for why, and the
                      caveat that it needs re-confirming on real hardware.

make_wavelet_transform(...) is the function to actually use; it just
returns SeparableWavelet2D. Wavelet2D stays in this file as a slower but
unconditionally-validated fallback -- construct it directly for any
wavelet where you'd rather not depend on the filt_len==2 fix.

Structured as a close mirror of ptwt's own ptwt/conv_transform_2.py --
wavedec2()/waverec2() below have the same per-level loop shape and
variable names (res_ll, res_lh, res_hl, res_hh, dec_filt, rec_filt) as
ptwt's, and the padding/cropping arithmetic is a verbatim copy of
ptwt._util's _get_pad / _adjust_padding_at_reconstruction. Diff this file
against ptwt/conv_transform_2.py to see exactly what changed:

1. Everything that depends only on (wavelet, img_size, level) -- filter
   kernels and per-level pad/crop amounts -- is computed ONCE (build_2d_
   filters / build_separable_filters / build_schedules), not on every
   call the way ptwt.wavedec2/waverec2 do. This matters most inside a
   CUDA-graph-replayed loop: any op that gets captured runs on every
   replay regardless of whether its inputs ever change.

2. wavedec2()/waverec2() (and their _separable counterparts) take and
   return a FLAT List[torch.Tensor] instead of ptwt's nested
   (approx, (lh,hl,hh), (lh,hl,hh), ...) tuple structure:

       [approx, lh_L, hl_L, hh_L, lh_{L-1}, hl_{L-1}, hh_{L-1}, ..., lh_1, hl_1, hh_1]

   Same ordering as ptwt, just flattened -- chunk after index 0 into
   groups of 3 to recover ptwt's tuples if needed. A flat, homogeneous
   List[Tensor] is a simpler pytree for torch.compile/torch.jit than a
   list mixing bare tensors and 3-tuples.

Known caveat, not introduced here: ptwt's own waverec2 was found to not
be bit-reproducible run-to-run for db3/db4 on at least one GPU/cuDNN
build (~1e-7; confirmed by calling plain ptwt.waverec2 twice on
identical input). That's cuDNN algorithm-selection behavior in
conv_transpose2d, present in ptwt itself -- this module calls the same
op and inherits it identically.
"""

# --------------------------------------------------------------------------- #
# Padding/cropping arithmetic -- verbatim copies of ptwt._util's
# _get_pad / _adjust_padding_at_reconstruction (pure ints, no tensors),
# used to precompute the whole per-level schedule once instead of
# recomputing it from .shape on every call.
# --------------------------------------------------------------------------- #
def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    padl = padr = (2 * filt_len - 3) // 2
    padr += data_len % 2
    return padl, padr


def _adjust_padding_at_reconstruction(
    res_size: int, coeff_size: int, pad_end: int, pad_start: int
) -> Tuple[int, int]:
    pred_size = res_size - (pad_start + pad_end)
    if coeff_size == pred_size:
        pass
    elif coeff_size == pred_size - 1:
        pad_end += 1
    else:
        raise AssertionError(
            "padding error: predicted reconstruction size does not match the "
            "target coefficient size (mirrors ptwt's own check of the same name)."
        )
    return pad_end, pad_start


def build_schedules(
    img_size: Tuple[int, int], filt_len: int, level: int
) -> Tuple[Schedule, Schedule, List[Tuple[int, int]]]:
    """Precompute, for a FIXED (img_size, filt_len, level, mode="zero"):

      fwd_schedule[i]  = (padl, padr, padt, padb) forward-pad amounts at
                         decomposition step i (i=0 finest .. level-1 coarsest)
      crop_schedule[c] = (padl, padr, padt, padb) crop amounts in the
                         reconstruction loop at position c (c=0 coarsest ..
                         level-1 finest) -- same order ptwt.waverec2 uses
      full_shapes[i]   = (h, w) after i forward decompositions, i=0..level;
                         full_shapes[0] is the original img_size

    Used by both Wavelet2D and SeparableWavelet2D -- the pad/crop amounts
    are computed independently per axis, so both dense and separable
    convolutions share this exact schedule.
    """
    h, w = img_size
    full_shapes: List[Tuple[int, int]] = [(h, w)]
    fwd_schedule: Schedule = []
    cur_h, cur_w = h, w
    for _ in range(level):
        padl, padr = _get_pad(cur_w, filt_len)
        padt, padb = _get_pad(cur_h, filt_len)
        fwd_schedule.append((padl, padr, padt, padb))
        cur_h = (cur_h + padt + padb - filt_len) // 2 + 1
        cur_w = (cur_w + padl + padr - filt_len) // 2 + 1
        full_shapes.append((cur_h, cur_w))

    base = (2 * filt_len - 3) // 2
    crop_schedule: Schedule = []
    for c_pos in range(level):
        entering_h, entering_w = full_shapes[level - c_pos]
        padl = padr = padt = padb = base
        if c_pos < level - 1:
            raw_w = (entering_w - 1) * 2 + filt_len
            raw_h = (entering_h - 1) * 2 + filt_len
            target_h, target_w = full_shapes[level - c_pos - 1]
            padr, padl = _adjust_padding_at_reconstruction(raw_w, target_w, padr, padl)
            padb, padt = _adjust_padding_at_reconstruction(raw_h, target_h, padb, padt)
        crop_schedule.append((padl, padr, padt, padb))

    return fwd_schedule, crop_schedule, full_shapes


# --------------------------------------------------------------------------- #
# Filter construction -- shared 1D-tap loading, then either the dense
# rank-2 outer-product filter (ptwt._construct_2d_filt, unchanged) or the
# separable rank-1 grouped-conv weights.
# --------------------------------------------------------------------------- #
def _get_1d_filters(wavelet, device: torch.device, dtype: torch.dtype):
    """The 1D filter taps ptwt itself starts from before building 2D
    filters. `wavelet` can be a string, pywt.Wavelet, or a tuple/
    WaveletTensorTuple of (dec_lo, dec_hi, rec_lo, rec_hi) -- anything
    ptwt._util._get_filter_tensors already accepts. Returns
    (dec_lo, dec_hi, rec_lo, rec_hi, filt_len), each a 1D tensor."""
    dec_lo, dec_hi, _, _ = _get_filter_tensors(wavelet, flip=True, device=device, dtype=dtype)
    _, _, rec_lo, rec_hi = _get_filter_tensors(wavelet, flip=False, device=device, dtype=dtype)
    filt_len = int(dec_lo.shape[-1])
    return dec_lo.reshape(-1), dec_hi.reshape(-1), rec_lo.reshape(-1), rec_hi.reshape(-1), filt_len


def build_2d_filters(
    wavelet, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (dec_filt, rec_filt, filt_len), each built exactly as
    ptwt._construct_2d_filt does -- values guaranteed identical to what
    ptwt.wavedec2/waverec2 build internally, just built once instead of
    on every call."""
    dec_lo, dec_hi, rec_lo, rec_hi, filt_len = _get_1d_filters(wavelet, device, dtype)
    dec_filt = _construct_2d_filt(dec_lo, dec_hi).contiguous()
    rec_filt = _construct_2d_filt(rec_lo, rec_hi).contiguous()
    return dec_filt, rec_filt, filt_len


def build_separable_filters(wavelet, device: torch.device, dtype: torch.dtype):
    """Returns (w_dec_width, w_dec_height, w_rec_height, w_rec_width, filt_len, rec_lo, rec_hi):

      w_dec_width  [2,1,1,L] : forward stage 1 (width analysis, lo/hi)
      w_dec_height [4,1,L,1] : forward stage 2 (height analysis, grouped=2)
      w_rec_height [4,1,L,1] : inverse stage 1 (height synthesis, grouped=2)
      w_rec_width  [2,1,1,L] : inverse stage 2 (width synthesis)

    The channel order these produce/consume (ll, lh, hl, hh) matches
    ptwt._construct_2d_filt's stacking order exactly. rec_lo/rec_hi (the
    raw 1D taps, length L) are also returned: waverec2_separable needs
    them directly when L==2, see its docstring.
    """
    dec_lo, dec_hi, rec_lo, rec_hi, filt_len = _get_1d_filters(wavelet, device, dtype)
    w_dec_width = torch.stack([dec_lo, dec_hi], dim=0).reshape(2, 1, 1, filt_len).contiguous()
    w_dec_height = torch.stack([dec_lo, dec_hi, dec_lo, dec_hi], dim=0).reshape(4, 1, filt_len, 1).contiguous()
    w_rec_height = torch.stack([rec_lo, rec_hi, rec_lo, rec_hi], dim=0).reshape(4, 1, filt_len, 1).contiguous()
    w_rec_width = torch.stack([rec_lo, rec_hi], dim=0).reshape(2, 1, 1, filt_len).contiguous()
    return w_dec_width, w_dec_height, w_rec_height, w_rec_width, filt_len, rec_lo, rec_hi


# --------------------------------------------------------------------------- #
# Dense hot path. Compare against ptwt.conv_transform_2.wavedec2/
# waverec2's per-level loops -- padding amounts and dec_filt/rec_filt are
# arguments instead of recomputed every call, and detail coefficients are
# appended flat instead of as a WaveletDetailTuple2d. Otherwise unchanged.
#
# Plain functions of (tensor, precomputed tensor, precomputed python list
# of int-tuples) -- suitable for torch.compile(wavedec2)/
# torch.compile(waverec2) directly, and safe inside torch.cuda.graph(...)
# capture.
# --------------------------------------------------------------------------- #
def wavedec2(data: torch.Tensor, dec_filt: torch.Tensor, fwd_schedule: Schedule) -> List[torch.Tensor]:
    """mode='zero' wavedec2 for a fixed shape/wavelet/level.

    `data`: any [..., H, W] tensor; leading dims are folded into one
    batch dim for the convolution and unfolded back on the way out.

    Returns a flat list: [approx, lh_L, hl_L, hh_L, ..., lh_1, hl_1, hh_1].
    """
    leading_shape = data.shape[:-2]
    res_ll = data.reshape(-1, 1, *data.shape[-2:])
    result_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for padl, padr, padt, padb in fwd_schedule:
        res_ll = F.pad(res_ll, (padl, padr, padt, padb))         # ptwt: res_ll = _fwt_pad2(res_ll, wavelet, mode=mode)
        res = F.conv2d(res_ll, dec_filt, stride=2)                # unchanged from ptwt
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)   # unchanged from ptwt
        res_ll = res_ll.contiguous()  # avoids a hidden re-layout cost next level (split's output is a strided view)
        result_lst.append((res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)))

    result_lst.reverse()          # unchanged from ptwt
    res_ll = res_ll.squeeze(1)    # unchanged from ptwt

    flat: List[torch.Tensor] = [res_ll.reshape(*leading_shape, *res_ll.shape[-2:])]
    for lh, hl, hh in result_lst:
        hw = lh.shape[-2:]
        flat.append(lh.reshape(*leading_shape, *hw))
        flat.append(hl.reshape(*leading_shape, *hw))
        flat.append(hh.reshape(*leading_shape, *hw))
    return flat


def waverec2(coeffs: List[torch.Tensor], rec_filt: torch.Tensor, crop_schedule: Schedule) -> torch.Tensor:
    """mode='zero' waverec2 for a fixed shape/wavelet/level.

    `coeffs`: flat list in the format wavedec2() above returns.
    """
    leading_shape = coeffs[0].shape[:-2]
    res_ll = coeffs[0].reshape(-1, *coeffs[0].shape[-2:])

    for c_pos, (padl, padr, padt, padb) in enumerate(crop_schedule):
        res_lh, res_hl, res_hh = coeffs[1 + 3 * c_pos : 4 + 3 * c_pos]
        hw = res_lh.shape[-2:]
        res_ll = torch.stack([
            res_ll,
            res_lh.reshape(-1, *hw),
            res_hl.reshape(-1, *hw),
            res_hh.reshape(-1, *hw),
        ], 1)  # unchanged from ptwt
        res_ll = F.conv_transpose2d(res_ll, rec_filt, stride=2).squeeze(1)  # unchanged from ptwt

        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]

    return res_ll.reshape(*leading_shape, *res_ll.shape[-2:])


class Wavelet2D:
    """Precomputes filters and the pad/crop schedule once for a fixed
    (wavelet, img_size, level, device, dtype), then .decompose()/
    .reconstruct() only ever do padding, conv2d/conv_transpose2d, and
    slicing.

    Construction mirrors ptwt.wavedec2's call signature, to make it
    obvious what's cached here vs. recomputed on every ptwt call:

        ptwt.wavedec2(data, wavelet, level=level, mode="zero")
        Wavelet2D(wavelet, img_size, level, device, dtype).decompose(data)
    """

    def __init__(
        self,
        wavelet,
        img_size: Tuple[int, int],
        level: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str = "zero",
    ) -> None:
        if mode != "zero":
            raise NotImplementedError(
                "only mode='zero' is implemented here (it's the only mode "
                "ProxOpSARAPos uses); ptwt.wavedec2 supports more."
            )
        self.dec_filt, self.rec_filt, self.filt_len = build_2d_filters(wavelet, device, dtype)
        self.fwd_schedule, self.crop_schedule, self.full_shapes = build_schedules(
            img_size, self.filt_len, level
        )

    def decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        return wavedec2(x, self.dec_filt, self.fwd_schedule)

    def reconstruct(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        return waverec2(coeffs, self.rec_filt, self.crop_schedule)


# --------------------------------------------------------------------------- #
# Separable (row-then-column) hot path: same per-level structure as
# above, but the O(L^2) 2D convolution is factored into two O(L) 1D
# convolutions (ptwt's 2D filters are rank-1 outer products of 1D
# filters -- see _construct_2d_filt). Matters most once you're
# compute-bound rather than launch-overhead-bound: inside a CUDA graph
# replay, launch overhead is already eliminated for both this and
# Wavelet2D, so raw FLOPs are what's left to cut.
# --------------------------------------------------------------------------- #
def wavedec2_separable(
    data: torch.Tensor,
    w_dec_width: torch.Tensor,
    w_dec_height: torch.Tensor,
    fwd_schedule: Schedule,
) -> List[torch.Tensor]:
    """mode='zero' wavedec2, separable version. Same flat-list output
    format as wavedec2() above; reuses the same fwd_schedule."""
    leading_shape = data.shape[:-2]
    res_ll = data.reshape(-1, 1, *data.shape[-2:])
    result_lst: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for padl, padr, padt, padb in fwd_schedule:
        x_wpad = F.pad(res_ll, (padl, padr, 0, 0))
        stage1 = F.conv2d(x_wpad, w_dec_width, stride=(1, 2))               # [N,2,H,w] = width_lo, width_hi
        stage1_hpad = F.pad(stage1, (0, 0, padt, padb))
        res = F.conv2d(stage1_hpad, w_dec_height, stride=(2, 1), groups=2)  # [N,4,h,w] = ll,lh,hl,hh

        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
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


def _transpose_conv1d_2tap(
    x0: torch.Tensor, x1: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, dim: int
) -> torch.Tensor:
    """Exact equivalent of a 2-input-channel, 1-output-channel,
    kernel_size=2, stride=2, no-padding transposed convolution along
    `dim`, computed with elementwise ops + reshape instead of
    conv_transpose2d. w0, w1 are length-2 1D tensors (the two filter taps
    for x0 and x1 respectively); the size along `dim` doubles.

    With no padding and kernel_size==stride, there's no overlap between
    adjacent output windows -- each output pair depends on exactly one
    input element, so this reduces to a linear combination plus
    interleaving rather than a genuine convolution. See
    waverec2_separable's docstring for why this matters.
    """
    a = x0 * w0[0] + x1 * w1[0]
    b = x0 * w0[1] + x1 * w1[1]
    d = dim % a.ndim
    stacked = torch.stack([a, b], dim=d + 1)
    new_shape = list(a.shape)
    new_shape[d] *= 2
    return stacked.reshape(new_shape)


def waverec2_separable(
    coeffs: List[torch.Tensor],
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    w_rec_height: torch.Tensor,
    w_rec_width: torch.Tensor,
    crop_schedule: Schedule,
) -> torch.Tensor:
    """mode='zero' waverec2, separable version. Same flat-list input
    format as waverec2() above.

    For filt_len==2 (db1/Haar), height/width synthesis uses
    _transpose_conv1d_2tap instead of conv_transpose2d: a
    conv_transpose2d call with kernel_size==stride==2 was found to give
    wrong results (~1e-3, not floating-point noise) specifically at
    filt_len=2 on at least one GPU/cuDNN build -- filt_len>=4 was never
    affected, including after de-grouping the same call, which pointed
    at kernel_size==stride rather than grouping as the trigger.
    _transpose_conv1d_2tap computes the mathematically identical result
    (see its docstring) without calling conv_transpose2d at all, so it
    never exercises that code path. w_rec_height/w_rec_width are unused
    when filt_len==2 but stay in the signature so every wavelet is called
    the same way.

    NOT independently re-confirmed on real hardware at the time of
    writing -- validate with validate_and_benchmark.py (db1 is included
    in its correctness sweep) before trusting this for filt_len==2.
    """
    filt_len = rec_lo.shape[-1]
    leading_shape = coeffs[0].shape[:-2]
    res_ll = coeffs[0].reshape(-1, *coeffs[0].shape[-2:])

    for c_pos, (padl, padr, padt, padb) in enumerate(crop_schedule):
        res_lh, res_hl, res_hh = coeffs[1 + 3 * c_pos : 4 + 3 * c_pos]
        hw = res_lh.shape[-2:]
        res_lh_f = res_lh.reshape(-1, *hw)
        res_hl_f = res_hl.reshape(-1, *hw)
        res_hh_f = res_hh.reshape(-1, *hw)

        if filt_len == 2:
            width_lo_domain = _transpose_conv1d_2tap(res_ll, res_lh_f, rec_lo, rec_hi, dim=1)
            width_hi_domain = _transpose_conv1d_2tap(res_hl_f, res_hh_f, rec_lo, rec_hi, dim=1)
            if padt > 0:
                width_lo_domain = width_lo_domain[..., padt:, :]
                width_hi_domain = width_hi_domain[..., padt:, :]
            if padb > 0:
                width_lo_domain = width_lo_domain[..., :-padb, :]
                width_hi_domain = width_hi_domain[..., :-padb, :]
            res_ll = _transpose_conv1d_2tap(width_lo_domain, width_hi_domain, rec_lo, rec_hi, dim=2)
            if padl > 0:
                res_ll = res_ll[..., padl:]
            if padr > 0:
                res_ll = res_ll[..., :-padr]
        else:
            stacked = torch.stack([res_ll, res_lh_f, res_hl_f, res_hh_f], 1)  # [N,4,h,w]
            stage_a = F.conv_transpose2d(stacked, w_rec_height, stride=(2, 1), groups=2)  # [N,2,raw_h,w]
            if padt > 0:
                stage_a = stage_a[..., padt:, :]
            if padb > 0:
                stage_a = stage_a[..., :-padb, :]

            stage_b = F.conv_transpose2d(stage_a, w_rec_width, stride=(1, 2))  # [N,1,h,raw_w]
            if padl > 0:
                stage_b = stage_b[..., padl:]
            if padr > 0:
                stage_b = stage_b[..., :-padr]
            res_ll = stage_b.squeeze(1)

    return res_ll.reshape(*leading_shape, *res_ll.shape[-2:])


class SeparableWavelet2D:
    """Row-then-column variant of Wavelet2D: O(L) work per level instead
    of O(L^2). Same construction/call pattern as Wavelet2D.

    filt_len==2 (db1/Haar) is handled inside waverec2_separable via
    _transpose_conv1d_2tap rather than conv_transpose2d -- see that
    function's docstring for why, and for the caveat that this hasn't
    been independently re-confirmed on real hardware. If it turns out to
    still be wrong for db1 in your testing, use Wavelet2D directly for
    that wavelet (it's unaffected: bit-exact vs. ptwt, validated across
    every round of testing in this file's history).
    """

    def __init__(
        self,
        wavelet,
        img_size: Tuple[int, int],
        level: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        (self.w_dec_width, self.w_dec_height, self.w_rec_height, self.w_rec_width,
         filt_len, self.rec_lo, self.rec_hi) = build_separable_filters(wavelet, device, dtype)
        self.fwd_schedule, self.crop_schedule, self.full_shapes = build_schedules(
            img_size, filt_len, level
        )

    def decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        return wavedec2_separable(x, self.w_dec_width, self.w_dec_height, self.fwd_schedule)

    def reconstruct(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        return waverec2_separable(
            coeffs, self.rec_lo, self.rec_hi, self.w_rec_height, self.w_rec_width, self.crop_schedule
        )


def make_wavelet_transform(
    wavelet,
    img_size: Tuple[int, int],
    level: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """The function to actually use. Returns SeparableWavelet2D for every
    wavelet, including db1 (see SeparableWavelet2D's docstring for the
    filt_len==2 caveat)."""
    return SeparableWavelet2D(wavelet, img_size, level, device, dtype)