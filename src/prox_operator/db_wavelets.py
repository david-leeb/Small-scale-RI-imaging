"""
Batched multi-basis wavelet bank.

Supported modes
---------------
'periodization' (recommended, default):
    Matches ptwt/pywt mode='periodization' exactly.
    Right-wraps K-2 samples per 1-D pass.
    Output length = N//2 for any even N and any K — uniform split_sizes.
    Safe wrapping via index_select avoids PyTorch's circular-pad overflow
    restriction (pad >= signal_length) at deep decomposition levels.

'zero':
    Cannot be matched with uniform split_sizes because pywt zero mode gives
    output length (N-1)//2 + K//2 which grows with K. Matching it exactly
    would require per-basis variable-size coefficients, destroying the flat
    vector representation. Use 'periodization' and pass mode='periodization'
    to ProxOpSARAPos_original if you need coefficient-level comparison.

Subband storage order (matches ptwt tuple ordering)
----------------------------------------------------
pywt/ptwt returns (cH, cV, cD) per level where:
    cH = Lo(rows) * Hi(cols)   [horizontal / col-Hi]
    cV = Hi(rows) * Lo(cols)   [vertical   / row-Hi]
    cD = Hi(rows) * Hi(cols)   [diagonal]
In the flat vector each level stores [cH_0, cV_0, cD_0, cH_1, cV_1, cD_1, ...].

Speed gains over the original sequential ptwt loop
----------------------------------------------------
* Filters pre-loaded on GPU as Parameters — zero per-call allocation.
* LL states held as a plain Python list of same-device tensors.
* Fixed flat-vector layout (split_sizes) computed once at construction.
* No pywt/ptwt Python overhead (format conversion, boundary dispatch, etc.).
"""

import torch
import torch.nn.functional as F
import pywt
from typing import List, Tuple

import ptwt  # kept for API compatibility; not called at runtime


class CompiledWaveletBank(torch.nn.Module):
    def __init__(
        self,
        wl_dict:  List[str],
        dec_lev:  int,
        img_size: Tuple[int, int],
        mode:     str,            # 'periodization' recommended; 'zero' falls back to periodization
        device:   torch.device,
        dtype:    torch.dtype,
        dirac:    bool,
    ) -> None:
        super().__init__()
        self.dec_lev   = dec_lev
        self.img_size  = img_size
        self.wl_dict   = wl_dict
        self.num_bases = len(wl_dict)
        self.dirac     = dirac
        self.pixels_per_basis = img_size[0] * img_size[1]

        if mode != "periodization":
            print(
                f"INFO: mode='{mode}' cannot be matched with uniform split_sizes. "
                "Using 'periodization'. Pass mode='periodization' to "
                "ProxOpSARAPos_original for a fair comparison."
            )
        self.mode = "periodization"

        # Per-basis metadata --------------------------------------------------
        self.filter_lens: List[int] = []
        # dec_right[i] = K-2: right-wrap amount for periodization
        #   Proof: out = (N + (K-2) - K)//2 + 1 = (N-2)//2 + 1 = N//2  ✓  (even N)
        self.dec_right:   List[int] = []
        # rec_pad[i] = (K-2)//2: F.conv_transpose2d padding giving N from N//2
        #   Proof: (N//2-1)*2 + K - 2*((K-2)//2) = N-2+K-(K-2) = N  ✓  (even K)
        self.rec_pad:     List[int] = []

        # Filter banks: one ParameterList per basis ---------------------------
        # Shape conventions:
        #   row filters: (1, 1, 1, K)  — applied along W (width/cols)
        #   col filters: (1, 1, K, 1)  — applied along H (height/rows)
        # Stored WITHOUT flipping: F.conv2d cross-correlation matches pywt's
        # correlation convention directly.
        self.row_dec: torch.nn.ModuleList = torch.nn.ModuleList()
        self.col_dec: torch.nn.ModuleList = torch.nn.ModuleList()
        self.col_rec: torch.nn.ModuleList = torch.nn.ModuleList()
        self.row_rec: torch.nn.ModuleList = torch.nn.ModuleList()

        def _p(arr, shape):
            return torch.nn.Parameter(
                torch.tensor(arr, dtype=dtype, device=device).view(shape),
                requires_grad=False,
            )

        for basis_str in wl_dict:
            wv = pywt.Wavelet(basis_str)
            K  = len(wv.dec_lo)
            self.filter_lens.append(K)
            self.dec_right.append(K - 2)
            self.rec_pad.append((K - 2) // 2)

            self.row_dec.append(torch.nn.ParameterList([
                _p(wv.dec_lo, (1, 1, 1, K)), _p(wv.dec_hi, (1, 1, 1, K)),
            ]))
            self.col_dec.append(torch.nn.ParameterList([
                _p(wv.dec_lo, (1, 1, K, 1)), _p(wv.dec_hi, (1, 1, K, 1)),
            ]))
            self.col_rec.append(torch.nn.ParameterList([
                _p(wv.rec_lo, (1, 1, K, 1)), _p(wv.rec_hi, (1, 1, K, 1)),
            ]))
            self.row_rec.append(torch.nn.ParameterList([
                _p(wv.rec_lo, (1, 1, 1, K)), _p(wv.rec_hi, (1, 1, 1, K)),
            ]))

        self._setup_split_sizes()

    # ------------------------------------------------------------------ #
    # Flat coefficient layout                                             #
    # Level 0 (finest) … dec_lev-1 (coarsest) detail, deepest LL, dirac #
    # Within each level: [cH_0, cV_0, cD_0,  cH_1, cV_1, cD_1, …]     #
    # ------------------------------------------------------------------ #
    def _setup_split_sizes(self) -> None:
        self.split_sizes: List[int] = []
        H, W = self.img_size
        for lev in range(self.dec_lev):
            lh = H // (2 ** (lev + 1))
            lw = W // (2 ** (lev + 1))
            self.split_sizes.append(self.num_bases * 3 * lh * lw)
        dh = H // (2 ** self.dec_lev)
        dw = W // (2 ** self.dec_lev)
        self.split_sizes.append(self.num_bases * dh * dw)
        if self.dirac:
            self.split_sizes.append(self.pixels_per_basis)

    # ------------------------------------------------------------------ #
    # Safe right-circular-wrap (works even when amount >= signal length)  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap_right(x: torch.Tensor, amount: int, dim: int) -> torch.Tensor:
        if amount == 0:
            return x
        N = x.shape[dim]
        idx = torch.arange(amount, device=x.device) % N
        return torch.cat([x, x.index_select(dim, idx)], dim=dim)

    # ------------------------------------------------------------------ #
    # 1-D analysis                                                        #
    # ------------------------------------------------------------------ #
    def _row_dec_i(self, x: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-wrap along W, then stride-2 conv with Lo and Hi row filters."""
        xp = self._wrap_right(x, self.dec_right[i], dim=3)
        return (F.conv2d(xp, self.row_dec[i][0], stride=(1, 2)),
                F.conv2d(xp, self.row_dec[i][1], stride=(1, 2)))

    def _col_dec_i(self, x: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-wrap along H, then stride-2 conv with Lo and Hi col filters."""
        xp = self._wrap_right(x, self.dec_right[i], dim=2)
        return (F.conv2d(xp, self.col_dec[i][0], stride=(2, 1)),
                F.conv2d(xp, self.col_dec[i][1], stride=(2, 1)))

    # ------------------------------------------------------------------ #
    # 1-D synthesis                                                       #
    # ------------------------------------------------------------------ #
    def _col_rec_i(self, lo: torch.Tensor, hi: torch.Tensor, i: int) -> torch.Tensor:
        rp = self.rec_pad[i]
        return (F.conv_transpose2d(lo, self.col_rec[i][0], stride=(2, 1), padding=(rp, 0))
              + F.conv_transpose2d(hi, self.col_rec[i][1], stride=(2, 1), padding=(rp, 0)))

    def _row_rec_i(self, lo: torch.Tensor, hi: torch.Tensor, i: int) -> torch.Tensor:
        rp = self.rec_pad[i]
        return (F.conv_transpose2d(lo, self.row_rec[i][0], stride=(1, 2), padding=(0, rp))
              + F.conv_transpose2d(hi, self.row_rec[i][1], stride=(1, 2), padding=(0, rp)))

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def decompose_flat(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, 1, H, W)  →  1-D flat coefficient vector."""
        flat_parts: List[torch.Tensor] = []
        curr_ll: List[torch.Tensor] = [x] * self.num_bases

        for lev in range(self.dec_lev):
            level_details: List[torch.Tensor] = []
            next_ll:        List[torch.Tensor] = []

            for i in range(self.num_bases):
                # Row pass along W: lo_row = Lo(cols), hi_row = Hi(cols)
                lo_row, hi_row = self._row_dec_i(curr_ll[i], i)

                # Col pass on Lo-row along H:
                #   lo_col(lo_row) = Lo(rows)*Lo(cols) = LL
                #   hi_col(lo_row) = Hi(rows)*Lo(cols) = cV  (vertical detail)
                ll_i, cV_i = self._col_dec_i(lo_row, i)

                # Col pass on Hi-row along H:
                #   lo_col(hi_row) = Lo(rows)*Hi(cols) = cH  (horizontal detail)
                #   hi_col(hi_row) = Hi(rows)*Hi(cols) = cD  (diagonal)
                cH_i, cD_i = self._col_dec_i(hi_row, i)

                next_ll.append(ll_i)
                # Store in pywt tuple order: (cH, cV, cD)
                level_details.append(torch.cat([cH_i, cV_i, cD_i], dim=1))

            curr_ll = next_ll
            flat_parts.append(torch.stack(level_details, dim=1).reshape(-1))

        flat_parts.append(torch.stack(curr_ll, dim=1).reshape(-1))
        if self.dirac:
            flat_parts.append(x.reshape(-1))

        return torch.cat(flat_parts)

    def reconstruct_from_flat(self, flat_coeffs: torch.Tensor) -> torch.Tensor:
        H, W = self.img_size
        dh   = H // (2 ** self.dec_lev)
        dw   = W // (2 ** self.dec_lev)
        chunks = torch.split(flat_coeffs, self.split_sizes)

        ll_all  = chunks[self.dec_lev].view(self.num_bases, 1, dh, dw)
        curr_ll: List[torch.Tensor] = [ll_all[i : i + 1] for i in range(self.num_bases)]

        for lev in range(self.dec_lev - 1, -1, -1):
            lh  = H // (2 ** lev)
            lw  = W // (2 ** lev)
            det = chunks[lev].view(self.num_bases, 3, lh // 2, lw // 2)
            next_ll: List[torch.Tensor] = []

            for i in range(self.num_bases):
                # Unpack in pywt order: (cH, cV, cD)
                cH_i = det[i : i + 1, 0:1]   # Lo(rows) * Hi(cols) -> feeds hi_row recon
                cV_i = det[i : i + 1, 1:2]   # Hi(rows) * Lo(cols) -> feeds lo_row recon
                cD_i = det[i : i + 1, 2:3]   # Hi(rows) * Hi(cols) -> feeds hi_row recon

                # Col synthesis:
                #   lo_row = LL + cV  (Lo-col branch: ll and cV came from lo_row)
                #   hi_row = cH + cD  (Hi-col branch: cH and cD came from hi_row)
                lo_row = self._col_rec_i(curr_ll[i], cV_i, i)
                hi_row = self._col_rec_i(cH_i,       cD_i, i)

                # Row synthesis
                next_ll.append(self._row_rec_i(lo_row, hi_row, i))

            curr_ll = next_ll

        result = curr_ll[0].clone()
        for i in range(1, self.num_bases):
            result = result + curr_ll[i]

        if self.dirac:
            result = result + chunks[-1].view(1, 1, H, W)

        return result