import torch
import torch.nn.functional as F
import pywt
from typing import Tuple, List
import ptwt

class CompiledWaveletBank(torch.nn.Module):
    def __init__(self, wl_dict: List[str], dec_lev: int, img_size: Tuple[int, int], mode: str, device: torch.device, dtype: torch.dtype, dirac: bool):
        super().__init__()
        self.dec_lev = dec_lev
        self.img_size = img_size
        self.wl_dict = wl_dict
        self.num_bases = len(wl_dict)
        self.dirac = dirac
        self.pixels_per_basis = img_size[0] * img_size[1]
        
        # Set padding mode
        if mode != "zero":
            print("INFO: Wavelet padding mode not supported! Using zero padding instead!")
        
        # Determine the maximum filter length to unify allocations
        self.max_K = max(len(pywt.Wavelet(b).dec_lo) for b in wl_dict)
        self.pad = (self.max_K - 2) // 2
        
        # Meta-information for tracking flat splits safely during torch.compile
        self._setup_split_sizes()

        # Initialize parameter tensors for grouped 1D convolutions
        self.row_dec_filters = torch.nn.Parameter(torch.zeros((2 * self.num_bases, 1, 1, self.max_K), dtype=dtype, device=device), requires_grad=False)
        self.col_dec_filters = torch.nn.Parameter(torch.zeros((4 * self.num_bases, 1, self.max_K, 1), dtype=dtype, device=device), requires_grad=False)
        
        self.col_rec_filters = torch.nn.Parameter(torch.zeros((4 * self.num_bases, 1, self.max_K, 1), dtype=dtype, device=device), requires_grad=False)
        self.row_rec_filters = torch.nn.Parameter(torch.zeros((2 * self.num_bases, 1, 1, self.max_K), dtype=dtype, device=device), requires_grad=False)

        self._compile_filter_banks(wl_dict, dtype, device)

    def _setup_split_sizes(self):
        """Precomputes exact segment sizes for compiler-friendly splitting."""
        self.split_sizes = []
        H, W = self.img_size
        
        # Details sizes per level (3 subbands: LH, HL, HH)
        for lev in range(self.dec_lev):
            level_h, level_w = H // (2**(lev+1)), W // (2**(lev+1))
            self.split_sizes.append(self.num_bases * 3 * level_h * level_w)
            
        # Deepest LL size
        deep_h, deep_w = H // (2**self.dec_lev), W // (2**self.dec_lev)
        self.split_sizes.append(self.num_bases * deep_h * deep_w)
        
        if self.dirac:
            self.split_sizes.append(self.pixels_per_basis)

    def _compile_filter_banks(self, wl_dict: List[str], dtype: torch.dtype, device: torch.device):
        """Pads and packs all 1D filters into grouped configuration matrices."""
        for i, basis_str in enumerate(wl_dict):
            wavelet = pywt.Wavelet(basis_str)
            
            # Load raw 1D filters
            dec_lo = torch.tensor(wavelet.dec_lo, dtype=dtype, device=device)
            dec_hi = torch.tensor(wavelet.dec_hi, dtype=dtype, device=device)
            rec_lo = torch.tensor(wavelet.rec_lo, dtype=dtype, device=device)
            rec_hi = torch.tensor(wavelet.rec_hi, dtype=dtype, device=device)
            
            # Left-pad filters with zeros to align center-phase uniformly
            # pad_left = (self.max_K - len(dec_lo)) // 2
            # pad_right = self.max_K - len(dec_lo) - pad_left
            pad_left = self.max_K - len(dec_lo)
            pad_right = 0
            
            d_lo = F.pad(torch.flip(dec_lo, [0]), (pad_left, pad_right))
            d_hi = F.pad(torch.flip(dec_hi, [0]), (pad_left, pad_right))
            r_lo = F.pad(rec_lo, (pad_left, pad_right))
            r_hi = F.pad(rec_hi, (pad_left, pad_right))

            # 1. Row Decomposition Layout
            self.row_dec_filters.data[2*i, 0, 0, :] = d_lo
            self.row_dec_filters.data[2*i+1, 0, 0, :] = d_hi

            # 2. Column Decomposition Layout
            self.col_dec_filters.data[4*i, 0, :, 0] = d_lo
            self.col_dec_filters.data[4*i+1, 0, :, 0] = d_hi
            self.col_dec_filters.data[4*i+2, 0, :, 0] = d_lo
            self.col_dec_filters.data[4*i+3, 0, :, 0] = d_hi

            # 3. Column Reconstruction Layout
            self.col_rec_filters.data[4*i, 0, :, 0] = r_lo
            self.col_rec_filters.data[4*i+1, 0, :, 0] = r_hi
            self.col_rec_filters.data[4*i+2, 0, :, 0] = r_lo
            self.col_rec_filters.data[4*i+3, 0, :, 0] = r_hi

            # 4. Row Reconstruction Layout
            self.row_rec_filters.data[2*i, 0, 0, :] = r_lo
            self.row_rec_filters.data[2*i+1, 0, 0, :] = r_hi

    def decompose_flat(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate input image across all bases into a single grouped channel dimension
        curr_x = x.repeat(1, self.num_bases, 1, 1) 
        
        flat_components = []
        H, W = self.img_size

        for lev in range(self.dec_lev):
            level_h, level_w = H // (2**lev), W // (2**lev)
            
            # Symmetrical image padding for the unified filter length
            curr_x = F.pad(curr_x, (self.pad, self.pad, self.pad, self.pad), mode='circular')
            # curr_x = F.pad(curr_x, (self.pad, self.pad, self.pad, self.pad), mode='zero')
            
            # Row Pass: Filter horizontally, downsample columns
            row_out = F.conv2d(curr_x, self.row_dec_filters, stride=(1, 2), groups=self.num_bases)
            
            # Column Pass: Filter vertically, downsample rows
            col_out = F.conv2d(row_out, self.col_dec_filters, stride=(2, 1), groups=2 * self.num_bases)
            
            # Reshape out to separate subbands: (1, Num_Bases, 4 [LL, LH, HL, HH], H//2, W//2)
            bands = col_out.view(1, self.num_bases, 4, level_h // 2, level_w // 2)
            
            curr_x = bands[:, :, 0, :, :] # Next iteration's low frequency (LL)
            details = bands[:, :, 1:, :, :] # LH, HL, HH coefficients
            flat_components.append(details.reshape(-1))

        # Append the deepest LL coefficients
        flat_components.append(curr_x.reshape(-1))
        
        if self.dirac:
            flat_components.append(x.reshape(-1))

        # Optimization 4: Simple, compiler-friendly concatenation
        return torch.cat(flat_components)

    def reconstruct_from_flat(self, flat_coeffs: torch.Tensor) -> torch.Tensor:
        # Optimization 4: Slice the flat vector without in-place tensor mutations
        chunks = torch.split(flat_coeffs, self.split_sizes)
        
        H, W = self.img_size
        deep_h, deep_w = H // (2**self.dec_lev), W // (2**self.dec_lev)
        
        # Pull the deepest LL state
        curr_LL = chunks[self.dec_lev].view(1, self.num_bases, deep_h, deep_w)
        
        for lev in range(self.dec_lev - 1, -1, -1):
            level_h, level_w = H // (2**lev), W // (2**lev)
            
            # Pull corresponding detail chunk
            details = chunks[lev].view(1, self.num_bases, 3, level_h // 2, level_w // 2)
            
            # Merge into a grouped 4-channel representation [LL, LH, HL, HH]
            sub_bands = torch.cat([curr_LL.unsqueeze(2), details], dim=2)
            sub_bands = sub_bands.view(1, 4 * self.num_bases, level_h // 2, level_w // 2)
            
            # Column Transpose Pass (Up-sample vertically)
            col_rec = F.conv_transpose2d(sub_bands, self.col_rec_filters, stride=(2, 1), padding=(self.pad, 0), groups=2 * self.num_bases)
            
            # Row Transpose Pass (Up-sample horizontally)
            curr_LL = F.conv_transpose2d(col_rec, self.row_rec_filters, stride=(1, 2), padding=(0, self.pad), groups=self.num_bases)

        # Reduce back by summing across the basis channels
        result = torch.sum(curr_LL, dim=1, keepdim=True)
        
        if self.dirac:
            result += chunks[-1].view(1, 1, H, W)
            
        return result