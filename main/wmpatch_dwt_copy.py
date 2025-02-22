import numpy as np
from scipy.stats import norm
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from diffusers.utils.torch_utils import randn_tensor

class GTWatermark():
    def __init__(self, device, shape=(1,4,64,64), dtype=torch.float32, w_channel=3, w_radius=10, generator=None):
        self.device = device
        # from latent tensor
        self.shape = shape
        self.dtype = dtype
        # from hyperparameters
        self.w_channel = w_channel
        self.w_radius = w_radius

        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        self.mu, self.sigma = self.watermark_stat()

    def _circle_mask(self, size=64, r=10, x_offset=0, y_offset=0):
        # Reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y-y0)**2) <= r**2

    def _get_watermarking_pattern(self, gt_init):
        # Initialize DWT
        dwt = DWTForward(J=1, wave='haar').to(gt_init.device)

        # Perform DWT
        Yl, Yh = dwt(gt_init)

        # Yl is the low-frequency component
        # Yh is a list of high-frequency components at different scales

        # Assuming watermarking is applied to the first scale's high-frequency components
        # Yh[0] has shape (N, C, 3, H, W) corresponding to LH, HL, HH components

        # Apply the circular mask to the high-frequency components
        for i in range(self.w_radius, 0, -1):  # from outer circle to inner circle
            tmp_mask = torch.tensor(self._circle_mask(Yh[0].shape[-1], r=i)).to(self.device)  # circle mask in bool value
            for j in range(3):  # Apply mask to LH, HL, HH components
                Yh[0][:, self.w_channel, j, tmp_mask] = Yh[0][0, self.w_channel, j, 0, i].item()

        return Yl, Yh

    def _get_watermarking_mask(self, dwt_coeffs):
        """
        Generate a watermarking mask for DWT coefficients.

        Parameters:
        - dwt_coeffs: Tuple containing the low-frequency component (Yl) and a list of high-frequency components (Yh).

        Returns:
        - watermarking_mask: A mask with the same structure as dwt_coeffs, indicating where the watermark will be applied.
        """
        Yl, Yh = dwt_coeffs

        # Initialize masks for Yl and Yh
        Yl_mask = torch.zeros_like(Yl, dtype=torch.bool).to(self.device)
        Yh_masks = []

        for Yh_scale in Yh:
            scale_masks = []
            for subband in Yh_scale:
                # Create a mask for each subband
                subband_mask = torch.zeros_like(subband, dtype=torch.bool).to(self.device)
                # Apply the circular mask to the subband
                subband_mask[self.w_channel] = torch.tensor(
                    self._circle_mask(subband.shape[-1], r=self.w_radius)
                ).to(self.device)
                scale_masks.append(subband_mask)
            Yh_masks.append(scale_masks)

        # Combine Yl_mask and Yh_masks into a tuple
        watermarking_mask = (Yl_mask, Yh_masks)

        return watermarking_mask


    def _gen_gt(self, generator=None):
        # Generate initial random tensor
        gt_init = torch.randn(self.shape, generator=generator, device=self.device, dtype=self.dtype)

        # Initialize DWT
        dwt = DWTForward(J=1, wave='haar').to(self.device)

        # Perform DWT to obtain coefficients
        Yl, Yh = dwt(gt_init)

        # Generate watermarking pattern based on DWT coefficients
        gt_patch = self._get_watermarking_pattern((Yl, Yh))

        # Generate watermarking mask based on DWT coefficients
        watermarking_mask = self._get_watermarking_mask((Yl, Yh))

        return gt_patch, watermarking_mask

    def inject_watermark(self, latents):
        # Initialize DWT and inverse DWT
        dwt = DWTForward(J=1, wave='haar').to(latents.device)
        idwt = DWTInverse(wave='haar').to(latents.device)

        # Perform DWT
        Yl, Yh = dwt(latents)

        # Yl is the low-frequency component
        # Yh is a list of high-frequency components at different scales

        # Assuming watermarking is applied to the first scale's high-frequency components
        # Yh[0] has shape (N, C, 3, H, W) corresponding to LH, HL, HH components

        # Apply the watermarking mask to the high-frequency components
        for i in range(len(Yh[0])):
            Yh[0][:, self.w_channel, i] = (
                Yh[0][:, self.w_channel, i] * ~self.watermarking_mask
                + self.gt_patch * self.watermarking_mask
            )

        # Perform inverse DWT to reconstruct the watermarked latents
        latents_w = idwt((Yl, Yh))

        return latents_w


    def eval_watermark(self, latents_w):
        # Initialize the DWT
        dwt = DWTForward(J=1, wave='haar').to(latents_w.device)

        # Perform the DWT on the watermarked latents
        Yl_w, Yh_w = dwt(latents_w)

        # Compute the L1 metric for the high-frequency components
        l1_metric = 0.0
        count = 0

        for scale in range(len(Yh_w)):
            for subband in range(Yh_w[scale].shape[2]):
                # Extract the watermarked coefficients for the current subband
                coeffs_w = Yh_w[scale][:, self.w_channel, subband]

                # Extract the ground truth coefficients and mask for the current subband
                gt_coeffs = self.gt_patch[scale][:, self.w_channel, subband]
                mask = self.watermarking_mask[scale][:, self.w_channel, subband]

                # Compute the L1 difference within the masked region
                l1_metric += torch.abs(coeffs_w[mask] - gt_coeffs[mask]).sum().item()
                count += mask.sum().item()

        # Normalize the L1 metric by the number of masked elements
        l1_metric /= count

        return l1_metric


    def watermark_stat(self):
        dis_all = []
        for i in range(1000):
            rand_latents = randn_tensor(self.shape, device=self.device, dtype=self.dtype)
            dis = self.eval_watermark(rand_latents)
            dis_all.append(dis)
        dis_all = np.array(dis_all)
        return dis_all.mean(), dis_all.var()

    def one_minus_p_value(self, latents):
        # Compute the L1 metric in the DWT domain
        l1_metric = self.eval_watermark(latents)
        
        # Calculate the p-value using the cumulative distribution function (CDF) of the normal distribution
        p_value = norm.cdf(l1_metric, self.mu, self.sigma)
        
        # Return 1 minus the p-value
        return abs(0.5 - p_value) * 2

    def tree_ring_p_value(self, latents):
        # Perform the DWT on the latents
        Yl, Yh = self.dwt(latents)
        
        # Flatten the high-frequency components (Yh) and the ground truth patch
        target_patch = self.gt_patch[self.watermarking_mask].flatten()
        target_patch = torch.cat([target_patch.real, target_patch.imag])
        
        # Initialize a list to store p-values for each scale and subband
        p_values = []
        
        # Iterate over each scale and subband
        for scale in range(len(Yh)):
            for subband in range(Yh[scale].shape[2]):
                # Extract the coefficients for the current subband
                coeffs = Yh[scale][:, self.w_channel, subband]
                
                # Flatten the coefficients and concatenate real and imaginary parts
                coeffs_flat = torch.cat([coeffs.real, coeffs.imag])
                
                # Compute the standard deviation of the coefficients
                sigma_w = coeffs_flat.std()
                
                # Compute the lambda parameter for the non-central chi-squared distribution
                lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
                
                # Compute the x statistic
                x_w = (((coeffs_flat - target_patch) / sigma_w) ** 2).sum().item()
                
                # Compute the p-value using the non-central chi-squared CDF
                p_w = ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)
                
                # Append the p-value to the list
                p_values.append(p_w)
        
        # Return the list of p-values
        return p_values


class GTWatermarkMulti(GTWatermark):
    def __init__(self, device, shape=(1,4,64,64), dtype=torch.float32, w_settings={0:[1,5,9], 1:[2,6,10], 2:[3,7], 3:[4,8]}, generator=None):
        self.device = device
        self.shape = shape
        self.dtype = dtype
        self.w_settings = w_settings

        # Initialize DWT and IDWT
        self.dwt = DWTForward(J=1, mode='zero', wave='haar').to(self.device)
        self.idwt = DWTInverse(mode='zero', wave='haar').to(self.device)

        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        self.mu, self.sigma = self.watermark_stat()

    def _get_watermarking_pattern(self, gt_init):
        # Perform DWT
        Yl, Yh = self.dwt(gt_init)
        gt_patch = Yl
        watermarking_mask = torch.zeros(gt_init.shape, dtype=torch.bool).to(self.device)
        for w_channel in self.w_settings:
            for w_radius in self.w_settings[w_channel]:
                tmp_mask_alter, tmp_mask_inner = self._circle_mask(gt_init.shape[-1], r=w_radius), self._circle_mask(gt_init.shape[-1], r=w_radius-1)
                tmp_mask = torch.tensor(np.logical_xor(tmp_mask_alter, tmp_mask_inner)).to(self.device)
                gt_patch[:, w_channel, tmp_mask] = gt_patch[0, w_channel, 0, w_radius].item()
                watermarking_mask[:, w_channel, tmp_mask] = True
        return gt_patch, watermarking_mask

    def _gen_gt(self, generator=None):
        gt_init = randn_tensor(self.shape, generator=generator, device=self.device, dtype=self.dtype)
        gt_patch, watermarking_mask = self._get_watermarking_pattern(gt_init)
        return gt_patch, watermarking_mask
