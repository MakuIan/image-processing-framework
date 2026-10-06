import cv2
import numpy as np
from ipf.frame import Frame


class QualityEvaluator:
    """Berechnet Bild- und Video-Qualitätsmetriken."""

    @staticmethod
    def calculate_psnr(original: Frame, processed: Frame) -> float:
        """
        Berechnet den PSNR-Wert zwischen den beiden data-Arrays.

        Args:
            original: Original-Frame.
            processed: Verarbeiteter Frame.

        Returns:
            PSNR-Wert in dB (Peak Signal-to-Noise Ratio).
        """
        mse = np.mean((original.data.astype(np.float64) - processed.data.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        max_val = original.data.max()
        psnr = 10 * np.log10((max_val ** 2) / mse)
        return float(psnr)

    @staticmethod
    def calculate_ssim(original: Frame, processed: Frame) -> float:
        """
        Berechnet SSIM (Structural Similarity Index Measure).

        Args:
            original: Original-Frame.
            processed: Verarbeiteter Frame.

        Returns:
            SSIM-Wert zwischen 0 und 1.
        """
        def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
            """
            Berechnet SSIM für ein oder mehrere Kanäle (RGB/BGR).
            """
            if len(img1.shape) == 3 and img1.shape[-1] > 1:
                ssim_values = []
                for c in range(0, 3):
                    s = _compute_ssim_single_channel(img1[..., c], img2[..., c])
                    ssim_values.append(s)
                return float(np.mean(ssim_values))
            else:
                return float(_compute_ssim_single_channel(img1, img2))

        @staticmethod
        def _compute_ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
            """
            Berechnet SSIM für einen einzelnen Grauwert-Kanal.
            Implementierung nach Wang et al. (2004).
            """
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            mu1 = cv2.GaussianBlur(img1, (15, 15), 0)
            mu2 = cv2.GaussianBlur(img2, (15, 15), 0)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(img1.astype(np.float64) ** 2, (15, 15), 0) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2.astype(np.float64) ** 2, (15, 15), 0) - mu2_sq
            sigma12 = cv2.GaussianBlur((img1 * img2).astype(np.float64), (15, 15), 0) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return float(np.mean(ssim_map))

        return _compute_ssim(original.data.astype(np.float64), processed.data.astype(np.float64))

    @staticmethod
    def generate_diff_map(original: Frame, processed: Frame) -> Frame:
        """
        Generiert eine Heatmap-Darstellung der absoluten Differenz.

        Args:
            original: Original-Frame.
            processed: Verarbeiteter Frame.

        Returns:
            Neuer Frame mit einer farbig skalierten Heatmap der Differenz.
        """
        diff_gray = cv2.absdiff(original.data.astype(np.uint8), processed.data.astype(np.uint8))
        max_diff = diff_gray.max()
        if max_diff > 0:
            normalized = (diff_gray / max_diff * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        else:
            heatmap = np.zeros_like(original.data, dtype=np.uint8)
        return Frame(data=heatmap, index=-1, timestamp=-1.0, metadata={"diff_type": "jet"})
