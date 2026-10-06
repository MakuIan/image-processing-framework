#!/usr/bin/env python
"""Run Demo für das Bild- und Video-Processing-Framework."""

import sys

import cv2
import numpy as np


sys.path.insert(0, 'src')

from src.ipf.evaluator import QualityEvaluator
from src.ipf.filters import BilateralDenoiseFilter
from src.ipf.filters import BlurFilter
from src.ipf.filters import GaussianNoiseFilter
from src.ipf.frame import Frame
from src.ipf.image_sink import ImageSink
from src.ipf.image_source import ImageSource
from src.ipf.pipeline import Pipeline


def generate_synthetic_image(size: int = 512) -> np.ndarray:
    """Erzeugt ein Schachbrettmuster als Testbild."""
    h, w = size, size
    checkerboard = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, 64):
        for j in range(0, w, 64):
            if (i // 64 + j // 64) % 2 == 0:
                checkerboard[i:i+32, j:j+32] = 255
            else:
                checkerboard[i:i+32, j:j+32] = 128

    color_image = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
    return color_image


def main():
    """Main Demo-Funktion."""
    print("=" * 60)
    print("Bild-Processing-Demo: Filter-Kette mit Qualitätsprüfung")
    print("=" * 60)

    # Schritt 1: Erzeuge synthetisches Testbild
    print("\n[1] Erstelle synthetisches Schachbrettmuster (512x512)...")
    test_image = generate_synthetic_image(size=512)

    input_path = "test_input.png"
    cv2.imwrite(input_path, test_image)
    print(f"   Originalbild gespeichert als: {input_path}")

    # Schritt 2: Erstelle ImageSource
    print("\n[2] Initialisiere Bildquelle...")
    source = ImageSource(file_path=input_path)
    print("   ImageSource für 'test_input.png' erstellt.")

    # Schritt 3: Erstelle Pipeline mit Filtern
    print("\n[3] Erstelle Pipeline und füge Filter hinzu:")
    pipeline = Pipeline()
    noise_filter = GaussianNoiseFilter(sigma=30.0, name="GaussianNoise")
    pipeline.add_filter(noise_filter)

    denoise_filter = BilateralDenoiseFilter(name="BilateralDenoise")
    pipeline.add_filter(denoise_filter)

    blur_filter = BlurFilter(kernel_size=5, name="Blur")
    pipeline.add_filter(blur_filter)

    # Schritt 4: Führe Pipeline aus
    print("\n[4] Verarbeite Bild durch Filter-Kette...")
    sink_output = "test_output.png"
    sink = ImageSink(output_path=sink_output)
    processed_frames = pipeline.run(source=source, sink=sink)

    print(f"   Pipeline-Fertigstellung: {len(processed_frames)} Frame(n) verarbeitet.")
    print(f"   Ergebnis gespeichert als: {sink_output}")

    # Schritt 5 & 6: Qualitätsvergleich und Ausgeben von Messwerten
    print("\n[5] Qualitätsvergleich: Original vs. Verarbeitet")
    print("-" * 60)

    original_frame = Frame(data=test_image, index=0)
    processed_frame = processed_frames[-1] if len(processed_frames) > 0 else Frame(data=np.zeros_like(test_image), index=-1)

    psnr_value = QualityEvaluator.calculate_psnr(original_frame, processed_frame)
    print(f"\n   PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    if psnr_value == float("inf"):
        print("   => Bilder identisch!")

    ssim_value = QualityEvaluator.calculate_ssim(original_frame, processed_frame)
    print(f"   SSIM (Structural Similarity Index): {ssim_value:.4f}")
    if ssim_value == 1.0:
        print("   => Strukturen identisch!")

    # Schritt 7: Speichere Diff-Heatmap
    print("\n[6] Generiere Differenz-Heatmap...")
    diff_frame = QualityEvaluator.generate_diff_map(original_frame, processed_frame)
    diff_output = "diff_heatmap.png"
    cv2.imwrite(diff_output, diff_frame.data)
    print(f"   Diff-Heatmap gespeichert als: {diff_output}")

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG:")
    print("=" * 60)
    print(f"   Eingabebild:        {input_path}")
    print(f"   Ausgabebild:        {sink_output}")
    print(f"   Diff-Heatmap:       {diff_output}")
    print(f"\n   PSNR:  {psnr_value:.2f} dB")
    print(f"   SSIM:  {ssim_value:.4f}")
    print("=" * 60)

    cv2.imwrite("test_processed.png", processed_frame.data)
    print("\nVerarbeitetes Bild auch als 'test_processed.png' gespeichert.")

if __name__ == "__main__":
    main()
