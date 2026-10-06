#!/usr/bin/env python
"""Test-Skript für das Image Processing Framework mit Bild aus public Ordner."""

import sys


sys.path.insert(0, 'src')

import cv2
import numpy as np

from src.ipf.evaluator import QualityEvaluator
from src.ipf.filters import BilateralDenoiseFilter
from src.ipf.filters import BlurFilter
from src.ipf.filters import GaussianNoiseFilter
from src.ipf.frame import Frame
from src.ipf.image_sink import ImageSink
from src.ipf.image_source import ImageSource
from src.ipf.pipeline import Pipeline


def main():
    """Test-Funktion mit Bild aus public/cyberpunk_2077_test.png."""

    # Pfad zum Testbild im public Ordner
    input_path = "public/cyberpunk_2077_test.png"

    # Überprüfen, ob das Bild existiert
    import os
    if not os.path.exists(input_path):
        print(f"Fehler: Das Bild '{input_path}' wurde nicht gefunden!")
        return

    print("=" * 60)
    print("IMAGE PROCESSING FRAMEWORK TEST")
    print("=" * 60)

    # Schritt 1: Lade das Testbild
    print("\n[1] Lade Testbild aus 'public/cyberpunk_2077_test.png'...")
    test_image = cv2.imread(input_path)

    if test_image is None:
        print(f"Fehler: Konnte Bild '{input_path}' nicht laden!")
        return

    print(f"   Bild geladen: {test_image.shape[1]}x{test_image.shape[0]} px")
    print(f"   Farbraum: {'Farbe (RGB)' if len(test_image.shape) == 3 else 'Graustufen'}")

    # Schritt 2: Erstelle ImageSource
    print("\n[2] Initialisiere Bildquelle...")
    source = ImageSource(file_path=input_path)
    print("   ImageSource erfolgreich initialisiert.")

    # Schritt 3: Erstelle Pipeline mit Filtern
    print("\n[3] Erstelle Pipeline und füge Filter hinzu:")
    pipeline = Pipeline()

    # Gaussian Noise hinzufügen
    noise_filter = GaussianNoiseFilter(sigma=25.0, name="GaussianNoise")
    pipeline.add_filter(noise_filter)
    print(f"   - {noise_filter.name}: Sigma={noise_filter.sigma}")

    # Bilateral Denoise hinzufügen
    denoise_filter = BilateralDenoiseFilter(d=9, sigma_color=75, sigma_space=75, name="BilateralDenoise")
    pipeline.add_filter(denoise_filter)
    print(f"   - {denoise_filter.name}: d={denoise_filter.d}, sigmaColor={denoise_filter.sigma_color}, sigmaSpace={denoise_filter.sigma_space}")

    # Blur Filter hinzufügen
    blur_filter = BlurFilter(kernel_size=5, name="Blur")
    pipeline.add_filter(blur_filter)
    print(f"   - {blur_filter.name}: kernel_size={blur_filter.kernel_size}")

    # Schritt 4: Führe Pipeline aus
    print("\n[4] Verarbeite Bild durch Filter-Kette...")
    sink_output = "public/test_output_processed.png"
    sink = ImageSink(output_path=sink_output)
    processed_frames = pipeline.run(source=source, sink=sink)

    print(f"   Pipeline-Fertigstellung: {len(processed_frames)} Frame(n) verarbeitet.")
    print(f"   Ergebnis gespeichert als: {sink_output}")

    # Schritt 5 & 6: Qualitätsvergleich
    print("\n[5] Qualitätsvergleich: Original vs. Verarbeitet")
    print("-" * 60)

    # Beide Bilder in Graustufen konvertieren für die Qualitätsbewertung
    original_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    original_frame = Frame(data=original_gray, index=0)

    # Verarbeitetes Bild auch in Graustufen (es wurde als Farbbild gespeichert)
    processed_color = cv2.cvtColor(processed_frames[-1].data if len(processed_frames) > 0 else np.zeros_like(test_image), cv2.COLOR_BGR2GRAY) if len(processed_frames) > 0 else np.zeros_like(original_gray, dtype=np.uint8)
    processed_frame = Frame(data=processed_color, index=-1)

    psnr_value = QualityEvaluator.calculate_psnr(original_frame, processed_frame)
    print(f"\n   PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    if psnr_value == float("inf"):
        print("   => Bilder identisch!")

    ssim_value = QualityEvaluator.calculate_ssim(original_frame, processed_frame)
    print(f"   SSIM (Structural Similarity Index): {ssim_value:.4f}")
    if ssim_value == 1.0:
        print("   => Strukturen identisch!")

    # Schritt 7: Generiere Diff-Heatmap
    print("\n[6] Generiere Differenz-Heatmap...")
    diff_frame = QualityEvaluator.generate_diff_map(original_frame, processed_frame)
    diff_output = "public/diff_heatmap.png"
    cv2.imwrite(diff_output, diff_frame.data)
    print(f"   Diff-Heatmap gespeichert als: {diff_output}")

    # Schritt 8: Speichere auch verarbeitetes Bild in src (zusätzlich zu public)
    test_output_path = "test_output_processed.png"
    cv2.imwrite(test_output_path, processed_frame.data)
    print(f"\n   Verarbeitetes Grau-Ton-Bild gespeichert als: {test_output_path}")

    # Zusammenfassung
    print("\n" + "=" * 60)
    print("TEST ZUSAMMENFASSUNG:")
    print("=" * 60)
    print(f"   Eingabebild:        {input_path}")
    print(f"   Ausgabebild (Color):       {sink_output}")
    print(f"   Ausgabebild (Gray):        {test_output_path}")
    print(f"   Diff-Heatmap:       {diff_output}")
    print(f"\n   PSNR:  {psnr_value:.2f} dB")
    print(f"   SSIM:  {ssim_value:.4f}")

    # Bewertung
    if psnr_value > 30:
        print("   => Sehr gute Qualität!")
    elif psnr_value > 25:
        print("   => Gute Qualität")
    else:
        print("   => Mittlere bis niedrige Qualität (erwartbar bei starken Filtern)")

    print("=" * 60)


if __name__ == "__main__":
    main()
