import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import log10, sqrt
import main
import matplotlib
matplotlib.use('Agg')

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # Ni razlike
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def shannon_entropy(image):
    histogram = np.histogram(image.flatten(), bins=256, range=(0, 256))[0]
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum([p * np.log2(p) for p in histogram if p > 0])
    return entropy

def blockiness(image, block_size=8):
    h, w = image.shape
    vertical_lines = np.sum(np.abs(np.diff(image, axis=1)))
    horizontal_lines = np.sum(np.abs(np.diff(image, axis=0)))
    total_lines = vertical_lines + horizontal_lines
    return total_lines / (h * w)

def analyze_images(image_folder, thresholds, block_size=8):
    results = []
    images = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]

    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        original_image = np.array(Image.open(image_path).convert('L'))
        original_size = os.path.getsize(image_path)

        for thr in thresholds:
            compressed_file = f"compressed_{image_file}_thr_{thr}.bin"
            decompressed_file = f"decompressed_{image_file}_thr_{thr}.bmp"

            # Kompresija
            main.compress(image_path, compressed_file, thr)
            compressed_size = os.path.getsize(compressed_file)

            # Dekompresija
            main.decompress(compressed_file, decompressed_file)
            decompressed_image = np.array(Image.open(decompressed_file).convert('L'))

            # Izračun metrik
            psnr = calculate_psnr(original_image, decompressed_image)
            entropy_original = shannon_entropy(original_image)
            entropy_compressed = shannon_entropy(decompressed_image)
            blockiness_original = blockiness(original_image, block_size)
            blockiness_compressed = blockiness(decompressed_image, block_size)

            compression_ratio = original_size / compressed_size

            results.append({
                'image': image_file,
                'threshold': thr,
                'compression_ratio': compression_ratio,
                'psnr': psnr,
                'entropy_original': entropy_original,
                'entropy_compressed': entropy_compressed,
                'blockiness_original': blockiness_original,
                'blockiness_compressed': blockiness_compressed
            })

    return results

def plot_results(results, output_folder="graphs"):
    os.makedirs(output_folder, exist_ok=True)
    thresholds = sorted(list(set([r['threshold'] for r in results])))
    images = sorted(list(set([r['image'] for r in results])))

    for image in images:
        image_results = [r for r in results if r['image'] == image]
        cr = [r['compression_ratio'] for r in image_results]
        psnr = [r['psnr'] for r in image_results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(thresholds, cr, marker='o')
        plt.title(f'Compression Ratio - {image}')
        plt.xlabel('Threshold')
        plt.ylabel('Compression Ratio')

        plt.subplot(1, 2, 2)
        plt.plot(thresholds, psnr, marker='o', color='r')
        plt.title(f'PSNR - {image}')
        plt.xlabel('Threshold')
        plt.ylabel('PSNR')

        plt.tight_layout()
        graph_path = os.path.join(output_folder, f"{image}_metrics.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"Graf za {image} shranjen kot {graph_path}")


image_folder = "SlikeBMP10"  # Vhodna mapa s slikami
thresholds = [0, 25, 50, 100]

results = analyze_images(image_folder, thresholds)

results_df = pd.DataFrame(results)
print(results_df)

results_csv = "analysis_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"Rezultati analize so shranjeni v datoteko: {results_csv}")
