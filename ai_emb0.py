import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite  # For vector file output
from pathlib import Path

def vectorize_image_wu(image_path, n_colors=8, line_distance=5):
    # --- 1. Load Image ---
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not loaded.")
        return

    # Convert to RGB (OpenCV loads BGR by default) and float for clustering
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 2. Quantize to 8 Colors (Optimized Clustering) ---
    # Reshape the image to be a list of pixels (N x 3)
    h, w, d = image_rgb.shape
    pixels = image_rgb.reshape((-1, d))

    # Apply K-Means (an optimized quantization method)
    print(f"Quantizing to {n_colors} colors...")
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Map the clustered centers back to create the quantized image
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_pixels.reshape((h, w, d)).astype(np.uint8)

    # Get the 8 specific color values
    color_palette = kmeans.cluster_centers_.astype(np.uint8)

    # The rest of the pipeline will go here
    # ...

    return quantized_image, color_palette

# Example usage:
INPUT_IMAGE = "logo0.bmp"

path = Path(INPUT_IMAGE)

quantized_img, palette = vectorize_image_wu(INPUT_IMAGE)
cv2.imwrite(str(path.with_suffix(".quantized.bmp")), quantized_img)
