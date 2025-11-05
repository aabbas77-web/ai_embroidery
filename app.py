import cv2
import numpy as np
from pyembroidery import EmbPattern, write_dst, write_png, EmbConstant

# === Parameters ===
INPUT_IMAGE = "logo.png"
OUTPUT_DST = "logo_auto.dst"
STITCH_SPACING = 2.0  # mm equivalent in stitch units
COLOR_TOLERANCE = 40  # for color segmentation

# === Step 1. Load and preprocess image ===
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError("Cannot read image file.")

# Resize for smaller stitch count
scale = 200 / max(image.shape[:2])
image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

# Convert to LAB for better color clustering
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
pixels = lab.reshape(-1, 3)

# === Step 2. Cluster colors (segment the image) ===
n_colors = 4  # adjust as needed
_, labels, centers = cv2.kmeans(
    np.float32(pixels),
    n_colors,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
    10,
    cv2.KMEANS_RANDOM_CENTERS,
)

segmented = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.convertScaleAbs(segmented)

# === Step 3. Extract contours per color region ===
pattern = EmbPattern()

for i, color in enumerate(centers.astype(np.uint8)):
    mask = cv2.inRange(segmented, color - COLOR_TOLERANCE, color + COLOR_TOLERANCE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        continue

    # Convert LAB color back to BGR for reference
    color_bgr = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2BGR)[0][0]
    print(f"Color {i+1}: {color_bgr} with {len(contours)} contours")

    # pattern.add_color_change()
    pattern.color_change()

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:  # skip small details
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2:
            continue

        # Draw stitches in fill pattern
        x, y, w, h = cv2.boundingRect(cnt)
        for yy in np.arange(y, y + h, STITCH_SPACING):
            row_pts = cnt[(cnt[:, 1] >= yy) & (cnt[:, 1] < yy + STITCH_SPACING)]
            if len(row_pts) > 1:
                row_pts = np.sort(row_pts[:, 0])
                pattern.add_stitch_absolute(EmbConstant.STITCH, row_pts[0], yy)
                pattern.add_stitch_absolute(EmbConstant.STITCH, row_pts[-1], yy)

# === Step 4. Save as .DST ===
pattern.end()
# write_dst(pattern, OUTPUT_DST)
write_png(pattern, "demo.png")

print(f"✅ Embroidery file saved: {OUTPUT_DST}")
