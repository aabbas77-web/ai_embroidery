import cv2
import numpy as np
from pyembroidery import *
from pathlib import Path

# === PARAMETERS ===
INPUT_IMAGE = "logo0.bmp"
STITCH_SPACING = 3          # distance between rows (pixels)
POINT_SPACING = 3           # distance between points in each scan line
OUTPUT = "fill_contour.dst"

path = Path(INPUT_IMAGE)

# === Load image and extract contour ===
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)    # choose largest contour

# Create filled mask
h, w = img.shape
mask = np.zeros((h, w), np.uint8)
cv2.drawContours(mask, [cnt], -1, 255, -1)

# === Prepare Embroidery Pattern ===
pattern = EmbPattern()
pattern.color_change()

# === Generate continuous fill stitches ===
y_min, y_max = np.min(cnt[:,:,1]), np.max(cnt[:,:,1])

direction = 1  # left→right then right→left

for y in range(y_min, y_max, STITCH_SPACING):
    row = mask[y]   # 1D row of pixels

    xs = np.where(row == 255)[0]   # all filled pixels on this row
    if len(xs) < 2:
        continue

    # Downsample along row
    xs = xs[::POINT_SPACING]

    # Zig-zag direction
    if direction > 0:
        xs_sorted = xs
    else:
        xs_sorted = xs[::-1]

    for x in xs_sorted:
        pattern.stitch_abs(int(x), int(y))

    direction *= -1

# End and save
pattern.end()

# Save Results
write_dst(pattern, str(path.with_suffix(".dst")))
write_png(pattern, str(path.with_suffix(".png")))
print(f"✅ Satin embroidery saved...")
