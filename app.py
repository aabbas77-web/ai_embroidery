import cv2
import numpy as np
from skimage.morphology import skeletonize
from pyembroidery import *
from pathlib import Path

# === Parameters ===
INPUT_IMAGE = "logo.bmp"
THRESHOLD = 127
STITCH_SPACING = 1.0  # distance between stitches
SATIN_WIDTH = 4.0     # width of satin band
DESIGN_WIDTH = 640

path = Path(INPUT_IMAGE)

# === Step 1. Load and preprocess ===
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError("Cannot read image file.")

# Resize for smaller stitch count
scale = DESIGN_WIDTH / max(img.shape[:2])
img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
_, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

cv2.imwrite("gray.png", gray)
cv2.imwrite("mask.png", mask)

# === Step 2. Extract contours ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pattern = EmbPattern()

for ci, contour in enumerate(contours):
    if cv2.contourArea(contour) < 100:
        continue

    # Create filled mask for the shape
    shape_mask = np.zeros_like(mask)
    cv2.drawContours(shape_mask, [contour], -1, 255, 1)
    cv2.imwrite("shape_mask.png", shape_mask)

    # === Step 3. Skeletonization ===
    skeleton = skeletonize(shape_mask > 0)

    yx = np.argwhere(skeleton)
    if len(yx) < 2:
        continue

    pattern.color_change()
    direction = 1

    # === Step 4. Generate satin stitches ===
    for (y, x) in yx[::int(STITCH_SPACING)]:
        # Gradient-based perpendicular estimation
        gx = cv2.Sobel(shape_mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(shape_mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        nx, ny = gx[int(y), int(x)], gy[int(y), int(x)]
        nlen = np.hypot(nx, ny)
        if nlen == 0:
            continue
        nx, ny = nx / nlen, ny / nlen

        # Compute start and end of satin width
        x1 = x - nx * SATIN_WIDTH / 2
        y1 = y - ny * SATIN_WIDTH / 2
        x2 = x + nx * SATIN_WIDTH / 2
        y2 = y + ny * SATIN_WIDTH / 2

        if direction > 0:
            pattern.stitch_abs(x1, y1)
            pattern.stitch_abs(x2, y2)
        else:
            pattern.stitch_abs(x2, y2)
            pattern.stitch_abs(x1, y1)
        direction *= -1  # alternate direction

pattern.end()

# Preview
stitches = np.array(pattern.stitches, dtype=float)
stitches -= np.min(stitches, axis=0)
stitches /= np.max(stitches, axis=0)
canvas = np.ones((DESIGN_WIDTH, DESIGN_WIDTH, 3), np.uint8) * 255
for i in range(1, len(stitches)):
    p1 = tuple((stitches[i - 1] * 300).astype(int))
    p2 = tuple((stitches[i] * 300).astype(int))
    p1_2d = tuple(p1[:2])
    p2_2d = tuple(p2[:2])
    cv2.line(canvas, p1_2d, p2_2d, (0, 0, 0), 1)
cv2.imshow("Embroidery Preview", canvas)
cv2.imwrite("preview.png", canvas)
cv2.waitKey(0)

# Save Results
write_dst(pattern, str(path.with_suffix(".dst")))
write_png(pattern, str(path.with_suffix(".png")))
print(f"✅ Satin embroidery saved...")
