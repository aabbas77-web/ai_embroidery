import cv2
import numpy as np
# from pyembroidery import EmbPattern, write_dst, write_pec, write_pes, write_exp, write_vp3, write_jef, write_u01
# from pyembroidery import write_csv, write_json, write_txt, write_gcode, write_xxx, write_tbf, write_svg, write_png
# from pyembroidery import EmbConstant
from pyembroidery import *
from pathlib import Path
from PIL import Image

def wu_quantize_image(image, colors=8):
    """
    Quantizes an image using the optimized Wu's algorithm implementation
    in Pillow and saves the result.
    """
    img = cv2_to_pil(image)

    # 1. Quantize the image
    # The 'method' parameter is key here. Using 0/Image.Resample.FASTOCTREE
    # activates the optimized octree/Wu's quantization method.
    # Note: Pillow often uses a hybrid approach, but this setting
    # represents the highest quality quantization available in the library.
    quantized_img = img.quantize(
        colors=colors,
        method=Image.Quantize.FASTOCTREE
    )

    # 2. Save the result
    return pil_to_cv2(quantized_img)

# 1. OpenCV to PIL (Starting with an image loaded by OpenCV)
def cv2_to_pil(cv_img):
    """Converts a BGR OpenCV NumPy array to a RGB PIL Image object."""
    # Convert BGR (OpenCV standard) to RGB (PIL standard)
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Create PIL Image from the array
    pil_img = Image.fromarray(img_rgb)
    return pil_img

# 2. PIL to OpenCV (Starting with an image loaded by PIL)
def pil_to_cv2(pil_img):
    """Converts an RGB PIL Image object to a BGR OpenCV NumPy array."""
    # Convert PIL Image object to NumPy array (it will be in RGB)
    cv_img_rgb = np.array(pil_img)
    # Convert RGB array to BGR (OpenCV standard)
    cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
    return cv_img_bgr

# === Parameters ===
INPUT_IMAGE = "logo.bmp"
STITCH_SPACING = 2.0  # mm equivalent in stitch units
COLOR_TOLERANCE = 20  # for color segmentation

path = Path(INPUT_IMAGE)

# === Step 1. Load and preprocess image ===
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError("Cannot read image file.")

# Resize for smaller stitch count
scale = 512 / max(image.shape[:2])
image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

# Convert to LAB for better color clustering
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
pixels = lab.reshape(-1, 3)

# === Step 2. Cluster colors (segment the image) ===
n_colors = 3  # adjust as needed
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
    # pattern.add_stitch_relative(EmbConstant.COLOR_CHANGE, 0, 0)

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:  # skip small details
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2:
            continue

        # Draw stitches in fill pattern
        first = True
        x, y, w, h = cv2.boundingRect(cnt)
        for yy in np.arange(y, y + h, STITCH_SPACING):
            row_pts = cnt[(cnt[:, 1] >= yy) & (cnt[:, 1] < yy + STITCH_SPACING)]
            if len(row_pts) > 1:
                row_pts = np.sort(row_pts[:, 0])
                if first:
                    first = False
                    pattern.move()
                    # pattern.add_stitch_relative(EmbConstant.JUMP, 0, 0)

                pattern.stitch_abs(row_pts[0], yy)
                pattern.stitch_abs(row_pts[-1], yy)
                # pattern.add_stitch_absolute(EmbConstant.STITCH, row_pts[0], yy)
                # pattern.add_stitch_absolute(EmbConstant.STITCH, row_pts[-1], yy)

# === Step 4. Save as .DST ===
pattern.end()
# pattern.add_stitch_relative(EmbConstant.END, 0, 0)

write_dst(pattern, str(path.with_suffix(".dst")))
write_pec(pattern, str(path.with_suffix(".pec")))
write_pes(pattern, str(path.with_suffix(".pes")))
write_exp(pattern, str(path.with_suffix(".exp")))
write_vp3(pattern, str(path.with_suffix(".vp3")))
# write_jef(pattern, str(path.with_suffix(".jef")))
write_u01(pattern, str(path.with_suffix(".u01")))
write_csv(pattern, str(path.with_suffix(".csv")))
# write_json(pattern, str(path.with_suffix(".json")))
write_txt(pattern, str(path.with_suffix(".txt")))
write_gcode(pattern, str(path.with_suffix(".gcode")))
write_xxx(pattern, str(path.with_suffix(".xxx")))
write_tbf(pattern, str(path.with_suffix(".tbf")))
write_svg(pattern, str(path.with_suffix(".svg")))
write_png(pattern, str(path.with_suffix(".png")))

print(f"✅ Embroidery file saved...")
