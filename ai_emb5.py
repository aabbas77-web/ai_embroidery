import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite


# --- Helper Functions ---

def rgb_to_hex(rgb):
    """Converts a NumPy RGB array to a valid SVG hex color string (e.g., #131e19)."""
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


def is_inside_hole(point, outer_contour_idx, all_contours, all_hierarchy):
    """Checks if a point is inside any hole associated with the given outer_contour_idx."""
    for j, h_data in enumerate(all_hierarchy):
        if h_data[3] == outer_contour_idx:
            hole_contour = all_contours[j]
            if cv2.pointPolygonTest(hole_contour, point, measureDist=False) >= 0:
                return True
    return False


# --- Main Vectorization Function ---

def vectorize_image_wu(image_path, n_colors=8, line_distance=5, hatch_angle=-45.0, stroke_thickness=1.0, random_seed=42,
                       epsilon_factor=0.02):
    # 1. Load Image and Setup
    image = cv2.imread(image_path)
    if image is None: return

    # Resize for smaller stitch count
    scale = 400 / max(image.shape[:2])
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, d = image_rgb.shape
    pixels = image_rgb.reshape((-1, d))

    # 2. Quantization (Wu's proxy)
    kmeans = KMeans(n_clusters=n_colors, random_state=random_seed, n_init=10).fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_pixels.reshape((h, w, d)).astype(np.uint8)
    color_palette = kmeans.cluster_centers_.astype(np.uint8)

    color_data = []

    # 3. Masking, 4. Contour Extraction, and SIMPLIFICATION
    for i, color_rgb in enumerate(color_palette):
        mask = np.all((quantized_image == color_rgb), axis=2).astype(np.uint8) * 255

        # Extract raw contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # --- CONTOUR SIMPLIFICATION (New Optimization) ---
        simplified_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = epsilon_factor * perimeter  # Epsilon is now a percentage of perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
            simplified_contours.append(approx_contour)

        color_data.append({
            'color': color_rgb,
            'contours': simplified_contours,
            'hierarchy': hierarchy[0] if hierarchy is not None else []
        })

    # --- GEOMETRIC PREPARATION & CONTOUR ROTATION ---
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, hatch_angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, -hatch_angle, 1.0)

    rotated_color_data = []
    # FIX: Rotate ALL contours (outer and holes) once
    for data in color_data:
        rotated_contours_for_color = []
        for contour in data['contours']:  # Using simplified contours
            contour_float = contour.astype(np.float32)
            rotated_contour = cv2.transform(contour_float, M)
            rotated_contours_for_color.append(rotated_contour)

        rotated_color_data.append({
            'color': data['color'],
            'contours': rotated_contours_for_color,
            'hierarchy': data['hierarchy']
        })

    # --- 5. Custom Line Fill and 6. Save to Vector File (SVG) ---
    dwg = svgwrite.Drawing('output_final_vector.svg', size=(w, h), profile='tiny')

    for data in rotated_color_data:
        color_hex = rgb_to_hex(data['color'])
        contours = data['contours']
        hierarchy = data['hierarchy']
        outer_contours_indices = [idx for idx, h_data in enumerate(hierarchy) if h_data[3] == -1]

        for idx in outer_contours_indices:
            outer_contour = contours[idx]
            x, y, bw, bh = cv2.boundingRect(outer_contour)

            for y_line in range(y, y + bh, line_distance):
                is_drawing = False
                start_x = -1

                for x_pos in range(x, x + bw):
                    point_rotated = (x_pos, y_line)
                    is_inside_outer = cv2.pointPolygonTest(outer_contour, point_rotated, measureDist=False) >= 0
                    is_outside_holes = not is_inside_hole(point_rotated, idx, contours, hierarchy)
                    is_valid_point = is_inside_outer and is_outside_holes

                    if is_valid_point:
                        if not is_drawing:
                            is_drawing = True
                            start_x = x_pos
                    else:
                        if is_drawing:
                            # INVERSE TRANSFORMATION
                            points_to_transform = np.array([[start_x, y_line], [x_pos, y_line]],
                                                           dtype=np.float32).reshape(-1, 1, 2)
                            points_transformed = cv2.transform(points_to_transform, M_inv).reshape(-1, 2)

                            dwg.add(dwg.line(
                                start=points_transformed[0].tolist(),
                                end=points_transformed[1].tolist(),
                                stroke=color_hex,
                                stroke_width=stroke_thickness
                            ))
                            is_drawing = False

                if is_drawing:
                    points_to_transform = np.array([[start_x, y_line], [x + bw, y_line]],
                                                   dtype=np.float32).reshape(-1, 1, 2)
                    points_transformed = cv2.transform(points_to_transform, M_inv).reshape(-1, 2)

                    dwg.add(dwg.line(
                        start=points_transformed[0].tolist(),
                        end=points_transformed[1].tolist(),
                        stroke=color_hex,
                        stroke_width=stroke_thickness
                    ))

    dwg.save()
    print("Processing complete. Results saved to 'output_final_vector.svg'")
    return quantized_image, color_palette

# Example Usage:
vectorize_image_wu('logo0.bmp', n_colors=10, hatch_angle=45.0, line_distance=3, stroke_thickness=0.5, epsilon_factor=0.001)
