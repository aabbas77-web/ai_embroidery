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
        # h_data[3] is the parent index. If it matches the outer contour's index, it's a hole.
        if h_data[3] == outer_contour_idx:
            hole_contour = all_contours[j]
            if cv2.pointPolygonTest(hole_contour, point, measureDist=False) >= 0:
                return True
    return False


# --- Main Vectorization Function ---

def vectorize_image_wu(image_path, n_colors=8, line_distance=5, hatch_angle=-45.0, stroke_thickness=1.0, random_seed=42,
                       epsilon_factor_outer=0.01, epsilon_factor_hole=0.04):
    # 1. Load Image and Quantize
    image = cv2.imread(image_path)
    if image is None:
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, d = image_rgb.shape
    pixels = image_rgb.reshape((-1, d))

    kmeans = KMeans(n_clusters=n_colors, random_state=random_seed, n_init=10).fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_pixels.reshape((h, w, d)).astype(np.uint8)
    color_palette = kmeans.cluster_centers_.astype(np.uint8)

    color_data = []

    # 2. Masking, Contour Extraction, and ADAPTIVE SIMPLIFICATION
    for i, color_rgb in enumerate(color_palette):
        mask = np.all((quantized_image == color_rgb), axis=2).astype(np.uint8) * 255

        # Extract raw contours and hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            color_data.append({'color': color_rgb, 'contours': [], 'hierarchy': []})
            continue

        simplified_contours = []
        hierarchy_data = hierarchy[0]

        # ADAPTIVE SIMPLIFICATION based on Hierarchy
        for idx, contour in enumerate(contours):
            # Check the Parent Index (position 3)
            if hierarchy_data[idx][3] == -1:
                # Outer boundary: Use less simplification
                epsilon_factor = epsilon_factor_outer
            else:
                # Hole: Use more simplification
                epsilon_factor = epsilon_factor_hole

            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = epsilon_factor * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
            simplified_contours.append(approx_contour)

        color_data.append({
            'color': color_rgb,
            'contours': simplified_contours,
            'hierarchy': hierarchy_data
        })

    # 3. Geometric Preparation & CONTOUR ROTATION (Fix for hole check mismatch)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, hatch_angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, -hatch_angle, 1.0)

    rotated_color_data = []
    for data in color_data:
        rotated_contours_for_color = []
        for contour in data['contours']:
            contour_float = contour.astype(np.float32)
            rotated_contour = cv2.transform(contour_float, M)
            rotated_contours_for_color.append(rotated_contour)

        rotated_color_data.append({
            'color': data['color'],
            'contours': rotated_contours_for_color,  # NOW ROTATED
            'hierarchy': data['hierarchy']
        })

    # 4. Custom Line Fill using OPTIMIZED SVG PATH DATA
    dwg = svgwrite.Drawing('output_final_optimized.svg', size=(w, h), profile='tiny')

    for data in rotated_color_data:
        color_hex = rgb_to_hex(data['color'])
        contours = data['contours']
        hierarchy = data['hierarchy']
        outer_contours_indices = [idx for idx, h_data in enumerate(hierarchy) if h_data[3] == -1]

        # Initialize Path Data string for this color/area
        path_data_d = ""

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
                            # INVERSE TRANSFORMATION of Line Segment
                            points_to_transform = np.array([[start_x, y_line], [x_pos, y_line]],
                                                           dtype=np.float32).reshape(-1, 1, 2)
                            points_transformed = cv2.transform(points_to_transform, M_inv).reshape(-1, 2)

                            # SVG PATH DATA ACCUMULATION (Optimized M L commands)
                            start_pt = points_transformed[0]
                            end_pt = points_transformed[1]
                            path_data_d += f"M {start_pt[0]:.2f} {start_pt[1]:.2f} L {end_pt[0]:.2f} {end_pt[1]:.2f} "

                            is_drawing = False

                if is_drawing:
                    points_to_transform = np.array([[start_x, y_line], [x + bw, y_line]],
                                                   dtype=np.float32).reshape(-1, 1, 2)
                    points_transformed = cv2.transform(points_to_transform, M_inv).reshape(-1, 2)

                    # SVG PATH DATA ACCUMULATION (End of scan line)
                    start_pt = points_transformed[0]
                    end_pt = points_transformed[1]
                    path_data_d += f"M {start_pt[0]:.2f} {start_pt[1]:.2f} L {end_pt[0]:.2f} {end_pt[1]:.2f} "

        # FINAL: Add the single PATH element to the drawing for the entire color
        if path_data_d:
            dwg.add(dwg.path(
                d=path_data_d.strip(),
                stroke=color_hex,
                stroke_width=stroke_thickness,
                fill='none'
            ))

    dwg.save()
    print("Processing complete. Results saved to 'output_final_optimized.svg'")
    return quantized_image, color_palette


# Example Usage:
# INPUT_IMAGE = "logo0.bmp"
INPUT_IMAGE = "Mouse01.bmp"
vectorize_image_wu(INPUT_IMAGE, n_colors=15, hatch_angle=45.0, line_distance=3, stroke_thickness=0.5,
                   epsilon_factor_outer=0.001, epsilon_factor_hole=0.002)
