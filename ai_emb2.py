import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite


# Helper function to convert RGB array to SVG hex color string
# def rgb_to_hex(rgb):
#     return f'#{rgb[0]:02x} {rgb[1]:02x} {rgb[2]:02x}'

# CORRECTED CODE:
def rgb_to_hex(rgb):
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
# This produces strings like '#131e19'
def vectorize_image_wu(image_path, n_colors=8, line_distance=5):
    # --- 1. Load Image and 2. Quantize (Omitted for brevity, assume previous setup) ---
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not loaded.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, d = image_rgb.shape
    pixels = image_rgb.reshape((-1, d))

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_pixels.reshape((h, w, d)).astype(np.uint8)
    color_palette = kmeans.cluster_centers_.astype(np.uint8)

    color_data = []

    # --- 3. Masking & 4. Contour Extraction (Omitted for brevity, assume previous setup) ---
    for i, color_rgb in enumerate(color_palette):
        mask_boolean_3d = (quantized_image == color_rgb)
        mask_2d = np.all(mask_boolean_3d, axis=2)
        mask = mask_2d.astype(np.uint8) * 255

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,  # Essential for getting hierarchy (parent/child) info
            cv2.CHAIN_APPROX_SIMPLE
        )

        color_data.append({
            'color': color_rgb,
            'contours': contours,
            'hierarchy': hierarchy[0] if hierarchy is not None else []
        })

    # --- Hole Exclusion Helper Function ---
    def is_inside_hole(point, outer_contour_idx, all_contours, all_hierarchy):
        """Checks if a point is inside any hole associated with the given outer_contour_idx."""

        # Iterate over all contours
        for j, h_data in enumerate(all_hierarchy):
            # h_data[3] is the parent index. If it matches the outer contour's index, it's a hole.
            if h_data[3] == outer_contour_idx:
                hole_contour = all_contours[j]

                # Check if the point is inside the hole (0 or positive result)
                if cv2.pointPolygonTest(hole_contour, point, measureDist=False) >= 0:
                    return True  # It IS inside a hole
        return False  # It is NOT inside any hole

    # --- 5. Apply Custom Line Fill and 6. Save to Vector File (SVG) ---
    dwg = svgwrite.Drawing('output.svg', size=(w, h), profile='tiny')

    print("Applying custom line fill (hatching) and saving to SVG...")

    for data in color_data:
        color_hex = rgb_to_hex(data['color'])
        contours = data['contours']
        hierarchy = data['hierarchy']

        # Get indices of the OUTER contours (those with no parent: hierarchy[3] == -1)
        outer_contours_indices = [
            idx for idx, h_data in enumerate(hierarchy)
            if h_data[3] == -1
        ]

        for idx in outer_contours_indices:
            outer_contour = contours[idx]

            # Find the bounding box of the outer contour
            x, y, bw, bh = cv2.boundingRect(outer_contour)

            # Iterate through horizontal lines inside the bounding box
            for y_line in range(y, y + bh, line_distance):
                is_drawing = False
                start_x = -1

                # Scan the line segment horizontally
                for x_pos in range(x, x + bw):
                    point = (x_pos, y_line)

                    # 1. Check if inside the OUTER contour
                    is_inside_outer = cv2.pointPolygonTest(outer_contour, point, measureDist=False) >= 0

                    # 2. Check if outside ALL HOLES associated with this outer contour
                    # We pass the index of the outer contour (idx) to find its children (holes)
                    is_outside_holes = not is_inside_hole(point, idx, contours, hierarchy)

                    # A point is valid for drawing if it's inside the outer shape AND outside all holes
                    is_valid_point = is_inside_outer and is_outside_holes

                    if is_valid_point:
                        if not is_drawing:
                            # Start a new line segment
                            is_drawing = True
                            start_x = x_pos
                    else:
                        if is_drawing:
                            # End the line segment and draw it
                            dwg.add(dwg.line(
                                start=(start_x, y_line),
                                end=(x_pos, y_line),
                                stroke=color_hex,
                                stroke_width=1
                            ))
                            is_drawing = False

                # If the line ends at the edge of the bounding box while drawing
                if is_drawing:
                    dwg.add(dwg.line(
                        start=(start_x, y_line),
                        end=(x + bw, y_line),
                        stroke=color_hex,
                        stroke_width=1
                    ))

    dwg.save()
    print(f"Processing complete. Results saved to 'output.svg' with line distance: {line_distance}")

    return quantized_image, color_palette

# Example usage (assuming 'input.jpg' exists in the script directory):
print("Starting vectorization...")
vectorize_image_wu('logo0.bmp', line_distance=5)