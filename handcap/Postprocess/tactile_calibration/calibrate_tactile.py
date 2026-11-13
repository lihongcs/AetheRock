import os
import cv2
import numpy as np

def _line_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
    x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
    
    det1 = x1*y2 - y1*x2
    det2 = x3*y4 - y3*x4
    dx1, dy1 = x1 - x2, y1 - y2
    dx2, dy2 = x3 - x4, y3 - y4
    
    denom = (dx1 * dy2) - (dy1 * dx2)
    if denom == 0:
        return None
    px = (det1 * dx2 - dx1 * det2) / denom
    py = (det1 * dy2 - dy1 * det2) / denom
    return (px, py)

def compute_calibration_params(reference_image):
    height, width = reference_image.shape[:2]
    
        
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the reference image")
        
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    pts = approx.reshape(-1, 2)
    valid_lines = []
    margin = 5
    
    for i in range(len(pts)):
        pt1 = pts[i]
        pt2 = pts[(i+1)%len(pts)]
        if abs(pt1[0] - pt2[0]) < margin and (pt1[0] <= margin or pt1[0] >= width - margin):
            continue
        if abs(pt1[1] - pt2[1]) < margin and (pt1[1] <= margin or pt1[1] >= height - margin):
            continue
        on_border = False
        for coord, max_val in [(0, width), (1, height)]:
            if (pt1[coord] <= margin and pt2[coord] <= margin) or (pt1[coord] >= max_val - margin and pt2[coord] >= max_val - margin):
                on_border = True
                break
        if not on_border:
            valid_lines.append((pt1, pt2))
            
    if len(valid_lines) < 6:
        raise ValueError(f"Could not extract exactly 6 valid lines! Found {len(valid_lines)}")
    elif len(valid_lines) > 6:
        # Keep the 6 longest lines to filter out small noisy edges or flattened corners
        line_lengths = [np.hypot(p1[0]-p2[0], p1[1]-p2[1]) for p1, p2 in valid_lines]
        longest_indices = sorted(np.argsort(line_lengths)[-6:])
        valid_lines = [valid_lines[i] for i in longest_indices]
        
    true_vertices = []
    for i in range(6):
        intersect = _line_intersection(valid_lines[i], valid_lines[(i+1)%6])
        if intersect is None:
            raise ValueError("Adjacent lines are parallel.")
        true_vertices.append(intersect)
        
    true_vertices = np.array(true_vertices)
    cx = np.mean(true_vertices[:, 0])
    cy = np.mean(true_vertices[:, 1])
    
    angles = np.arctan2(true_vertices[:, 1] - cy, true_vertices[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    sorted_vertices = true_vertices[sorted_idx]
    
    lengths = []
    for i in range(6):
        p1 = sorted_vertices[i]
        p2 = sorted_vertices[(i+1)%6]
        lengths.append(np.hypot(p1[0]-p2[0], p1[1]-p2[1]))
        
    L_avg = np.mean(lengths)


    L_horizontal = width / np.sqrt(3)
    L_vertical = height / 2.0
    TARGET_L = min(L_horizontal, L_vertical)
    
    target_cx = width / 2.0
    target_cy = height / 2.0
    
    ideal_pts = np.array([
        [target_cx - TARGET_L*np.sqrt(3)/2, target_cy - TARGET_L/2], 
        [target_cx, target_cy - TARGET_L],                         
        [target_cx + TARGET_L*np.sqrt(3)/2, target_cy - TARGET_L/2], 
        [target_cx + TARGET_L*np.sqrt(3)/2, target_cy + TARGET_L/2], 
        [target_cx, target_cy + TARGET_L],                         
        [target_cx - TARGET_L*np.sqrt(3)/2, target_cy + TARGET_L/2]  
    ], dtype=np.float32)

    H, _ = cv2.findHomography(np.array(sorted_vertices, dtype=np.float32), ideal_pts)
    ideal_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(ideal_mask, [np.int32(ideal_pts)], 255)
    
        
    return H, ideal_mask, width, height

def apply_calibration(image, params, interpolation=cv2.INTER_LANCZOS4):
    H, ideal_mask, out_w, out_h = params
    warped = cv2.warpPerspective(image, H, (out_w, out_h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    final_img = cv2.bitwise_and(warped, warped, mask=ideal_mask)
    return final_img

def calibrate_tactile_sensor(input_data):
    """
    Calibrate and transform a tactile sensor image.
    
    Args:
        input_data (str or np.ndarray): File path to an image or a preloaded BGR image array.

    Returns:
        np.ndarray: The corrected image.
    """
    is_np_array = isinstance(input_data, np.ndarray)
    
    if is_np_array:
        image = input_data
    else:
        image = cv2.imread(input_data)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {input_data}")
            
    params = compute_calibration_params(image)
    corrected_img = apply_calibration(image, params)
    
    return corrected_img

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Correct a tactile sensor image.")
    parser.add_argument("input", help="Path to a tactile image")
    parser.add_argument("output", help="Path for the corrected image")
    args = parser.parse_args()
    if not cv2.imwrite(args.output, calibrate_tactile_sensor(args.input)):
        raise OSError(f"Could not save the corrected image to {args.output}")
