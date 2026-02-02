"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def detect_lines(img_path):
    
    # default_file = 'sudoku.png'
    # filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(img_path), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + img_path + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    cv.waitKey()
    return 0


def get_prioritized_x_axis(img_path, bucket_size=5, span_threshold=0.9):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    edges = cv.Canny(src, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    if lines is None: return None

    buckets = {}
    overall_max_span = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) <= 3:
            y_avg = (y1 + y2) // 2
            y_bucket = (y_avg // bucket_size) * bucket_size
            
            x_min, x_max = min(x1, x2), max(x1, x2)
            
            if y_bucket not in buckets:
                buckets[y_bucket] = [x_min, x_max]
            else:
                buckets[y_bucket][0] = min(buckets[y_bucket][0], x_min)
                buckets[y_bucket][1] = max(buckets[y_bucket][1], x_max)
            
            # Keep track of the absolute longest span found anywhere
            current_span = buckets[y_bucket][1] - buckets[y_bucket][0]
            if current_span > overall_max_span:
                overall_max_span = current_span

    # Iterate from highest Y (bottom of image) to lowest Y (top)
    # Sorting keys in reverse order
    sorted_y_buckets = sorted(buckets.keys(), reverse=True)

    for y in sorted_y_buckets:
        span = buckets[y][1] - buckets[y][0]
        # If this line is at least 90% as long as the longest line in the image
        if span >= (span_threshold * overall_max_span):
            return y

    return None

def get_topmost_horizontal_line(img_path, bucket_size=5, span_threshold=0.9):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    edges = cv.Canny(src, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    if lines is None: return None

    buckets = {}
    overall_max_span = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) <= 3:
            y_avg = (y1 + y2) // 2
            y_bucket = (y_avg // bucket_size) * bucket_size
            x_min, x_max = min(x1, x2), max(x1, x2)
            
            if y_bucket not in buckets:
                buckets[y_bucket] = [x_min, x_max]
            else:
                buckets[y_bucket][0] = min(buckets[y_bucket][0], x_min)
                buckets[y_bucket][1] = max(buckets[y_bucket][1], x_max)
            
            span = buckets[y_bucket][1] - buckets[y_bucket][0]
            overall_max_span = max(overall_max_span, span)

    # Sort ASCENDING: Start from y=0 (top) and move down
    sorted_y_buckets = sorted(buckets.keys())

    for y in sorted_y_buckets:
        span = buckets[y][1] - buckets[y][0]
        if span >= (span_threshold * overall_max_span):
            return y
    return None

def get_prioritized_y_axis(img_path, bucket_size=5, span_threshold=0.9):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    # Edge detection
    edges = cv.Canny(src, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    if lines is None: return None

    buckets = {}
    overall_max_span = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check if roughly vertical (x1 approx x2)
        if abs(x1 - x2) <= 3:
            # Normalize x to the nearest bucket
            x_avg = (x1 + x2) // 2
            x_bucket = (x_avg // bucket_size) * bucket_size
            
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            if x_bucket not in buckets:
                buckets[x_bucket] = [y_min, y_max]
            else:
                # Expand the vertical boundaries
                buckets[x_bucket][0] = min(buckets[x_bucket][0], y_min)
                buckets[x_bucket][1] = max(buckets[x_bucket][1], y_max)
            
            # Update the absolute longest vertical span found
            current_span = buckets[x_bucket][1] - buckets[x_bucket][0]
            if current_span > overall_max_span:
                overall_max_span = current_span

    # Iterate from lowest X (left side of image) to highest X (right side)
    # Sorting keys in ascending order (left to right)
    sorted_x_buckets = sorted(buckets.keys())

    for x in sorted_x_buckets:
        span = buckets[x][1] - buckets[x][0]
        # If this vertical line is at least 90% as long as the tallest line
        if span >= (span_threshold * overall_max_span):
            return x

    return None

def get_rightmost_vertical_line(img_path, bucket_size=5, span_threshold=0.9):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    edges = cv.Canny(src, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    if lines is None: return None

    buckets = {}
    overall_max_span = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) <= 3:
            x_avg = (x1 + x2) // 2
            x_bucket = (x_avg // bucket_size) * bucket_size
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            if x_bucket not in buckets:
                buckets[x_bucket] = [y_min, y_max]
            else:
                buckets[x_bucket][0] = min(buckets[x_bucket][0], y_min)
                buckets[x_bucket][1] = max(buckets[x_bucket][1], y_max)
            
            span = buckets[x_bucket][1] - buckets[x_bucket][0]
            overall_max_span = max(overall_max_span, span)

    # Sort DESCENDING: Start from x=Width (right) and move left
    sorted_x_buckets = sorted(buckets.keys(), reverse=True)

    for x in sorted_x_buckets:
        span = buckets[x][1] - buckets[x][0]
        if span >= (span_threshold * overall_max_span):
            return x
    return None

def get_x_axis_limits(img_path, x_axis_y, y_axis_x,tick_size=10, density_threshold=0.5):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    # Threshold to binary (black/white) to make density counting easy
    _, binary = cv.threshold(src, 150, 255, cv.THRESH_BINARY_INV)

    height, width = binary.shape
    marking_x_coords = []

    # Define the vertical range to check for ticks (e.g., 10px below x_axis_y)
    y_start = max(0, x_axis_y)
    y_end = min(height, x_axis_y + tick_size)

    search_start = y_axis_x + 10
    search_end = width - 2

    # Iterate through every x column
    for x in range(search_start, search_end):
        # Extract the vertical slice at this x
        column_slice = binary[y_start:y_end, x]
        
        # Calculate density (percentage of black pixels in this slice)
        density = np.sum(column_slice > 0) / tick_size
        
        if density > density_threshold:
            marking_x_coords.append(x)

    if not marking_x_coords:
        return None, None

    # print(marking_x_coords)
    # Group adjacent x-coordinates that belong to the same tick mark
    # and take the average/midpoint of each cluster
    final_ticks = []
    if marking_x_coords:
        current_cluster = [marking_x_coords[0]]
        for i in range(1, len(marking_x_coords)):
            if marking_x_coords[i] - marking_x_coords[i-1] <= 2: # pixels apart
                current_cluster.append(marking_x_coords[i])
            else:
                final_ticks.append(sum(current_cluster) // len(current_cluster))
                current_cluster = [marking_x_coords[i]]
        final_ticks.append(sum(current_cluster) // len(current_cluster))

    # The min and max markings are the first and last detected ticks
    return min(final_ticks), max(final_ticks)

def get_y_axis_limits(img_path, x_axis_y, y_axis_x, tick_size=10, density_threshold=0.5):
    src = cv.imread(cv.samples.findFile(str(img_path)), cv.IMREAD_GRAYSCALE)
    if src is None: return None

    # Threshold to binary (inverted: markings are > 0)
    _, binary = cv.threshold(src, 150, 255, cv.THRESH_BINARY_INV)
    height, width = binary.shape
    marking_y_coords = []

    # Define the horizontal range to check for ticks 
    # (Checking from the y-axis line towards the left)
    x_start = max(0, y_axis_x - tick_size)
    x_end = min(width, y_axis_x)

    # Search from the top of the image down to just before the x-axis intersection
    search_start = 2
    search_end = x_axis_y - 10 

    # Iterate through every y row
    for y in range(search_start, search_end):
        # Extract the horizontal slice at this y
        row_slice = binary[y, x_start:x_end]
        
        # Calculate density (percentage of black pixels in this slice)
        density = np.sum(row_slice > 0) / tick_size
        
        if density > density_threshold:
            marking_y_coords.append(y)

    if not marking_y_coords:
        return None, None

    # Group adjacent y-coordinates into clusters
    final_ticks = []
    current_cluster = [marking_y_coords[0]]
    for i in range(1, len(marking_y_coords)):
        if marking_y_coords[i] - marking_y_coords[i-1] <= 2:
            current_cluster.append(marking_y_coords[i])
        else:
            final_ticks.append(sum(current_cluster) // len(current_cluster))
            current_cluster = [marking_y_coords[i]]
    final_ticks.append(sum(current_cluster) // len(current_cluster))

    # Note: min(final_ticks) is the top-most marking (highest value)
    # max(final_ticks) is the bottom-most marking (lowest value)
    return min(final_ticks), max(final_ticks)

def get_pixel_coordinates(img_path):
    origin_x, origin_y = get_prioritized_y_axis(img_path), get_prioritized_x_axis(img_path)
    terminal_x, terminal_y = get_rightmost_vertical_line(img_path), get_topmost_horizontal_line(img_path)
    x_min, x_max = get_x_axis_limits(img_path, origin_y, origin_x)
    y_min, y_max = get_y_axis_limits(img_path, origin_y, origin_x)

    return {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "terminal_x": terminal_x,
        "terminal_y": terminal_y,
        "px_x_min": x_min,
        "px_x_max": x_max,
        "px_y_min": y_max,  ## Note: y-axis inverted in images (top to bottom is 0 to H)
        "px_y_max": y_min
    }

if __name__ == "__main__":
    detect_lines("../ground_truth/raman1.png")

    print("Pixel Coordinates:", get_pixel_coordinates("../ground_truth/raman1.png"))