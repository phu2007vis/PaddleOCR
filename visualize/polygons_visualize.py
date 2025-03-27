import cv2
import numpy as np
import os

# Define the input path
input_path = "/work/21013187/phuoc/paddle_detect/data/testv1/20250308_140743_130origin.png"

# Define the output directory
output_dir = "/work/21013187/phuoc/paddle_detect/data/cropped_polygons"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn’t exist

# Define the bounding boxes (normalized coordinates)
screen1_box = [
     [[0.25, 0], [0.75, 0], [0.975, 0.95], [0.025, 0.95]],
    [[0.451769, 0.048466], [0.235031, 0.993690], [0.007846, 0.901081], [0.224584, -0.044144]],
    [[0.364600, 0.011600], [0.642000, 0.011600], [0.642000, 0.868067], [0.364600, 0.868067]],
    [[0.073650, 0.787067], [0.383800, 0.787067], [0.383800, 0.998600], [0.073650, 0.998600]],
    [[0.600350, 0.782607], [0.946200, 0.782607], [0.946200, 0.998600], [0.600350, 0.998600]],
    [[0.808787, 0.972856], [0.589139, 0.049634], [0.815283, -0.046016], [1.034931, 0.877206]]
]

# Load the image
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {input_path}")

# Get image dimensions
height, width = image.shape[:2]

# Process each polygon
for i, box in enumerate(screen1_box):
    # Convert normalized coordinates to pixel values
    pts = np.array([[int(x * width), int(y * height)] for x, y in box], dtype=np.int32)
    
    # Find the bounding rectangle of the polygon
    x, y, w, h = cv2.boundingRect(pts)
    
    # Ensure the bounding box stays within image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        print(f"Skipping polygon {i}: Invalid bounding box dimensions")
        continue
    
    # Crop the rectangular region
    cropped = image[y:y+h, x:x+w].copy()
    
    # Create a mask for the polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_pts = pts - [x, y]  # Shift points relative to the cropped region
    cv2.fillPoly(mask, [shifted_pts], 255)
    
    # Apply the mask to the cropped image (optional: keeps only the polygon area)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    
    # Save the cropped image
    output_path = os.path.join(output_dir, f"polygon_{i}.png")
    cv2.imwrite(output_path, cropped)
    print(f"Saved cropped image to {output_path}")

print(f"All cropped images saved to {output_dir}")