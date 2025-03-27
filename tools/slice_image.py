

import matplotlib.pyplot as plt
import os
import paddle
import numpy as np
import json
import os
import cv2
from typing import Optional,List
from shapely.geometry import Polygon

import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ppocr.phuoc import *

def quadrilateral_to_rectangle(points):
	# Ensure input is a numpy array with 4 points
	points = np.array(points)
	if points.shape != (4, 2):
		raise ValueError("Input must be 4 points with x, y coordinates")

	# Step 1: Calculate the centroid
	centroid = np.mean(points, axis=0)

	# Step 2: Determine orientation from the first side (P0 to P1)
	v1 = points[1] - points[0]  # Vector from P0 to P1
	length_v1 = np.linalg.norm(v1)
	v1_normalized = v1 / length_v1  # Unit vector for direction

	# Perpendicular vector (rotate 90 degrees counterclockwise)
	v2_normalized = np.array([-v1_normalized[1], v1_normalized[0]])

	# Step 3: Estimate rectangle dimensions
	# Width: average length of P0P1 and P2P3
	width = (np.linalg.norm(points[1] - points[0]) + 
			 np.linalg.norm(points[3] - points[2])) / 2

	# Height: average length of P1P2 and P3P0
	height = (np.linalg.norm(points[2] - points[1]) + 
			  np.linalg.norm(points[0] - points[3])) / 2

	# Step 4: Define the four corners of the rectangle
	# Start from centroid and extend along v1 and v2 directions
	half_width = width / 2
	half_height = height / 2

	# Rectangle points: centroid ± (width/2 * v1) ± (height/2 * v2)
	rect_points = np.array([
		centroid - half_width * v1_normalized - half_height * v2_normalized,
		centroid + half_width * v1_normalized - half_height * v2_normalized,
		centroid + half_width * v1_normalized + half_height * v2_normalized,
		centroid - half_width * v1_normalized + half_height * v2_normalized
	])

	return rect_points



def get_slice_bboxes(
	image_height: int,
	image_width: int,
	slice_height: Optional[int] = None,
	slice_width: Optional[int] = None,
	overlap_height_ratio: float = 0.2,
	overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
   
	slice_bboxes = []
	y_max = y_min = 0

  
	y_overlap = int(overlap_height_ratio * slice_height)
	x_overlap = int(overlap_width_ratio * slice_width)
	

	while y_max < image_height:
		x_min = x_max = 0
		y_max = y_min + slice_height
		while x_max < image_width:
			x_max = x_min + slice_width
			if y_max > image_height or x_max > image_width:
				xmax = min(image_width, x_max)
				ymax = min(image_height, y_max)
				xmin = max(0, xmax - slice_width)
				ymin = max(0, ymax - slice_height)
				slice_bboxes.append([xmin, ymin, xmax, ymax])
			else:
				slice_bboxes.append([x_min, y_min, x_max, y_max])
			x_min = x_max - x_overlap
		y_min = y_max - y_overlap
	return slice_bboxes

def denormalize_image(image,
					  scale = 1.0,
					  mean = paddle.to_tensor([0.485, 0.456, 0.406], dtype='float32'),
					  std = paddle.to_tensor([0.229, 0.224, 0.225], dtype='float32')
					  ):

	if isinstance(image,np.ndarray):
		image = paddle.to_tensor(image)
	if image.shape[0] == 3:
		image = paddle.transpose(image, perm=[1, 2, 0])
	scale = 1.0 / scale
	mean = mean.reshape([1, 1, -1])
	std = std.reshape([1, 1, -1])
	denorm_image = (image * std + mean) / scale
	if scale == 1/255.0:
		return paddle.to_tensor(paddle.clip(denorm_image, 0, 255))
	return paddle.clip(denorm_image, 0, 1)
class SliceImage:
	def __init__(self,
				 slice_height, 
				 slice_width,
	 			 overlap):
		self.slice_height = slice_height
		self.slice_width = slice_width
		self.overlap = overlap
	def __call__(self, image):
		height,width = image.shape[0], image.shape[1]
		x_coords, y_coords, slice_width, slice_height = get_slice_parameters(width, height, 
																	   slice_width=self.slice_width,
																	   slice_height=self.slice_height, 
																	   overlap=self.overlap)
		# Store results
		slices = []
		all_slice_bounds = []
		# Slice and adjust labels
		for y in y_coords:
			for x in x_coords:
				slice_img, slice_bounds = slice_image(image, 
                                          				x,
                                              			y,
                                                 		slice_width,
                                                   		slice_height,
                                                     	width,
                                                      	height)
				slices.append(np.copy(slice_img))
				all_slice_bounds.append(slice_bounds)
			
		return slices,all_slice_bounds




# Example usage
if __name__ == "__main__":
	# Example parameters
	img_name = None
	label_path = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2/labels.txt"
	img_dir = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2"
	save_folder = "/work/21013187/phuoc/paddle_detect/data/crop_visualization"
	save_crop_folder = os.path.join(save_folder,'crop')
	save_merge_folder = os.path.join(save_folder,'merge')
	visualize = True
	import shutil
	shutil.rmtree(save_folder)
	os.makedirs(save_folder,exist_ok=True)
	os.makedirs(save_crop_folder,exist_ok=True)
	os.makedirs(save_merge_folder,exist_ok=True)
	from merge_image import Merger
	merger = Merger()
	# Get visualized image
	from tqdm import tqdm
	for image_name in tqdm(os.listdir(img_dir),total = len(os.listdir(img_dir))):
		image_id = os.path.splitext(image_name)[0]

		original_image,images,all_polygons,all_slice_bounds = visualize_poly_image(image_name, label_path, img_dir,slice_width=320, slice_height=320, overlap=0.3)
		for i,image in enumerate(images):
			new_image_name = f"{image_id}_{i}.jpg"
			image_path = os.path.join(save_crop_folder,new_image_name)
			cv2.imwrite(image_path, image)
		merge_image = merger(original_image,all_polygons,all_slice_bounds)
		image_path = os.path.join(save_merge_folder,f"merge_image_{image_id}.jpg")
		cv2.imwrite(image_path,merge_image)
		



