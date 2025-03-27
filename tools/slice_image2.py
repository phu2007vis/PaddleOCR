

import matplotlib.pyplot as plt
import os
import paddle
import numpy as np
import json
import os
import cv2
from typing import Optional,List

import sys
from merge_image import Merger

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ppocr.phuoc import *
merger = Merger()



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

		
def visualize_poly_image(
	image_name: str,
	label_txt_path: str,
	image_dir: str,
	color: tuple = (0, 255, 0),  # Green in BGR
	thickness: int = 2,
	**kwargs
) -> Optional[np.ndarray]:
	
	
		image_path = os.path.join(image_dir, image_name)

		# Read the image
		image = cv2.imread(image_path)
		original_image =np.copy(image)
		# Find matching line in label file
		polygons = None
		with open(label_txt_path, 'r', encoding='utf-8') as f:
			for line in f:
				if line.strip().startswith(image_name):
					# Extract JSON part after tab
					json_str = line.strip().split('\t', 1)[1]
					label_data = json.loads(json_str)
					polygons = [entry['points'] for entry in label_data]
					break
		# all_polygons = []
		# for poly in polygons:
		# 	all_polygons.append(quadrilateral_to_rectangle(poly))
		images,all_polygons,all_slice_bounds = process_image_and_labels_v2(image,polygons,**kwargs)
		all_images = []
		for image, polygons in zip(images,all_polygons):
		# Draw polygons on image
			for poly in polygons:
				# Convert to numpy array and reshape for cv2.polylines
				points = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
				cv2.polylines(
					image,
					[points],
					isClosed=True,
					color=color,
					thickness=thickness
				)
			all_images.append(image)
		all_images = [image for i,image in enumerate(all_images) if  len(all_polygons[i] )!=0]
		return original_image,all_images,all_polygons,all_slice_bounds



# Example usage
if __name__ == "__main__":
	# Example parameters
	img_name = None
	label_path = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2/labels.txt"
	img_dir = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2"
	save_folder = "/work/21013187/phuoc/paddle_detect/data/crop_visualization"
	save_txt_path = ""
	remove_exists = True
	save_crop_folder = os.path.join(save_folder,'crop')
	save_merge_folder = os.path.join(save_folder,'merge')
	visualize = True
	import shutil
	if remove_exists:
		shutil.rmtree(save_folder)
	os.makedirs(save_folder,exist_ok=True)
	os.makedirs(save_crop_folder,exist_ok=True)
	os.makedirs(save_merge_folder,exist_ok=True)
 
	# Get visualized image
	from tqdm import tqdm
	for image_name in tqdm(os.listdir(img_dir),total = len(os.listdir(img_dir))):
		image_id = os.path.splitext(image_name)[0]
		original_image,images,all_polygons,all_slice_bounds = visualize_poly_image(image_name, label_path, img_dir,slice_width=320, slice_height=320, overlap=0.3)
		for i,image in enumerate(images):
			new_image_name = f"{image_id}_{i}.jpg"
			image_path = os.path.join(save_crop_folder,new_image_name)
		
			cv2.imwrite(image_path, image)
		# merge_image = merger(original_image,all_polygons,all_slice_bounds)
		# image_path = os.path.join(save_merge_folder,f"merge_image_{image_id}.jpg")
		# cv2.imwrite(image_path,merge_image)
		



