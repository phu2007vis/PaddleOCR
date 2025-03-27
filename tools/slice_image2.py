

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
import shutil
from tqdm import tqdm
import cv2
import paddle
import numpy as np
import json
from typing import Optional, List
from tools.convert_polys import process_polygons_to_label

from merge_image import Merger
from ppocr.phuoc import process_image_and_labels_v2









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
    visualize = None,
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
		# all_polygons = [np.array(poly, dtype=np.int32).reshape((-1, 2)) if len(poly) > 0 else np.array([], dtype=np.int32).reshape(0, 2) for poly in polygons]
		new_all_polygons = []
		for i,polys in enumerate(all_polygons):
			n = len(polys)
			if n <=0:
				continue
			polys = [np.array(poly, dtype=np.int32).reshape(-1,2) for poly in polygons]
			new_all_polygons.append(polys)
			
		all_images = []
		for image, polygons in zip(images,all_polygons):
			if visualize:
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
		return original_image,all_images,new_all_polygons,all_slice_bounds





def parse_args():
    parser = argparse.ArgumentParser(description='Process and visualize image slices with polygons')
    parser.add_argument('--label_path', 
                       help='Path to the labels file')
    parser.add_argument('--img_dir',
                       help='Directory containing input images')
    parser.add_argument('--save_folder',
                       default="/work/21013187/phuoc/paddle_detect/data/crop_visualization",
                       help='Base folder for saving results')
    parser.add_argument('--save_txt_path',
                       default="",
                       help='Path to save text output')
    parser.add_argument('--remove_exists',
                       action='store_true',
                       help='Remove existing save folder if True')
    parser.add_argument('--visualize',
                       action='store_true',
                       help='Enable visualization')
    parser.add_argument('--slice_width',
                       type=int,
                       default=320,
                       help='Width of image slices')
    parser.add_argument('--slice_height',
                       type=int,
                       default=320,
                       help='Height of image slices')
    parser.add_argument('--overlap',
                       type=float,
                       default=0.3,
                       help='Overlap ratio between slices')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set up directories
    save_crop_folder = os.path.join(args.save_folder, 'crop')
    save_merge_folder = os.path.join(args.save_folder, 'merge')
    visualize = args.visualize

    # Clean and create directories
    if args.remove_exists and os.path.exists(args.save_folder):
        shutil.rmtree(args.save_folder)
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(save_crop_folder, exist_ok=True)
    os.makedirs(save_merge_folder, exist_ok=True)

    # Process images
    for image_name in tqdm(os.listdir(args.img_dir), total=len(os.listdir(args.img_dir))):
        image_id = os.path.splitext(image_name)[0]
        original_image, images, all_polygons, all_slice_bounds = visualize_poly_image(
            image_name, 
            args.label_path, 
            args.img_dir,
            slice_width=args.slice_width,
            slice_height=args.slice_height,
            overlap=args.overlap,
			visualize = visualize
        )
       
        for i, image in enumerate(images):
            new_image_name = f"{image_id}_{i}.jpg"
            image_path = os.path.join(save_crop_folder, new_image_name)
            
            cv2.imwrite(image_path, image)
            process_polygons_to_label(all_polygons[i], new_image_name, args.save_txt_path)
        # merger = Merger()
		# merge_image = merger(original_image,all_polygons,all_slice_bounds)
		# image_path = os.path.join(save_merge_folder,f"merge_image_{image_id}.jpg")
		# cv2.imwrite(image_path,merge_image)
		



