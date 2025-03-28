import json
def get_name_to_polys(label_txt_path):
	name_to_polys = {}
	with open(label_txt_path, 'r', encoding='utf-8') as f:
			for line in f:
				image_name = line.split('\t',1)[0]
				json_str = line.strip().split('\t', 1)[1]
				label_data = json.loads(json_str)
				polygons = [entry['points'] for entry in label_data]
				name_to_polys[image_name] = polygons
	return name_to_polys


import json
from typing import List, Dict, Any
import os

def process_polygons_to_label(
	polygons: List[List[List[float]]], 
	image_name: str, 
	file_path, 
	transcription: str = "hehe",
	difficult: bool = False
) -> None:
	"""
	Process polygons into a label dictionary and write to file.
	
	Args:
		polygons: List of polygons, where each polygon is a list of [x, y] coordinates
		image_name: Name of the image
		file_path: Open file object to write to
		transcription: Text label for each polygon (default: "hehe")
		difficult: Flag indicating if the detection is difficult (default: False)
	"""
  
	# Get unique coordinates using set comprehension for better performance
	# Convert inner arrays to tuples for hashability
	
	unique_coords = {tuple(tuple(point) for point in poly) for poly in polygons}
	
	# Build label dictionary using list comprehension with explicit type hints
	all_label_dict: List[Dict[str, Any]] = [
			{
				"transcription": transcription,
				# Convert numpy types to native Python types
				"points": [[int(point[0]), int(point[1])] for point in group],
				"difficult": difficult
			}
			for group in unique_coords
	]
	
	txt_dir = os.path.dirname(file_path)
	if txt_dir and not os.path.exists(txt_dir):
		os.makedirs(txt_dir, exist_ok=True)
	# Create empty file if it doesn't exist
	if not os.path.exists(file_path):
		open(file_path, 'a').close()
	
	# Write directly to file with proper error handling
	with open(file_path, 'a') as f:
		line = f"{image_name}\t{json.dumps(all_label_dict, ensure_ascii=False)}"
		f.write(line + "\n")
	

def get_label_map(label_txt_path):
	label_map = {}
	with open(label_txt_path, 'r', encoding='utf-8') as f:
			lines= f.readlines()
			for line in lines:
				img_name = line.strip().split('\t', 1)[0]
				json_str = line.strip().split('\t', 1)[1]
				label_data = json.loads(json_str)
				polygons = [entry['points'] for entry in label_data]
				label_map[img_name] = polygons
	return label_map
					