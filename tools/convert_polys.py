import json
from typing import List, Dict, Any


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

	
	# Write directly to file with proper error handling
	with open(file_path, 'a') as f:
		line = f"{image_name}\t{json.dumps(all_label_dict, ensure_ascii=False)}"
		f.write(line + "\n")
	


