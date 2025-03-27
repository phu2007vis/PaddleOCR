import os 
label_path = "/work/21013187/phuoc/paddle_detect/data/testv1/Label.txt"
import json
count = 0

with open(label_path, 'r', encoding='utf-8') as f:
			for line in f:
				
				# Extract JSON part after tab
				json_str = line.strip().split('\t', 1)[1]
				label_data = json.loads(json_str)
				polygons = [entry['points'] for entry in label_data]
				count+= len(polygons)
print(count)
				