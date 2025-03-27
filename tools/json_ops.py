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