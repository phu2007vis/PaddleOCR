
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from evaluate_phuoc.utils_evalute_phuoc import process_in_batches
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
from tools.slice_image import SliceImage
import cv2
from tools.merge_image import Merger


def draw_det_res(dt_boxes, img, img_name, save_path,color = (255, 255, 0)):

	src_im = img
	for box in dt_boxes:
		box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(src_im, [box], True, color, thickness=2)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path, os.path.basename(img_name))
	cv2.imwrite(save_path, src_im)
	logger.info("The detected Image saved in {}".format(save_path))


@paddle.no_grad()
def main():
	slicer_config = config['ImageSlicer']
	image_slicer = SliceImage(**slicer_config)
	merger = Merger()
	global_config = config["Global"]

	# build model
	model = build_model(config["Architecture"])

	load_model(config, model)
	# build post process
	print(config['PostProcess'])
	post_process_class = build_post_process(config["PostProcess"])

	# create data ops
	transforms = []
	for op in config["Eval"]["dataset"]["transforms"]:
		op_name = list(op)[0]
		if "Label" in op_name:
			continue
		elif op_name == "KeepKeys":
			op[op_name]["keep_keys"] = ["image", "shape"]
		transforms.append(op)

	ops = create_operators(transforms, global_config)

	save_res_path = config["Global"]["save_res_path"]
	if not os.path.exists(os.path.dirname(save_res_path)):
		os.makedirs(os.path.dirname(save_res_path))
  
	save_det_path = (
					os.path.dirname(config["Global"]["save_res_path"]) + "/det_results/"
				)
	model.eval()
	with open(save_res_path, "wb") as fout:
		for file in get_image_file_list(config["Global"]["infer_img"]):
			logger.info("infer_img: {}".format(file))
			image = cv2.imread(file)
			images,locations = image_slicer(image)
			images_preprocessed =[]
			shape_list_preprocessed = []
			for img in images:
				data = {"image": img}
				batch = transform(data, ops)
				images_preprocessed.append(batch[0])
				shape_list_preprocessed.append(batch[1])
				
			image = cv2.imread(file)
			original_image = np.copy(image)
			images,locations = image_slicer(image)
   
			final_box,post_result = process_in_batches(images_preprocessed,shape_list_preprocessed,model,post_process_class,merger,locations,batch_size=40)
			image_size = (original_image.shape[1],original_image.shape[1])
			final_box = merger.convert_to_four_points(final_box)
			dbscan_final_box = merger.dbcan_merge(final_box,image_size = image_size)
			
			
			db_visualize = merger.db_scan.visualize(original_image)
			draw_det_res(final_box, np.copy(original_image), file, os.path.join(save_det_path,'final'))
			
			draw_det_res(dbscan_final_box, original_image, file, os.path.join(save_det_path,'dbscan2'),color = (0,255,0))
			
		
			dt_boxes_json = []
			# parser boxes if post_result is dict
			file_id = os.path.splitext(file)[0]

			
			# draw_det_res([],db_visualize,file,os.path.join(save_det_path,'final_dbscan'))
			for i,(src_img,boxes) in enumerate(zip(images,post_result)):
				file_name = f"{file_id}_{i}.png"
				boxes = post_result[i]["points"]
				dt_boxes_json = []
				# write result
				for box in boxes:
					tmp_json = {"transcription": ""}
					tmp_json["points"] = np.array(box).tolist()
					dt_boxes_json.append(tmp_json)
				

				draw_det_res(boxes, src_img, file_name, os.path.join(save_det_path,'crop_result'))
			
			otstr = file_name + "\t" + json.dumps(dt_boxes_json) + "\n"
			fout.write(otstr.encode())

	logger.info("success!")


if __name__ == "__main__":
	config, device, logger, vdl_writer = program.preprocess()
	main()
