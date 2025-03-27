
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
from tools.json_ops import get_name_to_polys	
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
from evaluate_phuoc.phuoc_iou_metrics import DetectionIoUEvaluator
from evaluate_phuoc.utils_evalute_phuoc import process_in_batches

evaluator = DetectionIoUEvaluator()
def draw_det_res(dt_boxes, img, img_name, save_path):

	src_im = img
	for box in dt_boxes:
		box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
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
	txt_label_path = config['Eval']['dataset']['label_file_list']
	name2polys = get_name_to_polys(txt_label_path)
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


	model.eval()
	all_file = get_image_file_list(config["Global"]["infer_img"])
	for file in tqdm(all_file,total = len(all_file)):
		logger.info("infer_img: {}".format(file))
		original_image = cv2.imread(file)
		images,locations = image_slicer(original_image)
		

		draw_image = np.copy(original_image)
		images_preprocessed =[]
		shape_list_preprocessed = []
		for img in images:
			data = {"image": img}
			batch = transform(data, ops)
			images_preprocessed.append(batch[0])
			shape_list_preprocessed.append(batch[1])
			
		# images = np.stack(images_preprocessed, axis=0)  # Shape: (batch_size, H, W, C)
		# shape_list = np.stack(shape_list_preprocessed, axis=0)
		final_box,_ = process_in_batches(images_preprocessed,shape_list_preprocessed,model,post_process_class,merger,locations,batch_size=40)
		# images = paddle.to_tensor(images)
		# preds = model(images)
		
		# post_result = post_process_class(preds, shape_list)
		# phuoc_post_result = [p['points'] for p in post_result]
		# final_box = merger(polygons=phuoc_post_result,slice_locations=locations)
		image_name = os.path.basename(file)
		gt_box = name2polys[image_name]

		# merger.visualize(draw_image,gt_box,(255,0,0))
		merger.visualize(draw_image,final_box)
		
		
		evaluator.feetch_data(gt_box,final_box)
	evaluator.compute_results()
   
	


if __name__ == "__main__":
	config, device, logger, vdl_writer = program.preprocess()
	main()
