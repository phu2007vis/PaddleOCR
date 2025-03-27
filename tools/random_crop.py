import os
import sys
import argparse
from multiprocessing import Pool
from functools import partial

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import tools.program as program
from ppocr.data.simple_dataset import SimpleDataSet
from visualize import denormalize_image
import numpy as np
import cv2
from tqdm import tqdm
import random
from tools.convert_polys import process_polygons_to_label

random.seed(42)

class Cropper:
    def __init__(self, config, save_folder, max_number_images, visualize=False,number_of_processes = 1):
        self.config = config
        self.dataset = SimpleDataSet(config, mode='Train', logger=None)
        self.max_number_images = max_number_images
        self.save_folder = save_folder
        self.visualize = visualize
        os.makedirs(save_folder, exist_ok=True)
        self.txt_path = os.path.join(save_folder, 'Label.txt')
        self.number_of_processes = number_of_processes
    def get_image(self, index):
        data = self.dataset[index]
        return denormalize_image(data[0], scale=255.0).numpy().astype(np.uint8), np.array(data[-1], dtype=np.int32)
    
    def process_image(self, i):
        index = random.randint(0, len(self.dataset)-1)
        image, polys = self.get_image(index)
        image_name = f'image_{i}.jpg'
        
        if self.visualize:
            cv2.polylines(image, polys, isClosed=True, color=(0, 255, 0), thickness=2)
            
        # Save the image
        cv2.imwrite(os.path.join(self.save_folder, image_name), image)
        
        # Process polygons to label - using file locking for thread safety
        process_polygons_to_label(polys, image_name, self.txt_path)
        
        return True

    def main(self):
        # Determine number of processes (using CPU count by default)
        num_processes = min(self.number_of_processes, self.max_number_images)
        print(f'Number of proceses: ',num_processes)
        # Create a pool of workers
        with Pool(processes=num_processes) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(
                    self.process_image, 
                    range(self.max_number_images)
                ),
                total=self.max_number_images
            ))

def parse_args():
    parser = argparse.ArgumentParser(description='Image Cropper with Polygon Visualization')
    parser.add_argument('--config', 
                       type=str, 
                       default='/work/21013187/phuoc/paddle_detect/config/extract_config.yaml',
                       help='Path to config file')
    parser.add_argument('--save_folder', 
                       type=str, 
                       default='/work/21013187/phuoc/paddle_detect/data/testv2',
                       help='Folder to save output images')
    parser.add_argument('--max_images', 
                       type=int, 
                       default=400,
                       help='Maximum number of images to process')
    parser.add_argument('--number_of_processes', 
                       type=int, 
                       default=3,
                       help='Maximum number of images to process')
    parser.add_argument('--visualize', 
                       action='store_true',
                       help='Whether to visualize polygons on images')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = program.load_config(args.config)
    cropper = Cropper(
        config=config,
        save_folder=args.save_folder,
        max_number_images=args.max_images,
        visualize=args.visualize,
        number_of_processes=args.number_of_processes
    )
    cropper.main()