
import matplotlib.pyplot as plt
import os
import paddle
import numpy as np
import json
import os
import cv2
from typing import Optional

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

def visualize(dataloader, name, save_folder=None, num_images=4,**kwargs):
	# Calculate rows needed: one row per image, 3 columns for 3 types
	rows = num_images  # Each image gets its own row
	cols = 3           # One column per image type
	
	# Create subplot grid
	fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Width=15, Height=5 per row
	plt.subplots_adjust(wspace=0.3, hspace=0.5)
	
	# If num_images=1, axs might not be a 2D array; handle this case
	if num_images == 1:
		axs = np.array([axs])  # Convert to 2D array for consistent indexing
	from tqdm import tqdm
	n = 0
	for data in tqdm(dataloader,total = len(dataloader)):
	
			img_org = data[0]          # Original RGB image (BxCxHxW)
			threshold_map_org = data[1] # Threshold map (BxHxW or similar)
			shrink_map = data[3]        # Shrink map (BxHxW or similar)
			
			batch_size = img_org.shape[0]
			for i in range(min(batch_size, num_images - n)):
				# Original RGB Image
				img = denormalize_image(paddle.transpose(img_org[i], perm=[1, 2, 0]),**kwargs).numpy()[:,:,::-1]
				axs[n, 0].imshow(img)  # RGB image, shape (H, W, 3)
				axs[n, 0].set_title("Original RGB")
				axs[n, 0].axis("off")
				
				# Threshold Map (grayscale)
				threshold_img = threshold_map_org[i].numpy()
				axs[n, 1].imshow(threshold_img, cmap='gray')
				axs[n, 1].set_title("Threshold Map")
				axs[n, 1].axis("off")
				
				# Shrink Map (heatmap)
				shrink_img = shrink_map[i].numpy()
				axs[n, 2].imshow(shrink_img, cmap='jet')
				axs[n, 2].set_title("Shrink Map")
				axs[n, 2].axis("off")
				
				n += 1
				if n >= num_images:
					break
			if n >= num_images:
				break
		
	
	# Hide unused subplots if any
	for i in range(n, rows):
		for j in range(cols):
			axs[i, j].axis("off")
	import shutil 
	if save_folder is not None and os.path.exists(save_folder):
		shutil.rmtree(save_folder)
		os.makedirs(save_folder)
		print(save_folder)
	# Save the result
	if save_folder is None:
		output_path = f'0_{name}_three_types.png'
	else:
		os.makedirs(save_folder, exist_ok=True)
		output_path = os.path.join(save_folder, f'0_{name}_three_types.png')
	
	plt.savefig(output_path, bbox_inches="tight")
	plt.close()
	print(f"Three-type visualization saved to: {os.path.abspath(output_path)}")

	
	if save_folder is None:
		return
	
	
	
	for batch_idx, data in enumerate(dataloader):
		img_org = data[0]          # Original RGB image (BxCxHxW)
		threshold_map_org = data[1] # Threshold map (BxHxW)
		shrink_map = data[3]        # Shrink map (BxHxW)
		
		batch_size = img_org.shape[0]
		for i in range(batch_size):
			# Original RGB Image
			img_rgb = denormalize_image(paddle.transpose(img_org[i], perm=[1, 2, 0]),**kwargs).numpy()
			img_bgr = (img_rgb[:,:,::-1]*255).astype(np.uint8)  # Convert to BGR and uint8
			
			# Threshold Map (grayscale)
			threshold_img = threshold_map_org[i].numpy()  # Shape (H, W), range 0-1
			threshold_img_uint8 = (threshold_img * 255).astype(np.uint8)  # Scale to 0-255
			threshold_img_bgr = cv2.cvtColor(threshold_img_uint8, cv2.COLOR_GRAY2BGR)  # To (H, W, 3)
			
			# Shrink Map (heatmap)
			shrink_img = shrink_map[i].numpy()  # Shape (H, W), range 0-1
			shrink_img_uint8 = (shrink_img * 255).astype(np.uint8)  # Scale to 0-255
			shrink_img_colored = cv2.applyColorMap(shrink_img_uint8, cv2.COLORMAP_JET)  # To (H, W, 3) BGR
			
			# Combine the three images horizontally
			combined_img = np.hstack((img_bgr, threshold_img_bgr, shrink_img_colored))
			
			# Save the combined image
			image_name = f'{batch_idx}_{i}.png'
			image_path = os.path.join(save_folder,image_name)
			cv2.imwrite(image_path, combined_img)


def visualize_poly_image(
	image_name: str,
	label_txt_path: str,
	image_dir: str,
	color: tuple = (0, 255, 0),  # Green in BGR
	thickness: int = 2
) -> Optional[np.ndarray]:
	"""
	Visualize polygons on an image based on label data.
	
	Args:
		image_name (str): Name of the image file (e.g., 'image1.jpg')
		label_txt_path (str): Path to the label.txt file
		image_dir (str): Directory containing the images
		color (tuple): BGR color for polygon lines (default: green)
		thickness (int): Thickness of polygon lines (default: 2)
	
	Returns:
		np.ndarray: Image with drawn polygons, or None if there's an error
	"""
	try:
		if image_name  is None:
			import random
			image_name = random.choice(os.listdir(image_dir))
		# Construct full image path
		image_path = os.path.join(image_dir, image_name)
		if not os.path.exists(image_path):
			raise FileNotFoundError(f"Image not found: {image_path}")

		# Read the image
		image = cv2.imread(image_path)
		if image is None:
			raise ValueError(f"Failed to load image: {image_path}")

		# Read label file
		if not os.path.exists(label_txt_path):
			raise FileNotFoundError(f"Label file not found: {label_txt_path}")

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
		
		if polygons is None:
			raise ValueError(f"No label data found for {image_name} in {label_txt_path}")

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

		return image

	except Exception as e:
		print(f"Error visualizing polygons: {str(e)}")
		return None

# Example usage
if __name__ == "__main__":
	# Example parameters
	img_name = None
	label_path = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2/labels.txt"
	img_dir = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs2"

	# Get visualized image
	result_image = visualize_poly_image(img_name, label_path, img_dir)
	
	if result_image is not None:
		
		
		# Optionally save the result
		cv2.imwrite("visualized_image.jpg", result_image)
	
			