from .dbscan_phuoc import DBSCAN
import numpy as np
from shapely import centroid,Polygon
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2

class DBScanPhuoc:
	def __init__(self,**kwargs):
		super(DBScanPhuoc, self).__init__(**kwargs)
		self.eps = 0.045# Adjust this value as needed (in normalized units)
		self.min_samples = 2 # Adjust this value as needed
		self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
	def normalize_centroids(self):
		self.normalized_centroids = np.copy(self.centroids)
		for poly in self.normalized_centroids:
			poly[ 0] = (poly[ 0])/self.w
			poly[ 1] = (poly[ 1]) /self.h
	def caculate_distance_matrix(self):
		n = len(self.normalized_centroids)
		self.distance_matrix = np.zeros((n, n), dtype=float)
		# Use NumPy's broadcasting to compute pairwise Euclidean distances efficiently
		diffs = self.normalized_centroids[:, np.newaxis, :] - self.normalized_centroids[np.newaxis, :, :]
		self.distance_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
		
	def caculate_orientation(self):
		self.vector_angles = []
		for poly in self.np_final_polys:
			x1,y1 = poly[0]
			x2,y2 = poly[1]
			self.vector_angles.append([x2-x1,y2-y1])
   

		
  
	def feetch_data(self,np_final_polys,image_size):
		self.w,self.h = image_size
		self.np_final_polys = np_final_polys
	
		self.final_polys = [Polygon(poly) for poly in self.np_final_polys]
		self.centroids = np.array([[poly.centroid.x, poly.centroid.y]  for poly in self.final_polys])
		
		self.normalize_centroids()
		self.caculate_orientation()
		self.caculate_distance_matrix()
		
		
		self.fit()
		if False:
			self.X_std = self.centroids	
			neighbors = len(self.X_std-1)
			nbrs = NearestNeighbors(n_neighbors=neighbors ).fit(np.copy(self.X_std))
			# Ma trận khoảng cách distances: (N, k)
			distances, indices = nbrs.kneighbors(self.X_std)
			# Lấy ra khoảng cách xa nhất từ phạm vi láng giềng của mỗi điểm và sắp xếp theo thứ tự giảm dần.
			distance_desc = sorted(distances[:, neighbors-1], reverse=True)
			# Vẽ biểu đồ khoảng cách xa nhất ở trên theo thứ tự giảm dần
			plt.figure(figsize=(12, 8))
			plt.plot(list(range(1,len(distance_desc )+1)), distance_desc)
			plt.axhline(y=0.12)
			plt.ylabel('distance')
			plt.xlabel('indice')
			plt.title('Sorting Maximum Distance in k Nearest Neighbor of kNN')
			import random 
			save_folder = "/work/21013187/phuoc/paddle_detect/checkpoints/det_db/tmp"
			import os
			os.makedirs(save_folder,exist_ok=True)
			plt.savefig(f"{save_folder}/knn_distance_plot_{random.uniform(0,10000)}.png", dpi=300, bbox_inches="tight")
			plt.close()
			plt.scatter(self.centroids[:,0], self.centroids[:,1], color='blue', marker='o')
			# Thêm tiêu đề và nhãn trục
			plt.title("2D Scatter Plot")
			plt.xlabel("X values")
			plt.ylabel("Y values")
			plt.savefig(f"{save_folder}/scatter_{random.uniform(0,10000)}.png", dpi=300, bbox_inches="tight")
			plt.close()
			exit()
	
	
	def visualize(self, image):
   

		# Loop through each polygon, centroid, and corresponding orientation vector
		for idx, (centroid, vector) in enumerate(zip(self.normalized_centroids, self.vector_angles)):
			cx, cy = int(centroid[0]*self.w), int(centroid[1]*self.h)  # Centroid coordinates
			vx, vy = vector[0], vector[1]  # Orientation vector components
			
			# Draw the centroid as a red filled circle
			cv2.circle(image, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
			
			# Calculate the endpoint of the orientation arrow (scaled for visibility)
			scale = 100# Adjust this to make arrows longer or shorter
			try:
				end_x = int(cx + vx * scale / np.sqrt(vx**2 + vy**2))  # Normalize vector length
				end_y = int(cy + vy * scale / np.sqrt(vx**2 + vy**2))
			except:
				import pdb;pdb.set_trace()
			
			# Draw the orientation arrow from centroid to endpoint in green
			cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color=(0, 255, 0), thickness=2, tipLength=0.2)

		scaled_centroids = self.centroids
		n = len(scaled_centroids)
		threshold = self.eps # Adjust this threshold as needed (in normalized units)
		for i in range(n):
			for j in range(i + 1, n):  # Avoid duplicate lines (i, j) and (j, i)
				if self.distance_matrix[i, j] < threshold:
					start_point = list(map(int,(scaled_centroids[i][0], scaled_centroids[i][1])))
					end_point = list(map(int,(scaled_centroids[j][0], scaled_centroids[j][1])))
					# Draw line in green
					cv2.line(image, start_point, end_point, color=(0, 255, 0), thickness=2)
					# Calculate midpoint for distance label
					mid_x, mid_y = (start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2
					# Add distance text
					distance_text = f"{self.distance_matrix[i, j]:.2f}"
					cv2.putText(image, distance_text, (mid_x, mid_y), 
								fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, 
								color=(0, 225, 0), thickness=2)
			# Return the modified frame
		return image
	def fit(self):
		dbscan = self.dbscan
		dbscan.fit(self.distance_matrix)
		self.labels = dbscan.labels_
		
		self.core_samples_indices = dbscan.core_sample_indices_
		self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
		
  
	
		# return self.labels, self.core_samples_indices, self.n_clusters_
		