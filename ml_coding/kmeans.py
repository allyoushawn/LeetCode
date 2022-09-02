# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

"""
Keep in mind
1. Using np.random.choice instead of random.choice
2. get_distance  np.sum((x1 - x2)**2)
3. new_cetroids = np.mean(sub_X, axis=0)) # Make the axis 0 disappeared
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# create data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)



def compute_dist(arr1, arr2):
	return np.sum((arr1 - arr2)**2)

def K_means(X, k):
	"""
	randomly select k initial centroids
	perform cluster-compute centroids iterations
	"""
	selected_index = np.random.choice(range(len(X)), k)
	cluster_idx_arr = np.zeros(len(X))
	centroids = X[selected_index]
	iter_num = 1000

	

	for iter_idx in range(iter_num):

		# clustering with centroids
		for idx in range(len(X)):
			x = X[idx]
			cur_min_dist = float("inf")
			assigned_cluster_idx = None
			for cluster_idx in range(k):
				centroid = centroids[cluster_idx]
				dist = compute_dist(x, centroid)
				if dist < cur_min_dist:
					assigned_cluster_idx = cluster_idx
					cur_min_dist = dist
			cluster_idx_arr[idx] = assigned_cluster_idx

		new_centroids_list = []
		
		# compute new centroid
		for cluster_idx in range(k):
			selected_idx_list = []
			for idx in range(len(X)):
				if cluster_idx_arr[idx] == cluster_idx:
					selected_idx_list.append(idx)
			selected_x = X[selected_idx_list]
			new_centroids_list.append(np.mean(selected_x, axis=0))
		new_centroids_arr = np.array(new_centroids_list)


		if np.sum(new_centroids_arr - centroids) == 0:
			break
		else:
			centroids = new_centroids_arr
	return cluster_idx_arr
y_pred = K_means(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()