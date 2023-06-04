'''
Description:  对比算法 KDEOS
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2022-04-14 09:51:38
LastEditTime: 2022-07-20 14:43:05
'''
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KDTree, BallTree
import math

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return np.array(point_set) 

def sok(point_set, kmin, kmax, d, result_dist, result_index):
	ps_size = len(point_set)
	sok_value =  [[0 for i in range(kmax)] for j in range(ps_size)]
	for i, ps in enumerate(point_set):
		knn = result_index[i]
		for k in range(kmin, kmax):
			# h = compute kernel bandwidth from Nmax[1; k]
			# h = result_dist[i][k-1]		# bandwidth
			h = 0.01
			for j, q in enumerate(knn):
				temp_p = result_dist[i][j]
				if sok_value[i][k-1]!= 0 :
					sok_value[i][k-1] += kuh(temp_p, h, d)
				else:
					sok_value[i][k-1] = kuh(temp_p, h, d)
	return sok_value

def kuh(u, h, d):
	kuh_value = 1 - (u*u)/(h*h)
	kuh_value = 3*kuh_value/(4*pow(h,d))
	return kuh_value

def norm_scores(p, fi):
	return fi*(1-p)/(fi+p)

def cd(idx, point_set, sok, kmin, kmax, result_index):
	kde = 0
	sok_list = []
	for k in range(kmin, kmax):
		for q in result_index[idx]:
			sok_list.append(sok[q][k-1])
		mu = np.average(sok_list)
		sigma = np.std(sok_list)
		kde += (mu-sok[idx][k-1])/sigma
	kde = kde/(kmax-kmin+1)
	return kde

def KDEOS(point_set, kmin, kmax):
	dim = len(point_set[0])
	kde_arr = []
	# The point_set should be numpy array
	my_cover = BallTree(point_set, leaf_size=2, metric = 'euclidean' )
	# my_cover = KDTree(point_set, leaf_size=1, metric = 'euclidean' )
	result_dist, result_index = my_cover.query(point_set, kmax) 
	# Compare densities:
	sok_value = sok(point_set, kmin, kmax, dim, result_dist, result_index)
	for idx, p in enumerate(point_set):
		kde = cd(idx, point_set, sok_value, kmin, kmax, result_index)
		kde = norm_scores(1-norm.cdf(kde), 0.001)		 # fi=0.001
		if math.isnan(kde):
			kde = 0
		kde_arr.append(kde)
	return kde_arr     

if __name__ == '__main__':
	fileName = "../2d/data_caiming.dat"
	point_set = loadData(fileName, float, ",")
	scores = KDEOS(point_set, kmin=3, kmax=7)
	print(scores)
