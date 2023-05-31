#!/usr/bin/env python
#-*- coding:utf-8 -*- 
'''
Description: io and plot
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2022-03-28 16:45:50
LastEditTime: 2023-03-26 14:50:40
'''
from scipy.io import loadmat
from scipy.io.arff import loadarff
# import seaborn as sns
import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# from KDEpy import FFTKDE, NaiveKDE
from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
import os
from itertools import islice, cycle
import matplotlib.pyplot as plt
from numpy import linalg , array
from sklearn.datasets import make_blobs
# from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from itertools import starmap, islice, cycle
import copy
import numpy as np
from sklearn.neighbors import KDTree, BallTree
from math import sqrt, exp, log, ceil,e, pi
from sklearn.metrics import roc_curve, auc
import arff
from os import walk, listdir
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,euclidean, cityblock, chebyshev, jaccard
from scipy import stats
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import *
import cv2
import pprint
from algorithms import * # cal_eigvals
# from od_compare import *
from scipy.spatial import KDTree
import math
from scipy import signal
import mat73
from sklearn.manifold import TSNE

def plt_high_dimension_data(filename, X, label):
	print(X.dtype)
	fig = plt.figure(0)
	ax = fig.add_subplot(111, projection='3d')  # New in version 1.0.0
	X_embedded = TSNE(n_components=2, init='pca').fit_transform(X)
	x = array(X_embedded)[:, 0]
	y = array(X_embedded)[:, 1]
	# z = array(X_embedded)[:, 2]
	plt.scatter(x,y, c=label, s=10)
	# ax.scatter(x,y,z, c=label, s=10)
	plt.savefig("../result/%s_tsne_2.pdf"%(filename.split("/")[-1].split(".")[0]))
	plt.close(0)
	return X_embedded

def load_http(fileName):
	m = mat73.loadmat(fileName)
	# print(data_dict)
	point_set = m["X"]; labels = m["y"].tolist()
	outlier_num = labels.count([1])
	return point_set, labels, outlier_num

def dist(point1,point2):
    sum_dis = 0.0
    dimension = len(point1)
    for index in range(dimension)  :
        sum_dis += (point2[index] - point1[index])**2
    return sqrt(sum_dis)

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return np.array(point_set) 

def load_mat(fileName):
    m = loadmat(fileName)
    point_set = m["X"]; labels = m["y"].tolist()
    outlier_num = labels.count([1])
    return point_set, labels, outlier_num

def load_arff1(fileName):
    with open(fileName) as fh:
        dataset = np.array(arff.load(fh)['data'])
        point_set = dataset[:,1:-1].astype(np.float)
        labels = dataset[:,0]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no' :
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(np.int), outlier_num

def plt_outliers(outliers, point_set,fileName, clf):
    plt.figure(0)
    plt.xticks([])
    plt.yticks([])
    for i, point in enumerate(point_set) :
        # plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=10)
        if i in outliers :
            plt.scatter(point[0],point[1],color= 'r', marker = '*',s=60)
        else:   
            plt.scatter(point[0],point[1],color= 'b', marker = 'o')
    plt.savefig('../result/%s_%s.png'%(fileName.split('/')[2].split('.')[0], clf))
    plt.close(0)

def plt_eigenvalue(scores, point_set, k):
	plt.figure(0)
	axisx = []; axisy = []
	for i, val in enumerate(scores):
		# plt.plot(i,val)
		# plt.annotate(i)
		# plt.annotate(i, xy = (i, val),xycoords = 'data',fontsize=6)
		axisx.append(i)
		axisy.append(val)
	plt.plot(axisx, axisy)
	plt.scatter(axisx, axisy, marker='o')
	plt.title("gini eigenvalue(k=%d)"%(k))
	plt.savefig('../result/%s_k=%d.jpg'%(fileName.split('/')[2].split('.')[0], k))
	plt.close(0)

def plt_point(point_set,fileName):
    plt.figure(0)
    for i, point in enumerate(point_set) :
        plt.scatter(point[0],point[1],s=10)
        plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=5)
    plt.savefig('../result/data1_point.pdf', dpi=400)
    plt.close(0)

def plt_density_colorbar(point_set, density_arr, k_threshold, filename):
	plt.figure(0)
	plt.rcParams['font.size']=16
	x = array(point_set)[:, 0]
	y = array(point_set)[:, 1]
	t = array(density_arr)
	# plt.xlabel('x') # point 
	# plt.ylabel('y') # point 
	plt.title("k=%d"%(k_threshold))
	# ps_size = 100 / len(point_set) 
	ps_size = 15+ 1000*t #  10# 
	# for idx, point in enumerate(point_set):
	#     plt.annotate(idx, xy = (point[0], point[1]),xycoords = 'data',fontsize=10)
	plt.scatter(x, y, c=t, cmap='cool',s=ps_size) # , marker='+' tab20  # viridis  bwr
	plt.colorbar() 
	plt.savefig("../result/%s_k=%d.png"%(filename.split("/")[-1].split(".")[0], k_threshold), dpi=500)
	plt.close(0) 

def print_sim_matrix(point_set, result_dist, result_index):
    for to_plt_point in [53, 95]: # 
        knn_set = []
        # plt.figure(0)
        for knn_idx in result_index[to_plt_point] :
            knn_set.append(point_set[knn_idx])
        sim_matrix = cal_sim_matrix(knn_set, result_dist, result_index, to_plt_point)
        print("sim_matrix_%d :\n"%(to_plt_point), sim_matrix)
        x = np.linalg.eigvals(sim_matrix)
        x.sort()
        print("eigenvalues_%d :\n"%(to_plt_point), x)
        diag_matrix = np.sum(sim_matrix, axis = 0)
        sqrt_diag_matrix = np.diag( (1.0 / (diag_matrix ** (0.5))))
        print("diag_matrix_%d :\n"%(to_plt_point), sqrt_diag_matrix)
        laplacian_matrix = np.diag(diag_matrix) - sim_matrix
        print("laplacian_matrix_%d :\n"%(to_plt_point), laplacian_matrix)
        x = np.linalg.eigvals(laplacian_matrix)
        x.sort()
        print("eigenvalues_%d :\n"%(to_plt_point), x)

def plt_entropy_change(indices, entropy_arr):
    plt.figure(0)
    x = np.arange(len(entropy_arr))
    len_ind = np.arange(len(indices))
    plt.xlabel('delete index')
    plt.ylabel('entropy')
    plt.title("data28")
    plt.plot(x, entropy_arr)
    print("indices: ", indices)
    plt.xticks(len_ind, indices[::-1], rotation=90, size=5 )
    # for i, point in enumerate(point_set) :
    #     plt.scatter(point[0],point[1],s=10)
    #     plt.annotate(all_scores[i], xy = (point[0], point[1]),xycoords = 'data',fontsize=5)
    plt.savefig('../result/data_syn2_1_entropy.png', dpi = 200)
    plt.close(0)

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return np.array(point_set) 

# case2: -2: id  -1: outlier 
def load_arff2(fileName):
	with open(fileName) as fh:
		dataset = np.array(arff.load(fh)['data'])
		point_set = dataset[:,:-2].astype(np.float64)
		labels = dataset[:,-1]
		outlier_num = 0
		for i, l in enumerate(labels):
			if l == 'no' :
				labels[i] = 0
			else:
				labels[i] = 1
				outlier_num += 1
	return point_set, labels.astype(np.int32), outlier_num

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return np.array(point_set) 

def plt_clusters(point_set, clusters, fileName):
    colors = array(list(islice(cycle(['y','r','g','b','purple','m','orange','deepskyblue','c','lime']),int(max(clusters) + 1))))
    plt.scatter(array(point_set)[:, 0],array(point_set)[:, 1], s=10, color=colors[clusters])
    plt.title("")
    plt.xticks([])
    plt.yticks([])
    # plt.xlim([])
    # plt.ylim([])
    plt.savefig('../result/%s_clusters.png'%(fileName.split('/')[2].split('.')[0]), dpi=500)

def plt_scores(point_set, all_scores):
    plt.figure(0)
    for i, point in enumerate(point_set) :
        plt.scatter(point[0],point[1],s=10)
        plt.annotate(all_scores[i], xy = (point[0], point[1]),xycoords = 'data',fontsize=5)
    plt.savefig('gini.png', dpi = 200)
    plt.close(0)

def plt_point(point_set,fileName):
    plt.figure(0)
    for i, point in enumerate(point_set) :
        plt.scatter(point[0],point[1],s=20)
        plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=10)
    plt.savefig('../result/%s_point.png'%(fileName), dpi=200)
    plt.close(0)

def plt_all_2d_data():
	path = "../2d/"
	for root, dir, files in os.walk(path):
		for filename in files:
			point_set = loadData(path+filename, float, ",")
			plt_point(point_set, filename)                 

def plt_eigval(point_set, all_scores, split, k_threshod):
	plt.figure(0)
	plt.title('sum_eigenvalue k=%d.png'%( k_threshod))
	# plt.title('%s_eigenvalue k=%d.png'%(split, k_threshod))
	for i, point in enumerate(point_set):
		plt.scatter(point[0],point[1],s=10)
		plt.annotate(all_scores[i], xy = (point[0], point[1]),xycoords = 'data',fontsize=5)
	plt.savefig('../test_eigenvalues/data20_%s_k=%d.png'%(split, k_threshod), dpi = 200)
	plt.close(0)

def plt_lapla_ev(point_set, k_threshod):
	plt.figure(0)
	plt.title('k=%d'%( k_threshod))
	# plt.title('%s_eigenvalue k=%d.png'%(split, k_threshod))
	for i, point in enumerate(point_set):
		print(i, point)
		plt.scatter(point[0],point[-1],s=10)
		plt.annotate(i, xy = (point[0], point[-1]),xycoords = 'data',fontsize=5)
	plt.savefig('../data20_k=%d.png'%(k_threshod), dpi = 200)
	plt.close(0)

def plt_sim_matrix(indexes, similarity):
	plt.figure(0)
	y_axis = indexes
	x_axis = indexes
	plt.rcParams['font.size']=16

	fig, ax = plt.subplots()
	im = ax.imshow(similarity, cmap='cool')

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(indexes)))
	ax.set_yticks(np.arange(len(indexes)))

	# ... and label them with the respective list entries
	ax.set_xticklabels(indexes)
	ax.set_yticklabels(indexes)

	#Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
				rotation_mode="anchor")

	##Loop over data dimensions and create text annotations.
	for i in range(len(indexes)):
		for j in range(len(indexes)):
			text = ax.text(j, i, round(similarity[i, j], 4), 
							ha="center", va="center", color="black")

	# ax.set_title("Laplacian matrix")  # similarity diagnal  Laplacian
	fig.tight_layout()
	plt.savefig("laplacian%d.png"%(indexes[0]), dpi=500,bbox_inches='tight')# laplacian
	plt.close(0)

def read_auc(fileName):
	auc_arr = []
	for line in open(fileName, 'r'): 
		vaule = [data for data in line.split(',')]
		vaule = float(vaule[-1])
		auc_arr.append(vaule)
	return auc_arr

def save_all_auc():
	i = 0
	result_path = "E:\\project\\my results\\"
	methods = os.listdir(result_path)
	# print()
	datasets = [ 'Arrhythmia', 'Cardiotocography', 'HeartDisease', 'Hepatitis', 'InternetAds' ,'PageBlocks', 'Parkinson', 'PenDigits', 'Pima', 'Shuttle', 'SpamBase', 'Stamps', 'Waveform', 'WBC', 'WDBC,' 'Wilt', 'WPBC']
	to_plt_data = dict()
	# to_plt_data = []
	to_plt_idx = 0
	for method in methods:
		for root,dirs,files in walk(result_path+method): 
			for name in files:
				file_name, file_type = os.path.splitext(name)
				data_name = file_name.split("_")[1]
				# print(data_name, datasets[0])
				# input()
				if data_name == datasets[to_plt_idx]:
					new_path = root+'\\'+ name
					auc_value = read_auc(new_path)
					print(method, "==", len(auc_value))
					to_plt_data[method] = auc_value
					# to_plt_data.append(auc_value) 
	print(to_plt_data)
	df = pd.DataFrame(to_plt_data)
	# df.boxplot(grid=False, rot=45, fontsize=15)
	
	df.plot.box(title=datasets[to_plt_idx])
	# plt.boxplot(to_plt_data, labels=methods) 
	# plt.show()				
	plt.savefig('../result/%s.png'%(datasets[to_plt_idx]), dpi = 200)
	# plt.title(method)
	return 0

def plt_1data_box():
	i = 0
	result_path = "E:\\project\\my results\\"
	# LOF 2000 CBLOF 2003 KDE 2007 SOD 2009 KDEOS 2014 ALAD 2018 MOGAAL 2019 COPOD 2020  ECOD 2022 
	methods = ['LOF','CBLOF','KDE','SOD','KDEOS','ALAD','MOGL', 'COPOD','ECOD','SAOD']
	# methods = ['KNN','LOF','CFAR','RDOS','KDEOS','SUOD','ECOD','COPOD','SAOD']
	datasets = [ 'Arrhythmia', 'Cardiotocography', 'HeartDisease', 'Hepatitis', 'InternetAds' ,'PageBlocks', 'Parkinson', 'PenDigits', 'Pima', 'Shuttle', 'SpamBase', 'Stamps', 'Waveform', 'WBC', 'WDBC', 'Wilt', 'WPBC']
	# 0'Arrhythmia', 1'Cardiotocography', 2'HeartDisease', 3'Hepatitis', 4'InternetAds' ,5'PageBlocks', 6'Parkinson', 7'PenDigits', 8'Pima', 9'Shuttle', 10'SpamBase', 11'Stamps', 12'Waveform', 13'WBC', 14'WDBC,' 15'Wilt', 16'WPBC'
	to_plt_data = dict()
	# to_plt_data = []
	to_plt_idx = 0
	for to_plt_idx in [6, ] : # range(0,1)  0,2,3,4,6,7,8,9,11,12,13,14,15,16
		for method in methods:
			for root,dirs,files in walk(result_path+method): 
				for name in files:
					file_name, file_type = os.path.splitext(name)
					data_name = file_name.split("_")[1]
					# print(data_name, datasets[0])
					# input()
					if data_name == datasets[to_plt_idx]:
						new_path = root+'\\'+ name
						# print(new_path)
						auc_value = read_auc(new_path)
						# print(method, "==", auc_value)
						print(data_name,method, "==", len(auc_value))
						to_plt_data[method] = auc_value
						# to_plt_data.append(auc_value) 
		# print(to_plt_data)
		df = pd.DataFrame(to_plt_data)
		plt.rcParams['font.size']=16
		# df.boxplot(grid=False, rot=45, fontsize=15)
		df.plot.box(title=datasets[to_plt_idx], rot=45, fontsize=16)
		plt.ylim(0,1)
		# plt.boxplot(to_plt_data, labels=methods) # 所有方法在 Arrhythmia 上的结果 
		# plt.show()				
		plt.savefig('../result/%s.png'%(datasets[to_plt_idx]), dpi = 500,bbox_inches='tight',)
		# plt.title(method)
	return 0

def roc_statistics(file_path):
	auc_value = read_auc(file_path)
	min_value = min(auc_value)
	max_value = max(auc_value)
	avg_value = round(np.mean(auc_value), 4)
	return min_value, max_value, avg_value

def min_max_avg_results():
	methods = ['CBLOF','LOF','KDE','SOD','KDEOS','ECOD','COPOD','SAOD'] # 'ALAD'
	# methods = ['KNN','LOF','CFAR','RDOS','KDEOS','SUOD','ECOD','COPOD','SAOD']
	result_path = "E:\\project\\my results\\ALAD\\"
	for root,dirs,files in walk(result_path):
		for every_file in files:
			# print("now process file: ",every_file)
			file_name = os.path.splitext(every_file)[0]
			min_value, max_value, avg_value = roc_statistics(root+every_file)
			print(file_name, min_value, max_value, avg_value)
	return 0

def combine_figure():
	img1 = cv2.imread("../result/12_point_ev.png") 
	img2 = cv2.imread("../result/data_syn1_1.dat_point.png") 
	# fig = plt.figure()       
	fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(20, 3))
	# frame = plt.gca()
	# frame.axes.get_yaxis().set_visible(False)
	# frame.axes.get_xaxis().set_visible(False)
	plt.axis('off')
	plt.xticks([])
	plt.yticks([])
	ax1.imshow(img1)              
	ax2.quiver(0, 0, 4, 4, color='deepskyblue', width=2, scale=3)
	ax3.imshow(img2) 
	# plt.subplot(1, 4, 1)
	# plt.imshow(img1) 
	# plt.axis('off')
	# plt.xticks([])
	# plt.yticks([])
	# plt.subplot(1, 4, 2)
	# plt.quiver(-50, 0, 13, 6, color='deepskyblue', width=0.05, scale=30)
	# plt.axis('off')
	# plt.xticks([])
	# plt.yticks([])
	# # plt.imshow(img1) 
	# plt.subplot(1, 4, 3)
	# plt.imshow(img1) 
	# plt.axis('off')
	# plt.xticks([])
	# plt.yticks([])
	plt.savefig('../result/combine.png', dpi = 200)
	# plt.show()
	return 0

def scores2labels(scores, outlier_num):
	labels = [0] * len(scores)
	# sorted_scores = sorted(scores, reverse=True)
	scores_arr = np.array(scores)
	outliers = np.argpartition(scores_arr, outlier_num)
	# print (sorted_scores)
	for i in outliers[:outlier_num]:
		labels[i] = 1
	return labels

def scores2outliers(scores, outlier_num):
	scores_arr = np.array(scores)
	outliers = np.argpartition(scores_arr, outlier_num)
	# sorted_scores = sorted(scores, reverse=True)
	# outliers = scores_arr.argmin(numberofvalues=outlier_num) 
	return outliers[:outlier_num]

def median_roc():
	result_path = "E:\\project\\my results\\"
	methods = ['CBLOF','LOF','KDE','SOD','KDEOS','ALAD',"MOGL", 'ECOD','COPOD','SAOD']
	# methods = ['KNN','LOF','CFAR','RDOS','KDEOS','SUOD','ECOD','COPOD','SAOD']
	datasets = [ 'Arrhythmia', 'Cardiotocography', 'HeartDisease', 'Hepatitis', 'InternetAds' ,'PageBlocks', 'Parkinson', 'PenDigits', 'Pima', 'Shuttle', 'SpamBase', 'Stamps', 'Waveform', 'WBC', 'WDBC', 'Wilt', 'WPBC']
	# 0'Arrhythmia', 1'Cardiotocography', 2'HeartDisease', 3'Hepatitis', 4'InternetAds' ,5'PageBlocks', 6'Parkinson', 7'PenDigits', 8'Pima', 9'Shuttle', 10'SpamBase', 11'Stamps', 12'Waveform', 13'WBC', 14'WDBC,' 15'Wilt', 16'WPBC'
	roc_dict = dict()
	# to_plt_idx = 16
	for to_plt_idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] : # range(0,16) 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
		for method in methods:
			for root,dirs,files in walk(result_path+method): 
				for name in files:
					file_name, file_type = os.path.splitext(name)
					data_name = file_name.split("_")[1]
					# print(data_name, datasets[0])
					# input()
					if data_name == datasets[to_plt_idx]:
						new_path = root+'\\'+ name
						# print(new_path)
						auc_value = read_auc(new_path)
						# print(method, "==", auc_value)
						# print(method, "==", len(auc_value))
						max_value = round(max(auc_value), 4) 
						mean = round(np.mean(auc_value), 4) 
						min_value = round(min(auc_value), 4) 
						std_value = round(np.std(auc_value), 4) 
						roc_dict[method] = std_value   # max # mean  # 
		sorted_dict = sorted(roc_dict.items(), key = lambda kv:(kv[1], kv[0]))
		print(datasets[to_plt_idx], sorted_dict)
	return sorted_dict

def std_roc():
	result_path = "E:\\project\\my results\\"
	methods = ['CBLOF','LOF','KDE','SOD','KDEOS','ALAD',"MOGL", 'ECOD','COPOD','SAOD']
	# methods = ['KNN','LOF','CFAR','RDOS','KDEOS','SUOD','ECOD','COPOD','SAOD']
	datasets = [ 'Arrhythmia', 'Cardiotocography', 'HeartDisease', 'Hepatitis', 'InternetAds' ,'PageBlocks', 'Parkinson', 'PenDigits', 'Pima', 'Shuttle', 'SpamBase', 'Stamps', 'Waveform', 'WBC', 'WDBC', 'Wilt', 'WPBC']
	# 0'Arrhythmia', 1'Cardiotocography', 2'HeartDisease', 3'Hepatitis', 4'InternetAds' ,5'PageBlocks', 6'Parkinson', 7'PenDigits', 8'Pima', 9'Shuttle', 10'SpamBase', 11'Stamps', 12'Waveform', 13'WBC', 14'WDBC,' 15'Wilt', 16'WPBC'
	roc_dict = dict()
	# to_plt_idx = 16
	# for method in methods:
	method = 'SAOD'
	for to_plt_idx in [0,2,3,4,6,7,8,9,11,12,14,16] : # range(0,16) 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
		for root,dirs,files in walk(result_path+method): 
			for name in files:
				file_name, file_type = os.path.splitext(name)
				data_name = file_name.split("_")[1]
				# print(data_name, datasets[0])
				# input()
				if data_name == datasets[to_plt_idx]:
					new_path = root+'\\'+ name
					# print(new_path)
					auc_value = read_auc(new_path)
					# print(method, "==", auc_value)
					# print(method, "==", len(auc_value))
					# max_value = round(max(auc_value), 4) 
					# mean = round(np.mean(auc_value), 4) 
					# min_value = round(min(auc_value), 4) 
					std_value = round(np.std(auc_value), 4) 
					print(std_value)
					roc_dict[method] = std_value   # max # mean  #
	# print(data_name,method, roc_dict[method]) 
	sorted_dict = sorted(roc_dict.items(), key = lambda kv:(kv[1], kv[0]))
	# print(datasets[to_plt_idx], sorted_dict)
	return sorted_dict

def my_methd_visualize():
	# fileName = "E:\\project\\ODdata\\2d\\2d-comma\\data_l2.dat" 
	fileName = "../2d/data_syn1_2.dat"  
	point_set = loadData(fileName, float, ',') # 20point

	plt.figure(0)
	fig, ax = plt.subplots() 
	for i, point in enumerate(point_set) :
		ax.scatter(point[0],point[1],s=20)
		ax.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=6)
	plt.savefig('../result/syn1_point.png', dpi=1000)

	kdt = KDTree(point_set)
	k_threshold = 6
	result_dist,result_index = kdt.query(point_set, k_threshold)
	# plt_point_knn(point_set, result_dist,result_index)
	dist_ev_arr = []; lapla_ev_arr = []
	fig = plt.figure()
	ax = Axes3D(fig)
	for i, point in enumerate(point_set) :
		# if i in [68, 58, 48] :
		temp_ps = []
		for p in result_index[i] : 
			temp_ps.append(point_set[p])
		dist_ev, lapla_ev = cal_eigvals(temp_ps, result_dist,result_index,i, k_threshold)
		for k_idx in range(1, len(result_index[i])):
			x = [point[0], point_set[result_index[i][k_idx]][0]]
			y = [point[1], point_set[result_index[i][k_idx]][1]]
			# print("plotting: ",i, result_index[i][k_idx])
			ax.plot(x,y)
		circle1 = plt.Circle((point[0],point[1]), result_dist[i][-1], color = 'b',fill=False)
		ax.add_patch(circle1)
	# plt.savefig('../result/syn1_point_knn.eps', dpi=1000)
		# all_values.extend(eigVals)
		# print(i,"===eigVals: ", lapla_ev)
		# input()
	# 	# dist_ev_arr.append(dist_ev)
	# 	lapla_ev_arr.append(lapla_ev)
	# 		# print(dist_ev_arr)
	# 		# plt_sim_matrix(result_index[i], dist_ev_arr)
	# 		# plt_3d(dist_ev_arr[:,1:])
		
	# 	# x = lapla_ev_arr[i][0]
	# 	# y = lapla_ev_arr[i][1]
	# 	# z = lapla_ev_arr[i][2]
	# 	x = lapla_ev_arr[i][3]
	# 	y = lapla_ev_arr[i][4]
	# 	z = lapla_ev_arr[i][5]
	# 	# x = dist_ev_arr[i][0]
	# 	# y = dist_ev_arr[i][1]
	# 	# z = dist_ev_arr[i][2]
	# 	# x = dist_ev_arr[i][0]
	# 	# y = dist_ev_arr[i][1]
	# 	# z = dist_ev_arr[i][2]

	# 	ax.scatter(x,y,z)
	# 	# plt.savefig('the figure named {}'.format(str), format='png', dpi=300, pad_inches=0)
	# 	ax.text(x,y,z,i)
	# 	# plt.scatter(dist_ev_arr[i][-1],lapla_ev_arr[i][1],s=20) # , c=color_arr[cls_idx]
	# 	# plt.annotate(i, xy = ([x,y,z]),xycoords = 'data',fontsize=10)
	# # plt.show()
	# plt.savefig('../result/syn1_lat_ev2.png', dpi=100)
	# print(dist_ev_arr)
	# print(lapla_ev_arr)
	return 0

def sensitivity_reason():
	fileName = '../sensitivity/InternetAds_norm_02_v10.arff' # InternetAds_norm_02_v10
	point_set, labels, outlier_num = load_arff2(fileName)
	# point_set, labels, outlier_num = load_mat("../mat/wbc.mat")
	point_set = np.array(point_set.tolist()).astype(np.float)
	plt.rcParams['font.size']=16
	labels = np.array(labels).astype(int)
	# print(labels)
	X_embedded = TSNE(n_components=2, init='random').fit_transform(point_set)
	x = np.array(X_embedded)[:, 0]
	y = np.array(X_embedded)[:, 1]
	for i in range(len(x)) :
		if labels[i] == 1:
			p1 = plt.scatter(x[i],y[i],s=10,c='r', marker='*')
		else:
			p2 = plt.scatter(x[i],y[i],s=10,c='b', marker='o')
	plt.legend([p1, p2], ('outliers', 'inliers'))
	plt.title("InternetAds")  # InternetAds
	# plt.title("Wbc")  # InternetAds
	plt.xticks([])
	plt.yticks([])
	plt.savefig("../result/InternetAds_2d.png", dpi=500)
	# plt.savefig("../result/wbc_2d.png", dpi=500)
	return 0

def main():
	sensitivity_reason()
	# result_path = "E:\\project\\my results\\MOGL\\"
	# for root,_,files in os.walk(result_path):
	# 	# method = file.split('_')[0]
	# 	for file_name in files:
	# 		new_name = file_name.replace("MOGAAL", "MOGL")
	# 		os.rename(os.path.join(root,file_name),os.path.join(root,new_name))
	# plt_1data_box()
	# min_max_avg_results()
	# file_name = "E:\\project\\my results\\KNN\\KNN_Annthyroid_auc.csv"
	# min_valusensitivity_reasone, max_value, avg_value = roc_statistics(file_name)
	# print(file_name, min_value, max_value, avg_value)
	# median_roc()
	# std_roc()
	# fileName = "E:\\project\\my results\\MOGL\\MOGL_InternetAds_auc.csv"
	# f1 = open('../result/MO_GAAL_InternetAds_auc_2.csv','a')
	# for line in open(fileName, 'r'): 
	# 	all_vaule = [data for data in line.split(',')]
	# 	vaule = float(all_vaule[-1].split("\"")[0])
	# 	f1.write( all_vaule[0] + ','  + str(vaule) + '\n')
		# print("MO_GAAL: ",every_file, roc_auc)
		# print(vaule)
		# input()
		# auc_arr.append(vaule)
	# my_methd_visualize()

if __name__ == "__main__" :
	main()