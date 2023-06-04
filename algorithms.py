'''
Description: the algorithms for SAOD
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2022-05-30 07:44:42
LastEditTime: 2023-03-20 19:16:05
'''
from utils import *
import math
import gc
import time

def gaussian_kernel(x1, x2, sigma1, sigma2):
    if sigma1 == 0 or sigma2 == 0 :
        return 1
    else:
        try:
            eu_dist = sqrt(np.sum((x1 -x2)**2))
            res = exp(-eu_dist / (sigma1 * sigma2)) 
        except:
            print("error: ", x1,x2)
            print(np.sum((x1 -x2)))
    return res
 
def get_similarity(point1, point2):
    eu_dist = sqrt(np.sum((point1 -point2)**2))
    sim = exp(-eu_dist / (2 * 2)) 
    return sim

def cal_sim_matrix(X, result_dist, result_index,point_i ):
    X = np.array(X)
    d = len(X[0])
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            try:
                sum_d = 0
                for k in range(d):
                    dif = (X[i][k]- X[j][k])**2
                    sum_d += dif
                sim = exp(-sqrt(sum_d) / (2 * 2))
            except:
                sim = 1
                print("error: ",X[i], X[j])
            S[i][j] = sim
            S[j][i] = S[i][j]
    return S

def cal_Laplacian_matrix(sim_matrix):
    diag_matrix = np.sum(sim_matrix, axis = 0)
    np.seterr(divide='ignore', invalid='ignore')
    sqrt_diag_matrix = np.diag( (1.0 / (diag_matrix ** (0.5))))
    laplacian_matrix = np.diag(diag_matrix) - sim_matrix
    # D^(-1/2) L D^(-1/2)
    # return np.dot(np.dot(sqrt_diag_matrix, laplacian_matrix), sqrt_diag_matrix)
    return laplacian_matrix

def cal_eigvals(point_set, result_dist, result_index,i,k_threshold):
	lapla_ev = []
	sim_matrix = cal_sim_matrix(point_set, result_dist, result_index,i )
	Laplacian = cal_Laplacian_matrix(sim_matrix)
	try:
		x_lapla = np.linalg.eigvals(Laplacian)
	except:
		lapla_ev = [0]* k_threshold
		print("exception raise")
		return lapla_ev
	else:
		for ev in x_lapla:
			if isinstance(ev, complex):
				ev = np.real(ev)
			lapla_ev.append(ev)
		lapla_ev.sort()
	return lapla_ev

def get_kdim_kde(x,data_array,bandwidth=2):
    def gauss(x, d):
        return (1/pow( math.sqrt(2*math.pi ), d))*math.exp(-0.5*(linalg.norm(x)))
    N=len(data_array)
    d = len(x)
    res=0
    if len(data_array)==0:
        return 0
    for i in range(len(data_array)):
        gauss_value = gauss((linalg.norm(x-data_array[i]))/bandwidth, d)
        res += gauss_value
    res /= (N*bandwidth)
    return res

def saod(values, d_dim, result_index, result_dist):
    global_of = [] 
    of = []
    nn_number = len(result_index)
    for i,v in enumerate(values):
        bw = 2
        temp_kde = get_kdim_kde(v[-d_dim:len(v)], values[:,-d_dim:len(v)], bandwidth=bw)  
        global_of.append(temp_kde)
    for i in range(len(values)):
        prob = 0
        for nn in result_index[i]:
            prob += global_of[nn]
        outlier_f = (prob/nn_number) /global_of[i]
        of.append(outlier_f)  
    return of 

def our_method(point_set, k_threshold, d_dim=3):
    method='saod'  
    start = time.time()
    kdt = KDTree(point_set)
    result_dist,result_index = kdt.query(point_set, k_threshold)
    dist_ev_arr = []; lapla_ev_arr = []
    start = time.time()
    for i in range(len(point_set)) :
        temp_ps = []
        for p in result_index[i] : 
            temp_ps.append(point_set[p])
        lapla_ev = cal_eigvals(temp_ps, result_dist,result_index,i, k_threshold)
        lapla_ev_arr.append(lapla_ev)
    lapla_ev_arr = np.array(lapla_ev_arr)
    start = time.time()
    outlier_factor = saod(lapla_ev_arr, d_dim, result_index, result_dist)
    return outlier_factor 

