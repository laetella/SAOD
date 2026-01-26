# -*- coding: utf-8 -*-
'''
Description: 
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2022-05-30 07:44:42
LastEditTime: 2023-03-24 16:29:36
'''
from SAOD import *
from kdeos import *
import time
from od_compare import OutlierExperiment

def two_dim_test(fileName, k_threshold, d_dim):
    # fileName = "../2d/4.dat" 
    point_set = loadData(fileName, float, ",")
    # plt_point(point_set,"data_syn1_2")
    outlier_factor = SAOD(point_set, k_threshold, d_dim)
    print( outlier_factor)
    return 0

def two_dim_test_all():
    data_path = "../2d_data/"
    for root,dirs,files in walk(data_path):
        for every_file in files:
            for k_threshold in (3,): # 4,5,6,7,8,
                d_dim = k_threshold
                point_set = loadData(root+every_file, float, ",")
                outlier_factor = SAOD(point_set, k_threshold, d_dim)
                indices = np.argsort(outlier_factor)
                print( outlier_factor)
    return 0

def low_dim():
    best_auc = 0; best_k = 3
    for k_threshold in (3,4,5,6,7,8,9,10,15,20,25,30,35, 40, 45): # 6,7,8,9,10,15,20,25,30,35 
        fileName = 'E:\\project\\ODdata\\mat\\pyOD_benchmark\\optdigits.mat'
        print(k_threshold, fileName)
        point_set, labels, outlier_num = load_mat(fileName)
        scores = SAOD(point_set, k_threshold, k_threshold)
        # print(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = round(auc(fpr,tpr), 4) 
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_k = k_threshold
        print(k_threshold, fileName,roc_auc)

def dami():
    data_path = 'E:\\project\\ODdata\\arff2\\Pima\\'  # \\pyOD_benchmark  odds_mat  Pima\\
    f1 = open('../result/SAOD_mat_d1_auc.csv','w')
    f2 = open('../result/SAOD_mat_d1_best_auc.csv','w')
    # k_threshold = 6
    for root,dirs,files in walk(data_path):
        for every_file in files:
            best_auc = 0; best_k = 3
            for k_threshold in (6,7,8,9,10,15,20,25,30,35, 40, 45): # 3,4,5,
                d_dimension = k_threshold
                point_set, labels, outlier_num = load_arff2(root+every_file)
                print(every_file, len(point_set), len(point_set[0]), outlier_num)
                scores = SAOD(point_set, k_threshold, d_dimension)
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = round(auc(fpr,tpr), 4) 
                f1.write( str(every_file) + ',' + str(k_threshold) + ',' + str(roc_auc) + '\n')
                # print("SAOD: ",every_file,roc_auc)
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_k = k_threshold
            f2.write( str(every_file) + ',' + str(best_k) + ',' + str(best_auc) + '\n')
            print("SAOD: ",every_file,best_k, best_auc)
    f1.close()
    f2.close()
    return 0

def main():
    # k_threshold = 3 # 5  # 3
    # d_dim = k_threshold
    # # two_dim_test(fileName, k_threshold, d_dim)
    # data_path = "../2d/"
    # for root,dirs,files in walk(data_path):
    #     for every_file in files:
    #         two_dim_test(root+every_file, k_threshold, d_dim)

    # two_dim_test_all()
    # low_dim()
    # dami()
    experiment = OutlierExperiment()
    experiment.execute_all_experiments()


if __name__ == '__main__':
    main()
