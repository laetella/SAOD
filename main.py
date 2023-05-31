# -*- coding: utf-8 -*-
'''
Description: 
version: 1.0
Author: Jia Li
Email: j.li@liacs.leidenuniv.nl
Date: 2022-05-30 07:44:42
LastEditTime: 2023-03-24 16:29:36
'''
from algorithms import *
from kdeos import *
import time
from pyod.models.lof import LOF   # LOF detector   2000
def two_dim_test(fileName, k_threshold, d_dim):
    # fileName = "../2d/4.dat" 
    point_set = loadData(fileName, float, ",")
    # plt_point(point_set,"data_syn1_2")
    outlier_factor = our_method(point_set, k_threshold, d_dim)
    plt_density_colorbar(point_set, outlier_factor, k_threshold, fileName)
    return 0

def two_dim_test_all():
    data_path = "../2d_data/"
    for root,dirs,files in walk(data_path):
        for every_file in files:
            for k_threshold in (3,): # 4,5,6,7,8,
                d_dim = k_threshold
                point_set = loadData(root+every_file, float, ",")
                outlier_factor = our_method(point_set, k_threshold, d_dim)
                indices = np.argsort(outlier_factor)
                plt_density_colorbar(point_set, outlier_factor, k_threshold, every_file)
    return 0

def low_dim():
    best_auc = 0; best_k = 3
    for k_threshold in (3,4,5,6,7,8,9,10,15,20,25,30,35, 40, 45): # 6,7,8,9,10,15,20,25,30,35 
        fileName = 'E:\\project\\ODdata\\mat\\pyOD_benchmark\\optdigits.mat'
        print(k_threshold, fileName)
        point_set, labels, outlier_num = load_mat(fileName)
        scores = our_method(point_set, k_threshold, k_threshold)
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
                scores = our_method(point_set, k_threshold, d_dimension)
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

def main_plt_matrix(point_set):
    kdt = KDTree(point_set)
    result_dist,result_index = kdt.query(point_set, 4)
    # print(result_index[0])
    for to_plt_point in [48,58, 68]: # 
    # for to_plt_point in [50,71]: # 
    # for to_plt_point in [0,1,73]: # 
        knn_set = []
        for knn_idx in result_index[to_plt_point] :
            knn_set.append(point_set[knn_idx])
        sim_matrix = cal_sim_matrix(knn_set, result_dist, result_index, 0)
        # plt_sim_matrix(result_index[to_plt_point], sim_matrix)
        diag_matrix = np.diag(np.sum(sim_matrix, axis = 0)) 
        # plt_sim_matrix(result_index[to_plt_point], diag_matrix)
        laplacian_matrix = diag_matrix - sim_matrix
        plt_sim_matrix(result_index[to_plt_point], laplacian_matrix)

def plt_2ev_figure(ev_list, name):
    '''
    description: 根据给定的特征值数组，画直方图和特征值的线
    return {*}
    '''    
    ps_size = len(ev_list)
    ev_arr = np.array(ev_list)
    plt.figure(0)
    plt.rcParams['font.size']=23
    plt.hist(ev_arr[:,1].ravel(), bins=30, rwidth=0.5, label="1st eigenvalue")  
    plt.hist(ev_arr[:,2].ravel(), bins=30, rwidth=0.5, label="2nd eigenvalue")  
    plt.hist(ev_arr[:,3].ravel(), bins=30, rwidth=0.5, label="largest eigenvalue")  
    plt.legend()  
    plt.savefig('../result/%s_hist.png'%(name), dpi=500, bbox_inches="tight")
    plt.close(0)
    
    plt.figure(0)
    plt.rcParams['font.size']=23
    plt.plot(ev_arr[:,1], label='1st eigenvalue', color='#4D85BD',ls='-',marker='x', markersize=3, linewidth = 1.5)
    plt.plot(ev_arr[:,2], label='2nd eigenvalue', color='#F7903D',ls='--',marker='^', markersize=3, linewidth = 1.5)
    plt.plot(ev_arr[:,3], label='largest eigenvalue', color='#59A95A',ls='-.',marker='+', markersize=3, linewidth = 1.5)
    # plt.plot(ev_arr[:,0], label='0 eigenvalue', color='#2CA02C',ls=':',marker='p', markersize=0.5, linewidth = 0.5)
    # plt.plot(ev_arr[:,1], label='1st eigenvalue', color='#e379c3',ls='-',marker='x', markersize=1.5, linewidth = 1.5)
    # plt.plot(ev_arr[:,2], label='2nd eigenvalue', color='#1F77B4',ls='--',marker='^', markersize=1.5, linewidth = 1.5)
    # plt.plot(ev_arr[:,3], label='largest eigenvalue', color='#9467BD',ls='-.',marker='+', markersize=1.5, linewidth = 1.5)
    # # plt.plot(ev_arr[:,4], label='4 eigenvalue', color='#2CA02C',ls=':',marker='s', markersize=0.5, linewidth = 0.5)
    # # plt.plot(ev_arr[:,5], label='5 eigenvalue', color='#FB1515',ls='-.',marker='o', markersize=0.5, linewidth = 0.5)
    # # plt.plot(ev_arr[:,6], label='6 eigenvalue', color='c',ls=':',marker='H', markersize=0.5, linewidth = 0.5)
    # plt.xlabel('point')
    # plt.ylabel('eigenvalues')
    # plt.title("k eigenvalues")
    # # plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=90)
    # # plt.xticks(np.arange(15), dataName, rotation=90, size=5)
    plt.legend()
    plt.savefig('../result/%s_value.png'%(name), dpi=500, bbox_inches="tight")
    plt.close(0)
    return 0

def plt_eigenvlue_line(point_set, k_threshold):
    kdt = KDTree(point_set)
    k_threshold = 4
    result_dist,result_index = kdt.query(point_set, k_threshold)
    all_values = []; ev_list = []
    ps_size = len(point_set)
    for i in range(ps_size) :
        temp_ps = []
        for p in result_index[i] :
            temp_ps.append(point_set[p])
        eigVals = cal_eigvals(temp_ps, result_dist, result_index,i, k_threshold)
        all_values.extend(eigVals)
        ev_list.append(eigVals)
    ev_arr = np.array(ev_list)
    # min_index2 = np.argsort(ev_arr[:,1])
    plt_2ev_figure(ev_list, 0)
    return all_values

def plt_3D(point_set, label, name):
    # plt.rcParams['font.size']=18
    # print("label: ", label)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # New in version 1.0.0
    plt.setp(ax.get_xticklabels(), rotation=20, rotation_mode="default", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_zticklabels(), fontsize=8)
    # ax.set_zticklabels(fontsize=10)
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    # ax.text(x=3.5, y=3, z=3.5, s="z", size=10) #  k=4
    # ax.text(x=4.5, y=4.1, z=4.7, s="z", size=10) #  k=6
    ax.text(x=4.9, y=4.6, z=4.8, s="z", size=10) #  k=6
    # ax.set_zlabel('z') 
    # ax.zaxis.set_label_coords(.3, .3,)
    for i, point in enumerate(point_set) :
        # x = point[1]
        # y = point[2]
        # z = point[3]
        x = point[3]
        y = point[4]
        z = point[5]
        # # if label[i] == [1]:
        #     color = "red"
        # else:
        #     color = "blue"
        ax.scatter(x,y,z, s=10)  # , c=color
        # if i == 94:
        #     ax.text(x-0.1,y+0.1,z,i,  fontsize=10)
        # elif i == 70:
        #     ax.text(x-0.1,y-0.1,z-0.1,i,  fontsize=10)
        # elif i == 69:
        #     ax.text(x-0.3,y,z,i,  fontsize=10)
        # else:  
        # if i in [42,81,189,190,479,481]:#  68,69,70,71,72,73 0,1,2,3,4  94,95,96,97,98
        #     ax.text(x,y,z,i, fontsize=10)
        # else:
        #     ax.text(x,y,z,i, fontsize=6)
        ax.text(x,y,z,i, fontsize=10)
    # plt.xticks(rotation=90, fontsize=18)
    # plt.yticks(rotation=90, fontsize=18)
    # plt.zticks(fontsize=18)
    # plt.savefig('../result/%s_lapla_2.png'%(name), dpi=500)
    plt.savefig('../result/%s_lapla_2.png'%(name), dpi=500, bbox_inches="tight")
    return 0

def eigenvalue_3d():
    # fileName = "E:\\project\\ODdata\\2d\\2d-comma\\data_l2.dat" 
    fileName = "../2d/data_syn1_1.dat"  
    point_set = loadData(fileName, float, ',') # 20point

    # plt.figure(0)
    # fig, ax = plt.subplots() 
    # # for i, point in enumerate(point_set) :
    # # 	ax.scatter(point[0],point[1],s=20)
    # # 	ax.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=6)
    # # plt.savefig('../result/syn1_point.png', dpi=1000)

    kdt = KDTree(point_set)
    k_threshold = 6
    result_dist,result_index = kdt.query(point_set, k_threshold)
    dist_ev_arr = []; lapla_ev_arr = []
    for i, point in enumerate(point_set) :
        temp_ps = []
        for p in result_index[i] : 
            temp_ps.append(point_set[p])
        lapla_ev = cal_eigvals(temp_ps, result_dist,result_index,i, k_threshold)
        # dist_ev_arr.append(dist_ev)
        lapla_ev_arr.append(lapla_ev)
    plt_3D(lapla_ev_arr, 0, "syn1_k=6_2")
    return 0

def plt_real_ev():
    fileName = '../sensitivity/InternetAds_norm_02_v10.arff' # InternetAds_norm_02_v10
    point_set, labels, outlier_num = load_arff2(fileName)
    kdt = KDTree(point_set)
    k_threshold = 10
    result_dist,result_index = kdt.query(point_set, k_threshold)
    dist_ev_arr = []; lapla_ev_arr = []
    for i, point in enumerate(point_set) :
        temp_ps = []
        for p in result_index[i] : 
            temp_ps.append(point_set[p])
        lapla_ev = cal_eigvals(temp_ps, result_dist,result_index,i, k_threshold)
        # dist_ev_arr.append(dist_ev)
        lapla_ev_arr.append(lapla_ev)
    x = np.array(lapla_ev_arr)[:, -1]
    y = np.array(lapla_ev_arr)[:, -2]
    plt.scatter(x,y,s=20,c=labels)
    plt.savefig("inter_k=10.png", dpi=500)
    return 0

def analysis_k():
    fileName = '../mat/satellite.mat'  #  vowels lympho  wbc  thyroid
    # fileName = '../sensitivity/Arrhythmia_withoutdupl_norm_46.arff' # InternetAds_norm_02_v10
    # point_set, labels, outlier_num = load_arff3(fileName)
    # print("labels: ", labels)
    # best_auc = 0; best_k = 3
    point_set, labels, outlier_num = load_mat(fileName)
    color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#BCBD22','#D62728'] # '#17BECF','#FB1515', '#7F7F7F',
    # color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#7F7F7F','#BCBD22','#17BECF','#FB1515','#D62728']
    ls_arr = array(list(islice(cycle(['-','--','-.',':']), 10)))
    marker_arr = array(list(islice(cycle(['D','^','+','s','o','1','2','X']), 10)))
    i = 0
    plt.figure(0)
    for k_threshold in (5, 7, 10 ,15, 20, 30, 40): # 6,7,8,9,10,15,20,25,30,35  3,4,5,6,7,8,9,10,15,20,25,30,35, 40, 45
        # print(k_threshold, fileName)
        # point_set, labels, outlier_num = load_http(fileName)
        # print(point_set)
        d_dim = k_threshold
        try:
            scores = our_method(point_set, k_threshold, d_dim)
            fpr, tpr, thresholds = roc_curve(labels, scores)
            auc_value = round(auc(fpr,tpr), 4)
            print(k_threshold,", auc_value: ", auc_value)
        except Exception as e:
            print(k_threshold, fileName, e)
        else:
            plt.plot(fpr, tpr, label="$k$=%s"%(k_threshold), c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i],markersize=5)
        i += 1
    # plt.title("ROC-curve on %s"%(fileName))
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    # plt.show()
    plt.savefig("../result/%s.png"%(fileName.split('/')[-1].split('.')[0]), dpi=300)
    plt.close(0)
    return 0

def plt_k_auc():
    color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#BCBD22','#D62728']
    ls_arr = array(list(islice(cycle(['-','--','-.',':']), 10)))
    marker_arr = array(list(islice(cycle(['D','^','+','s','o','1','2','X']), 10))) 
    # _withoutdupl_norm_46  _norm_02_v10
    # for pn_i, point_name in enumerate( ["mammography","thyroid", "lympho", "vowels","wbc", "Arrhythmia", "InternetAds"]):
    for pn_i, point_name in enumerate( ["wbc"]):
        plt.figure(0)
        plt.rcParams['font.size']=20
        i = 0; 
        for k_threshold in (5, 7, 10 ,15, 20, 30, 40): 
            data = np.load('../../temp results for spectral/result_v3.1_npz/sensitivity/%s_%d.npz'%(point_name, k_threshold))
            # data = np.load('../result/real_result_v3.0/sensitivity/%s_%d.npz'%(point_name, k_threshold))
            fpr = data["arr_0"]
            tpr = data["arr_1"]
            auc_value = round(auc(fpr,tpr), 4)
            plt.plot(fpr, tpr, label="$k$=%s"%(k_threshold), c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i],markersize=5)
            i+=1
        if point_name == "wbc":
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(point_name.capitalize())
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.1])
        plt.savefig("../result/%sk.png"%(point_name), dpi=500, bbox_inches='tight')
        plt.close(0)
    return 0

def analysis_distribution():
    fileName = '../2d/data_l2.dat'  # data5  28 3 l2
    # point_set, labels, outlier_num = load_mat(fileName)
    # plt_high_dimension_data(fileName, point_set, labels)
    point_set = loadData(fileName, float, ',') 
    name = fileName.split("/")[-1].split(".")[0]
    plt.figure(0)
    plt.rcParams['font.size']=18
    for i, point in enumerate(point_set) :
        plt.scatter(point[0],point[1],s=15)
        if i in [42,81,189,190,479,481]:# 0,1,2,3,4  94,95,96,97,98 191,192,193,194,195,196,
            plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=15)
        else:
            plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=5)
    plt.savefig('../result/%s_source_2.png'%(name), dpi=500)
    plt.close(0)

    kdt = KDTree(point_set)
    k_threshold = 4
    result_dist,result_index = kdt.query(point_set, k_threshold)
    dist_ev_arr = []; lapla_ev_arr = []
    for i, point in enumerate(point_set) :
        temp_ps = []
        for p in result_index[i] : 
            temp_ps.append(point_set[p])
        lapla_ev = cal_eigvals(temp_ps, result_dist,result_index,i, k_threshold)
        # dist_ev_arr.append(dist_ev)
        lapla_ev_arr.append(lapla_ev)
    # print("size: ", len(point_set), len(lapla_ev_arr))
    # plt_3D(lapla_ev_arr, 0, name)
    plt_2ev_figure(lapla_ev_arr,name)
    return 0

def plt_point_index():
    fileName = "../2d/data_syn1_1.dat"  # f  data28 5point  data_syn1_1===k-3
    point_set = loadData(fileName, float, ",")
    plt.rcParams['font.size']=20
    # plt.xticks(size=18)
    # plt.yticks(size=18)
    for i, point in enumerate(point_set) :
        plt.scatter(point[0],point[1],s=15)
        # plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=7)
        if i in [8,14,20,26,32]:
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-1.5, point[1]), xycoords = 'data',fontsize=14)
        elif i in[9,10,11,12,15,16,17,18,21,22,23,24,27,28,29,30,33, 34,35,36]:
            # plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-0.5, point[1]), xycoords = 'data',fontsize=14)
            continue
        elif i in[9,15,21,27,33]:
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-0.5, point[1]), xycoords = 'data',fontsize=14)
        elif i in[2,3,4,5,6,7]:
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-1, point[1]-1), xycoords = 'data',fontsize=14)
        elif i in [45,47, 50, 49] :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-0.8, point[1]), xycoords = 'data',fontsize=14)
        elif i in [43,44] :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-1.5, point[1]), xycoords = 'data',fontsize=14)
        elif i in [42,] :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-0.5, point[1]+0.5), xycoords = 'data',fontsize=14)
        elif i in [39,] :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]+0.5, point[1]+0.5), xycoords = 'data',fontsize=14)
        elif i == 41 :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]-0.5, point[1]-0.5), xycoords = 'data',fontsize=14)
        elif i == 0 :
            plt.annotate(i, xy = (point[0], point[1]),xytext= (point[0]+0.5, point[1]-0.5), xycoords = 'data',fontsize=14)
        else:
            plt.annotate(i, xy = (point[0], point[1]),xycoords = 'data',fontsize=14)
    plt.savefig('../result/data_syn1_1_point14.png', dpi=400)
    return 0

def plt_knn_graph():
    fileName = "../2d/data_syn1_1.dat"  # f  data28 5point  data_syn1_1===k-3
    point_set = loadData(fileName, float, ",")
    kdt = KDTree(point_set)
    k_threshold = 4
    result_dist,result_index = kdt.query(point_set, k_threshold)
    # print(result_index)
    for i, point in enumerate( point_set):
        if i in [0, 1, 73]:
            plt.figure(0)
            plt.rcParams['font.size']=30
            temp_set = [i]
            for k_idx in range(1, len(result_index[i])):
                temp_set.append(result_index[i][k_idx])
            print(temp_set)
            for p_1 in range(len(temp_set)-1) :
                for p_2 in range(p_1, len(temp_set)):
                    x=[point_set[temp_set[p_1]][0], point_set[temp_set[p_2]][0]]
                    y=[point_set[temp_set[p_1]][1], point_set[temp_set[p_2]][1]]
                    # y=[temp_set[p_1][1], temp_set[p_2][1]]
                    plt.plot(x,y, c ='b', linewidth=5)
                    plt.annotate(temp_set[p_1], xy = (point_set[temp_set[p_1]][0], point_set[temp_set[p_1]][1]),xycoords = 'data')
                # x = [point[0], point_set[result_index[i][k_idx]][0]]
                # y = [point[1], point_set[result_index[i][k_idx]][1]]
                # print("plotting: ",i, result_index[i][k_idx])
                # plt.plot(x,y, c ='b', linewidth=5)
                # plt.annotate(result_index[i][k_idx], xy = (point_set[result_index[i][k_idx]][0], point_set[result_index[i][k_idx]][1]),xycoords = 'data')
                    plt.scatter(x,y,s=80, c='r')
            plt.xticks([])
            plt.yticks([])
            plt.annotate(temp_set[-1], xy = (point_set[temp_set[-1]][0], point_set[temp_set[-1]][1]),xycoords = 'data')
            plt.savefig('../result/knn%d.png'%(i), dpi=200)
            plt.close(0)
    return 0

def plt_values():
    value_arr1 = [-4.440892098500626e-16, 2.2019652891754538, 2.5649960130217306, 2.9040435350130207]
    value_arr2 = [1.6653345369377348e-16, 0.6213885695077558, 0.7986461230594943, 1.1460191874407604]
    value_arr3 = [-2.7755575615628914e-17, 0.1692710320013631, 0.43972329877480865, 1.6806865035441276]
    # x_arr = []
    plt.figure(0)
    plt.rcParams['font.size']=16
    plt.plot(value_arr1, label='point 48')
    plt.plot(value_arr2, label='point 58')
    plt.plot(value_arr3, label='point 68')
    plt.plot(value_arr1, 'k.')
    plt.plot(value_arr2, 'k.')
    plt.plot(value_arr3, 'k.')
    plt.xticks(np.arange(4), ['1st', '2nd', '3rd', '4th'])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig('../result/3eigenvalue.png', dpi=500, bbox_inches='tight')
    plt.close(0)
    return 0

def main():
    pdf2img()
    # change_img()
    # # # 画论文中的图
    # plt_71()
    # eigenvalue_3d()
    # plt_point_index()
    # plt_knn_graph()
    # # plt_values()
    fileName = '../2d/data_syn1_1.dat'  # data5  28 3 l2  data_syn1_1 data24
    point_set = loadData(fileName, float, ',') 
    main_plt_matrix(point_set)
    # # k_threshold = 4
    # plt_eigenvlue_line(point_set, k_threshold)
    # plt_kde_distribution(point_set, k_threshold, result_dist,result_index)

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
    # load_http("E:\\project\\ODdata\\mat\\Outlier  detection\\http.mat")

    # ablation experiment
    # analysis_k()
    # analysis_distribution()
    # plt_real_ev()
    # plt_k_auc()
    # time complexity
    # complexity_analysis()

    # other test 
    # test_sum()
    # test_normalized_lapla()

if __name__ == '__main__':
    main()
