#!/usr/bin/env python
#-*- coding:utf-8 -*- 
# author: LiJia
# email: laetella@outlook.com
# date: 2018-10-20 10:50:37
# updateDate: 2018-10-20 10:50:37
# described: compare outlier detection method from pyod package 

from utils import *
import math
from pyod.models.knn import KNN   # kNN detector  2000
from pyod.models.lof import LOF   # LOF detector   2000
from pyod.models.mo_gaal import MO_GAAL   # 2019 detector
from pyod.models.copod import COPOD 		# 2020
from pyod.models.suod import SUOD 		# 2021
from pyod.models.ecod import ECOD  		#2022
from pyod.models.kde import KDE  		#2022
from pyod.models.cblof import CBLOF  		#2022
from pyod.models.loci import LOCI  		#2022
from pyod.models.sod import SOD  		#2022
# from pyod.models.xgbod import XGBOD  # 2018  是一种半监督的聚类算法，需要训练数据集及类标号
# from pyod.models.vae import VAE    # 2018
from pyod.models.alad import ALAD
from algorithms import *
# from rdos import *
from kdeos import *
# from cfar import *
# from keras import backend as K 
# K.clear_session()
# from sklearn.cluster import OPTICS
# from sklearn.cluster import DBSCAN

def plt_all_pr3(labels, point_set, outlier_num, fileName, k_threshold):
	# point_set = point_set.tolist()
	point_set = np.array(point_set.tolist()).astype(np.float)
	labels = np.array(labels).astype(int)
	ps_size = len(point_set)
	# k_threshold = 20; 
	od_rate = outlier_num / ps_size
	# pesa = PESA(point_set, k_threshold, outlier_num)
	iesa = IESA(point_set, k_threshold, outlier_num)
	gisa = GISA(point_set, k_threshold, outlier_num)
	lof = LOF(contamination=od_rate, n_neighbors=k_threshold)
	knn = KNN(contamination=od_rate, n_neighbors=k_threshold)
	ocsvm = OCSVM(contamination=od_rate)
	abod = ABOD(contamination=od_rate, n_neighbors=k_threshold, method='fast')
	mogaal = MO_GAAL(contamination=od_rate)
	sos = SOS(contamination=od_rate)
	cfar = CFAR(point_set, k_threshold, outlier_num)
	kdeos = KDEOS(point_set, kmin=3, kmax=7)
	rdos = RDOS(point_set, k_threshold, outlier_num)
	plt.figure(0)
	i = 0; min_recall = 1
	color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#7F7F7F','#BCBD22','#17BECF','#FB1515','#D62728']
	for clf, name in [(knn, 'knn'),(lof,'lof'),(ocsvm,'ocsvm'),(abod,'abod'),(mogaal,'mogaal'),(sos,'sos')] :
		try:
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
		except Exception as e:
			print(name, fileName, e)
			continue
		else:
			plt.plot(clf_rec, clf_pre, label=name, c=color_arr[i])
		i += 1
	#   ,(pesa, 'pesa')
	for scores,name in [(cfar,'cfar'), (kdeos,'kdeos'), (rdos,'rdos'), (iesa, 'iesa'), (gisa, 'gisa')] :
		clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
		if min_recall > min(clf_rec):
			min_recall = min(clf_rec)
		plt.plot(clf_rec, clf_pre, label=name, c=color_arr[i])
		i += 1
	# plt.title("Precision-Recall-curve on %s"%(fileName))  plt.gcf().transFigure
	# plt.tight_layout()
	# , loc="upper right"
	plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim([min_recall, 1.0])
	plt.ylim([0.0, 1.1])
	# plt.show()  , dpi=1000  
	plt.savefig("../result/%s_k=%d.png"%(fileName, k_threshold), bbox_inches='tight')
	plt.close(0)

# 专利中生成的图，画的线用不同标记表示，计算HeartDisease数据的结果
def plt_all_pr4(labels, point_set, outlier_num, fileName, k_threshold):
	# point_set = point_set.tolist()
	point_set = np.array(point_set.tolist()).astype(np.float)
	labels = np.array(labels).astype(int)
	ps_size = len(point_set)
	# k_threshold = 20; 
	od_rate = outlier_num / ps_size
	# pesa = PESA(point_set, k_threshold, outlier_num)
	iesa = IESA(point_set, k_threshold, outlier_num)
	gisa = GISA(point_set, k_threshold, outlier_num)
	lof = LOF(contamination=od_rate, n_neighbors=k_threshold)
	knn = KNN(contamination=od_rate, n_neighbors=k_threshold)
	ocsvm = OCSVM(contamination=od_rate)
	abod = ABOD(contamination=od_rate, n_neighbors=k_threshold, method='fast')
	plt.figure(0)
	i = 0; min_recall = 1
	color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#7F7F7F','#BCBD22','#17BECF','#FB1515','#D62728']
	ls_arr = array(list(islice(cycle(['-','--','-.',':']), 10)))
	marker_arr = array(list(islice(cycle(['D','^','+','s','o','1','2','X']), 10)))
	for clf, name in [(knn, 'knn'),(lof,'lof'),(ocsvm,'ocsvm'),(abod,'abod')] :
		try:
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
		except Exception as e:
			print(name, fileName, e)
			continue
		else:
			plt.plot(clf_rec, clf_pre, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i])
		i += 1
	#   ,(pesa, 'pesa')(cfar,'cfar'), (kdeos,'kdeos'), (rdos,'rdos'), 
	for scores,name in [(iesa, 'iesa'), (gisa, 'gisa')] :
		clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
		if min_recall > min(clf_rec):
			min_recall = min(clf_rec)
		plt.plot(clf_rec, clf_pre, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i])
		i += 1
	# plt.title("Precision-Recall-curve on %s"%(fileName))  plt.gcf().transFigure
	# plt.tight_layout()
	# , loc="upper right"
	plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim([min_recall, 1.0])
	plt.ylim([0.0, 1.1])
	# plt.show()  , dpi=1000  
	plt.savefig("../result/%s_k=%d.png"%(fileName, k_threshold), bbox_inches='tight')
	plt.close(0)

# outlier label : 1  else : 0
def plt_two_clusters(point_set, clusters, fileName, clf_name):
    plt.figure(0)
    plt.xticks([])
    plt.yticks([])
    for i, point in enumerate(point_set) :
        # plt.annotate(point_set.index(point), xy = (point[0], point[1]),xycoords = 'data',fontsize=3)
        if clusters[i] == 1 :
            plt.scatter(point[0],point[1],color= 'r', marker = '*', s=80)
        else:
            plt.scatter(point[0],point[1],color= 'b', marker = 'o')
    plt.savefig('../result/%s_%s.png'%(fileName.split('/')[2].split('.')[0], clf_name))
    plt.close(0)

def compare_others():
	data_path = 'E:\\project\\ODdata\\arff2\\Arrhythmia\\'  # Wilt  Waveform Shuttle
	f1 = open('../result/others_auc.csv','w')
	k_threshold = 10
	for root,dirs,files in walk(data_path):
		for every_file in files:
			# print("now process file: ",every_file)
			point_set, labels, outlier_num = load_arff2(root+every_file)
			od_rate = outlier_num / len(point_set)
			mogaal = MO_GAAL(contamination=od_rate)
			copod = COPOD(contamination=od_rate)
			suod = SUOD(contamination=od_rate)
			ecod = ECOD(contamination=od_rate)
			for clf, name in [(mogaal, 'mogaal'),(copod,'copod'),(suod,'suod'),(ecod,'ecod')] :
				try:
					clf.fit(point_set)
					scores = clf.decision_function(point_set)  # outlier scores
					fpr, tpr, thresholds = roc_curve(labels, scores)
					roc_auc = auc(fpr,tpr)
					# clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
				except Exception as e:
					print(name, fileName, e)
					continue
				else:
					print(name, every_file, roc_auc)
					f1.write( str(every_file) + ',' + str(k_threshold) + ',' +str(name)+',' + str(roc_auc) + '\n')
	f1.close()

def plt_all_roc(labels, point_set, outlier_num, fileName, k_threshold):
	# point_set = point_set.tolist()
	point_set = np.array(point_set.tolist()).astype(np.float)
	labels = np.array(labels).astype(int)
	ps_size = len(point_set)
	# k_threshold = 7; 
	od_rate = outlier_num / ps_size
	# knn = KNN(contamination=od_rate, n_neighbors=k_threshold)
	lof = LOF() # contamination=od_rate, n_neighbors=k_threshold
	cblof = CBLOF() # contamination=od_rate, n_neighbors=k_threshold
	# cfar = CFAR(point_set, k_threshold, outlier_num)
	kde = KDE()
	sod = SOD()
	kdeos = KDEOS(point_set, kmin=3, kmax=7)
	# rdos = RDOS(point_set, k_threshold, outlier_num)
	# suod = SUOD()
	alad = ALAD()
	mogl = MO_GAAL()
	copod = COPOD()
	ecod = ECOD()
	saod = our_method(point_set, k_threshold, k_threshold)
	plt.figure(0)
	plt.rcParams['font.size']=16
	i = 0; 
	roc_dict = dict()
	# color:线条的颜色，b, g, r,c,m,y,k,w,#000000, 灰度值0.5，
	# linestyle4种：-, --, -. :
	# 标记字符：. , ov, ^,V, < > 1 2 3 4 s p * h H + x D d |
	color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#17BECF','#FB1515','#BCBD22','#D62728'] # '#17BECF','#FB1515', '#7F7F7F',
	ls_arr = array(list(islice(cycle(['-', '--','-.',':']),12)))
	marker_arr = array(list(islice(cycle(['x', '^','+','s', 'o','p', 'H']),12)))
	for clf, name in [(lof,'LOF'),(cblof, 'CBLOF'),(kde,'KDE'),(sod,'SOD')] :
		try:
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			auc_value = round(auc(fpr,tpr), 4)
			# print(name, auc_value)
			roc_dict[name] = auc_value
		except Exception as e:
			print(name, fileName, e)
		else:
			plt.plot(fpr, tpr, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
		i += 1
	#  , (cod, 'cod')
	# fpr, tpr, thresholds = roc_curve(labels, scores)
	# plt.plot(TPR, FPR, label=name, c='#FB1515')
	# TODO 到底哪个是横坐标  哪个是纵坐标
	# plt.plot(fpr, tpr, label="our method", c='#FB1515') cfar,'cfar'),  (rdos,'rdos'), 
	fpr, tpr, thresholds = roc_curve(labels, kdeos)
	auc_value = round(auc(fpr,tpr), 4)
	# print(name,auc_value)
	roc_dict['KDEOS'] = auc_value 
	# TPR, FPR = get_roc(labels, scores, outlier_num)
	plt.plot(fpr, tpr, label='KDEOS', c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
	# plt.plot(fpr, tpr, label='KDEOS', c=color_arr[i])
	i += 1
	for clf, name in [(alad,'ALAD'),(mogl, 'MOGL'),(copod,'COPOD'),(ecod,'ECOD')] :
		try:
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			auc_value = round(auc(fpr,tpr), 4)
			# print(name, auc_value)
			roc_dict[name] = auc_value
		except Exception as e:
			print(name, fileName, e)
		else:
			# plt.plot(fpr, tpr, label=name, c=color_arr[i])
			plt.plot(fpr, tpr, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
		i += 1
	fpr, tpr, thresholds = roc_curve(labels, saod)
	auc_value = round(auc(fpr,tpr), 4)
	# print(name,auc_value)
	roc_dict['SAOD'] = auc_value 
	# TPR, FPR = get_roc(labels, scores, outlier_num)
	# plt.plot(fpr, tpr, label='SAOD', c=color_arr[i])
	plt.plot(fpr, tpr, label='SAOD', c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
	i += 1
	sorted_dict = sorted(roc_dict.items(), key = lambda kv:(kv[1], kv[0]))
	print(fileName, sorted_dict)
	# plt.title("ROC-curve on %s"%(fileName))
	# plt.legend()
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.1])
	# plt.show()
	plt.savefig("../result/%s.png"%(fileName), dpi=500)
	plt.close(0)

def plt_npz_roc(data_name):
	# data_name = "optdigits"
	plt.rcParams['font.size']=20
	
	plt.figure(0)
	i = 0; 
	roc_dict = dict()
	color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#17BECF','#FB1515','#BCBD22','#D62728'] # '#17BECF','#FB1515', '#7F7F7F',
	ls_arr = array(list(islice(cycle(['-', '--','-.',':']),12)))
	marker_arr = array(list(islice(cycle(['x', '^','+','s', 'o','p', 'H']),12)))
	for model in ['LOF','CBLOF','KDE','SOD','KDEOS','ALAD','MOGL', 'COPOD','ECOD','SAOD']:
		data = np.load('E:\\project\\temp results for spectral\\result_v3.1_npz\\%s\\%s_%s.npz'%(data_name, model, data_name))
		fpr = data["arr_0"]
		tpr = data["arr_1"]
		auc_value = round(auc(fpr,tpr), 4)
		# print(name, auc_value)
		roc_dict[model] = auc_value
		plt.plot(fpr, tpr, label=model, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
		i += 1
	sorted_dict = sorted(roc_dict.items(), key = lambda kv:(kv[1], kv[0]))
	print(data_name, sorted_dict)
	plt.title("%s"%(data_name.capitalize()))
	if data_name == "wbc":
		plt.legend(loc=2,bbox_to_anchor=(1.05,1.0),borderaxespad=0., fontsize=16)
	# plt.legend()
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.1])
	# plt.show()
	plt.savefig("../result/%s.png"%(data_name.capitalize()), dpi=500, bbox_inches='tight')
	plt.close(0)

def get_1method_1data(method, point_name):
	'''
	description: 计算一个方法在一个数据集上的fpr， tpr并存到npz文件中
	return {*}
	'''	
	data = np.load('ALAD_glass.npz')
	fpr = data["arr_0"]
	tpr = data["arr_1"]
	plt.figure(0)
	plt.plot(fpr, tpr, label="ALAD", linewidth = 1.5)
	plt.savefig("../result/%s.png"%("alad_test"), dpi=300)
	plt.close(0)	
	return 0

def save_all_auc():
	'''
	description: 计算所有ODDS上6个数据的所有方法的fpr tpr 并存为npz文件
	return {*}
	'''	
	lof = LOF()
	cblof = CBLOF()
	kde = KDE()
	sod = SOD()
	alad = ALAD()
	mogl = MO_GAAL()
	copod = COPOD()
	ecod = ECOD()
	k_threshold = [45] # 25,8,40 45,35 "glass", "optdigits","satellite","thyroid","vowels",
	for pn_i, point_name in enumerate( ["wbc"]):
		point_set, labels, outlier_num = load_mat("../mat/%s.mat"%(point_name))
		kdeos = KDEOS(point_set, kmin=3, kmax=7)
		saod = our_method(point_set, k_threshold[pn_i], k_threshold[pn_i])
		fpr, tpr, thresholds = roc_curve(labels, kdeos)
		np.savez("%s_%s.npz"%("kdeos", point_name), fpr, tpr)
		fpr, tpr, thresholds = roc_curve(labels, saod)
		np.savez("%s_%s.npz"%("saod", point_name), fpr, tpr)
		for clf, name in [(lof,'LOF'),(cblof, 'CBLOF'),(kde,'KDE'),(sod,'SOD'),(alad,'ALAD'),(mogl, 'MOGL'),(copod,'COPOD'),(ecod,'ECOD')] :
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			np.savez("%s_%s.npz"%(name, point_name), fpr, tpr)
	return 0

def copod_optdigits():
	'''
	description: 运行单个方法和数据并保存
	return {*}
	'''	 
	# point_name = "satellite" #  satellite  vowels satellite
	for pn_i, point_name in enumerate( ["glass", "optdigits","satellite","thyroid","vowels","wbc"]):
	# point_set, labels, outlier_num = load_mat("/scratch/lijialj/od/%s.mat"%(point_name))
		point_set, labels, outlier_num = load_mat("../mat/%s.mat"%(point_name))
		point_set = np.array(point_set.tolist()).astype(np.float)
		labels = np.array(labels).astype(int)
		# clf = ECOD()
		# clf = COPOD()
		clf = ALAD()
		# clf = MO_GAAL()
		sum_auc = 0
		for i in range(5):
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			auc_value = round(auc(fpr,tpr), 4)
			print(point_name, "epoch: ", i, auc_value )
			sum_auc += auc_value
		print(point_name, "average auc: ", round(sum_auc/5, 4))
	# np.savez("%s_%s.npz"%("ALAD", point_name), fpr, tpr)
	return 0

def plt_npz_auc():
	'''
	description: 读取存储的fpr tpr，直接进行画图
	return {*}
 	'''	
	data = np.load('ALAD_glass.npz')
	fpr = data["arr_0"]
	tpr = data["arr_1"]
	plt.figure(0)
	plt.plot(fpr, tpr, label="ALAD", linewidth = 1.5)
	plt.savefig("../result/%s.png"%("alad_test"), dpi=300)
	plt.close(0)	
	return 0

def plt_subfig():
	lof = LOF()
	cblof = CBLOF()
	kde = KDE()
	sod = SOD()
	alad = ALAD()
	mogl = MO_GAAL()
	copod = COPOD()
	ecod = ECOD()
	fig = plt.figure(1)
	plt.rcParams['font.size']=16
	ax2 = fig.add_subplot(2,3,2)
	ax3 = fig.add_subplot(2,3,3)
	ax4 = fig.add_subplot(2,3,4)
	ax5 = fig.add_subplot(2,3,5)
	ax6 = fig.add_subplot(2,3,6)
	k_threshold = [25,8,40,45,35,45]
	for pn_i, point_name in enumerate( ["glass", "optdigits","satellite","thyroid","vowels","wbc"]):
		ax = fig.add_subplot(2,3,pn_i+1)
		point_set, labels, outlier_num = load_mat("../mat/%s.mat"%(point_name))
		point_set = np.array(point_set.tolist()).astype(np.float)
		labels = np.array(labels).astype(int)
		ps_size = len(point_set)
		kdeos = KDEOS(point_set, kmin=3, kmax=7)
		saod = our_method(point_set, k_threshold[pn_i], k_threshold[pn_i])
		i = 0; 

		roc_dict = dict()
		color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#17BECF','#FB1515','#BCBD22','#D62728'] # '#17BECF','#FB1515', '#7F7F7F',
		ls_arr = array(list(islice(cycle(['-', '--','-.',':']),12)))
		marker_arr = array(list(islice(cycle(['x', '^','+','s', 'o','p', 'H']),12)))
		for clf, name in [(lof,'LOF'),(cblof, 'CBLOF'),(kde,'KDE'),(sod,'SOD')] :
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			auc_value = round(auc(fpr,tpr), 4)
			roc_dict[name] = auc_value
			ax.plot(fpr, tpr, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
			i += 1
		fpr, tpr, thresholds = roc_curve(labels, kdeos)
		auc_value = round(auc(fpr,tpr), 4)
		roc_dict['KDEOS'] = auc_value 
		ax.plot(fpr, tpr, label='KDEOS', c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
		i += 1
		for clf, name in [(alad,'ALAD'),(mogl, 'MOGL'),(copod,'COPOD'),(ecod,'ECOD')] :
			clf.fit(point_set)
			scores = clf.decision_function(point_set)  # outlier scores
			fpr, tpr, thresholds = roc_curve(labels, scores)
			auc_value = round(auc(fpr,tpr), 4)
			roc_dict[name] = auc_value
			ax.plot(fpr, tpr, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
			i += 1
		fpr, tpr, thresholds = roc_curve(labels, saod)
		auc_value = round(auc(fpr,tpr), 4)
		roc_dict['SAOD'] = auc_value 
		ax.plot(fpr, tpr, label='SAOD', c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i], markersize=0.5, linewidth = 1.5)
		i += 1
		sorted_dict = sorted(roc_dict.items(), key = lambda kv:(kv[1], kv[0]))
		ax.set_xlabel('FPR')
		ax.set_ylabel('TPR')
		# ax.xlabel('FPR')
		# ax.ylabel('TPR')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.1])
		if pn_i == 5:
			ax.legend(loc=2,bbox_to_anchor=(1.05,1.0),borderaxespad=0., fontsize=16)
		plt.savefig("../result/%d.png"%(pn_i),bbox_inches='tight', dpi=500)
	plt.close(0)

def plt_6fig():
	data_name = "Glass" # 
	color_arr = ['#e379c3','#1F77B4','#FF7F0E','#2CA02C','#9467BD','#936056','#17BECF','#FB1515','#BCBD22','#D62728'] # '#17BECF','#FB1515', '#7F7F7F',
	ls_arr = array(list(islice(cycle(['-', '--','-.',':']),12)))
	marker_arr = array(list(islice(cycle(['x', '^','+','s', 'o','p', 'H']),12)))
	fig = plt.figure(1)
	plt.gca().yaxis.set_minor_formatter(NullFormatter()) # 防止部分label丢失
	plt.rcParams['font.size']=16
	plt.subplots_adjust(hspace=0.2, wspace=0.2)
	for pn_i, point_name in enumerate( ["glass", "optdigits","satellite","thyroid","vowels","wbc"]):
		ax = fig.add_subplot(2,3,pn_i+1)
		for j, model in ['LOF','CBLOF','KDE','SOD','KDEOS','ALAD','MOGL', 'COPOD','ECOD','SAOD']:
			data = np.load('%s_%s.npz'%(model, point_name))
			fpr = data["arr_0"]
			tpr = data["arr_1"]
			ax.plot(fpr, tpr, label=model, c=color_arr[j],ls=ls_arr[j],marker=marker_arr[j], markersize=0.5, linewidth = 1.5)
		ax.set_xlabel('FPR')
		ax.set_ylabel('TPR')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.1])
		plt.title(point_name.capitalize())
	return 0

def save_k_fpr():
	'''
	description: 分析算法对于参数k的敏感性
	return {*}
	'''    
	for pn_i, point_name in enumerate( ["mammography","thyroid", "lympho", "vowels","wbc"]):
		for k_threshold in (5, 7, 10 ,15, 20, 30, 40): 
			point_set, labels, outlier_num = load_mat("../mat/%s.mat"%(point_name))
			saod = our_method(point_set, k_threshold, k_threshold)
			fpr, tpr, thresholds = roc_curve(labels, saod)
			np.savez("%s_%d.npz"%(point_name, k_threshold), fpr, tpr)
	# fileName = '../sensitivity/arrhythmia.mat'  # size最大: mammography 维度最大：InternetAds 离群点比例最大： arrhythmia
	fileName = '../sensitivity/Arrhythmia_withoutdupl_norm_46.arff' # InternetAds_norm_02_v10
	point_set, labels, outlier_num = load_arff3(fileName)
	for k_threshold in (5, 7, 10 ,15, 20, 30, 40): 
		saod = our_method(point_set, k_threshold, k_threshold)
		fpr, tpr, thresholds = roc_curve(labels, saod)
		np.savez("%s_%d.npz"%("Arrhythmia_withoutdupl_norm_46", k_threshold), fpr, tpr)
	fileName = '../sensitivity/InternetAds_norm_02_v10.arff' # InternetAds_norm_02_v10
	point_set, labels, outlier_num = load_arff2(fileName)
	for k_threshold in (5, 7, 10 ,15, 20, 30, 40): 
		saod = our_method(point_set, k_threshold, k_threshold)
		fpr, tpr, thresholds = roc_curve(labels, saod)
		np.savez("%s_%d.npz"%("InternetAds_norm_02_v10", k_threshold), fpr, tpr)
	return 0

def one_method(data):
	'''
	description: to compare method: MO_GAAL COPOD SUOD ECOD
	# Annthyroid Arrhythmia Cardiotocography HeartDisease Hepatitis InternetAds PageBlocks
	# Parkinson PenDigits Pima Shuttle SpamBase Stamps Waveform WBC WDBC Wilt WPBC
	return {*}
	'''
	data_path = '/scratch/lijialj/od_2/'
	# data_path = 'E:\\project\\ODdata\\arff2\\%s\\'%(data)
	# data_path = '../%s/'%(data)
	k_threshold = 7
	for root,dirs,files in walk(data_path):
		for every_file in files:
			# print("now process file: ",every_file)
			# point_set, labels, outlier_num = load_mat(root+every_file)
			try:
				f1 = open('../result/MO_GAAL_%s_auc_2.csv'%(data),'a')
				point_set, labels, outlier_num = load_arff2(root+every_file)
				# print(every_file, len(point_set), len(point_set[0]), round(outlier_num/len(point_set),4))
				# clf = LOF()
				# clf = KDE()
				# clf = CBLOF(n_clusters=10)
				# clf = LOCI()  # LOCI 运行非常慢
				# clf = SOD()  # SOD 需要占用大量的CPU资源
				# clf = KNN()
				clf = MO_GAAL(stop_epochs=20)  # 运行慢 训练时间长
				# clf = SUOD()
				# clf = ECOD()
				# clf = COPOD()
				# clf = ALAD()
				# print(point_set)
				a = clf.fit(point_set)
				scores = clf.decision_function(point_set)  # outlier scores
				# scores = CFAR(point_set, 7, outlier_num)
				# scores = KDEOS(point_set, kmin=3, kmax=7)
				# scores = RDOS(point_set, k_threshold, outlier_num)
				fpr, tpr, thresholds = roc_curve(labels, scores)
				roc_auc = round(auc(fpr,tpr), 4) 
				f1.write( str(every_file) + ','  + str(roc_auc) + '\n')
				print("MO_GAAL: ",every_file, roc_auc)
				f1.close()
			except:
				roc_auc = 0.5000
				print(every_file, "error!")

def main():
	# for pn_i, point_name in enumerate( ["wbc"]):  #  "optdigits","satellite","thyroid","vowels","wbc"  "glass","optdigits","satellite","thyroid","vowels",
	# 	plt_npz_roc(point_name)
	copod_optdigits()
	# save_k_fpr()
	# a =  "string".capitalize()
	# print(a)
	# plt_npz_auc()
	# plt_subfig()
	# save_all_auc()
	# plt_6fig()
	# # # print("wbc: ")
	# point_set, labels, outlier_num = load_arff4("E:\\project\\ODdata\\arff4\\Pima_withoutdupl_35.arff")
	# # # print(len(point_set), len(point_set[0]), outlier_num)
	# point_set, labels, outlier_num = load_mat("../mat/satellite.mat")
	# # # data = np.load(file="../mat/26_satellite.npz", allow_pickle=True)
	# # # point_set = data['X']
	# # # labels = data['y']
	# # # outlier_num = 0
	# # # print(satellite)
	# # # print(point_set[0])
	# # # for p in point_set:
	# # # 	for v in p:
	# # # 		if  math.isnan(v):
	# # 			# print(v)
	# # 	# print(p)
	# # # point_set = np.nan_to_num(point_set)
	# plt_all_roc(labels, point_set, outlier_num, "satellite", 40)
	# data_path = '../mat/' 
	# k_threshold = 6
	# for root,dirs,files in walk(data_path):
	# 	for every_file in files:
	# 		print("now process file: ",every_file)
	# 		point_set, labels, outlier_num = load_mat(root+every_file)
	# 		plt_all_roc(labels, point_set, outlier_num, every_file, k_threshold)
	# # MO_GAAL 对于Parkinson 的withoutdupl 05 和10 会报错
	# for data in [  "InternetAds"]:# "Wilt",
	# 	one_method(data)
	# compare_others()
	# fileName = 'E:\\project\\ODdata\\arff2\\WPBC\\WPBC_withoutdupl_norm.arff'
	# point_set, labels, outlier_num = load_arff2(fileName)
	# od_rate = outlier_num / len(point_set)
	# mogaal = MO_GAAL(contamination=od_rate)
	# copod = COPOD(contamination=od_rate)
	# suod = SUOD(contamination=od_rate)
	# ecod = ECOD(contamination=od_rate)
	# for clf, name in [(mogaal, 'mogaal'),(copod,'copod'),(suod,'suod'),(ecod,'ecod')] :
	# 	try:
	# 		clf.fit(point_set)
	# 		scores = clf.decision_function(point_set)  # outlier scores
	# 		fpr, tpr, thresholds = roc_curve(labels, scores)
	# 		roc_auc = auc(fpr,tpr)
	# 		# clf_pre, clf_rec = get_pr(labels, scores, outlier_num)
	# 	except Exception as e:
	# 		print(name, fileName, e)
	# 		continue
	# 	else:
	# 		print(name, roc_auc)
	# 		# plt.plot(clf_rec, clf_pre, label=name, c=color_arr[i],ls=ls_arr[i],marker=marker_arr[i])
	# 	# i += 1
	return 0

if __name__ == "__main__" :
	main()
	# k = 3
	# fileName = "../2d/data5.dat"   # 5
	# fileName = "../2d/data28.dat"   # 5
	# fileName = "../2d/data29.dat"   # 5  
	# # k = 5
	# # fileName = "../2d/data1.dat"   # 1
	# # fileName = "../2d/data24.dat"   # 7   
	# point_set = loadData(fileName, float, ",")
	# ps_size = len(point_set)
	# k_threshold = 3; outlier_num = 5
	# od_rate = outlier_num / ps_size

	# train compare detector
	# clf_name = 'loci'	# lof 	# knn # abod	# mo_gaal
	# iesa = IESA(point_set, k_threshold, outlier_num)
	# gisa = GISA(point_set, k_threshold, outlier_num)
	# lof = LOF(contamination=od_rate, n_neighbors=k_threshold)
	# knn = KNN(contamination=od_rate, n_neighbors=k_threshold)
	# ocsvm = OCSVM(contamination=od_rate)
	# abod = ABOD(contamination=od_rate, n_neighbors=k_threshold, method='fast')
	# # mogaal = MO_GAAL(contamination=od_rate)
	# # sos = SOS(contamination=od_rate)
	# # loci = LOCI(contamination=od_rate, k=k_threshold)
	# # cfar = CFAR(point_set, k_threshold, outlier_num)
	# # kdeos = KDEOS(point_set, kmin=3, kmax=k_threshold)
	# # rdos = RDOS(point_set, k_threshold, outlier_num)
	
	# # ***********our method************
	# # # outliers = spt_cls(point_set, 4, outlier_num)
	# # print (outlier_num)
	# # plt_outliers(outliers, point_set, fileName)
	
	# # # knn = KNN(); lof = LOF(); abod  = ABOD(); loci = LOCI() ,(mogaal,'mogaal'),(sos,'sos')
	# for clf, name in [(knn, 'knn'),(lof,'lof'),(ocsvm,'ocsvm'),(abod,'abod')] :
	# 	clf.fit(point_set)
	# 	clusters = clf.predict(point_set)  # outlier labels (0 or 1)
	# 	plt_two_clusters(point_set, clusters, fileName, name)
	# 	# plt_outliers(clusters, point_set,fileName, name)
	# # (cfar,'cfar'), (kdeos,'kdeos'), (rdos,'rdos'), (iesa, 'iesa'), (gisa, 'gisa')
	# for scores,name in [(iesa, 'iesa'), (gisa, 'gisa')] :
	# 	# outliers = scores2outliers(scores, outlier_num)
	# 	# print(outliers)
	# 	clusters = scores2labels(scores, outlier_num)
	# 	plt_two_clusters(point_set, clusters, fileName, name)
	# 	# plt_outliers(clusters, point_set,fileName, name)
	
	# fileName = "../test/bupa.arff"   # 5
	# point_set, labels, outlier_num = load_colon32('../data/colon32.arff')
	# # print (outlier_num, len(point_set))' '
	# point_set, labels, outlier_num = load_arff2('../good-data/HeartDisease_withoutdupl_norm_02_v05.arff')
	# point_set, labels, outlier_num = load_cls2o('../uci-cls2o/appendicitis.arff')
	# plt_all_pr4(labels, point_set, outlier_num, "HeartDisease_withoutdupl_norm_02_v05", 25)
