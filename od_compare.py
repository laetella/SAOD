#!/usr/bin/env python

from utils import *
import math
from pyod.models.knn import KNN  
from pyod.models.lof import LOF   
from pyod.models.mo_gaal import MO_GAAL   
from pyod.models.copod import COPOD 		
from pyod.models.suod import SUOD 		
from pyod.models.ecod import ECOD  		
from pyod.models.kde import KDE  	
from pyod.models.cblof import CBLOF  	
from pyod.models.loci import LOCI  	
from pyod.models.sod import SOD  	
from pyod.models.alad import ALAD
from SAOD import *
from kdeos import *
from ADRN import ADRN
from HDIOD import HDIOD
RESULT_DIR = '../result'
os.makedirs(RESULT_DIR, exist_ok=True)

class OutlierExperiment:
    """outlier detection framework"""
    
    def __init__(self):
        self.models = {
            'MO_GAAL': MO_GAAL,
            'COPOD': COPOD,
            'SUOD': SUOD,
            'ECOD': ECOD,
            'ADRN': ADRN,
            'HDIOD': HDIOD,
            'LOF': LOF,
            'CBLOF': CBLOF,
            'KDE': KDE,
            'SOD': SOD,
            'ALAD': ALAD,
            'KDEOS': lambda p: KDEOS(p, kmin=3, kmax=7),
            'SAOD': lambda p, k: SAOD(p, k, k)
        }
        self.datasets = {
            "glass": {"k_threshold": 25},
            "optdigits": {"k_threshold": 8},
            "satellite": {"k_threshold": 40},
            "thyroid": {"k_threshold": 45},
            "vowels": {"k_threshold": 35},
            "wbc": {"k_threshold": 45}
        }

    def load_dataset(self, name):
        """load the  ODDS mat data"""
        if name in ["glass", "optdigits", "satellite", "thyroid", "vowels", "wbc"]:
            point_set, labels, outlier_num = load_mat(f"../mat/{name}.mat")
            return point_set, labels, outlier_num
        else:
            raise ValueError(f"Please select one data : {"glass", "optdigits", "satellite", "thyroid", "vowels", "wbc"}")

    def execute_model(self, model_name, dataset_name, **kwargs):
        """Execute specified model on dataset"""
        point_set, labels, outlier_num = self.load_dataset(dataset_name)
        k = self.datasets[dataset_name]["k_threshold"]
        
        # Model-specific handling
        if model_name == "ADRN":
            model = ADRN(alpha=0.1, m=20, max_iter=30, tol=1e-6, 
                         step_size_w=0.001, step_size_s=0.01)
            model.fit(point_set, verbose=False)
            scores = model.detect_anomalies(point_set, k=k, return_scores=True)[1]
            
        elif model_name == "HDIOD":
            model = HDIOD(k=k)
            model.fit(point_set)
            scores = model.cof_scores
            
        elif model_name in ["KDEOS", "SAOD"]:
            if model_name == "KDEOS":
                scores = self.models[model_name](point_set)
            else:
                scores = self.models[model_name](point_set, k)
                
        else:
            od_rate = outlier_num / len(point_set)
            model = self.models[model_name](contamination=od_rate, **kwargs)
            model.fit(point_set)
            scores = model.decision_function(point_set)
            
        return labels, scores

    def calculate_auc(self, labels, scores):
        """Calculate AUC and ROC curve"""
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = round(auc(fpr, tpr), 4)
        return fpr, tpr, roc_auc

    def save_results(self, model_name, dataset_name, fpr, tpr):
        """Save experiment results"""
        os.makedirs(RESULT_DIR, exist_ok=True)
        np.savez(f"{RESULT_DIR}/{model_name}_{dataset_name}.npz", fpr=fpr, tpr=tpr)
        print(f"Saved {model_name} results for {dataset_name}")

    def compare_others(self):
        """Compare  other outlier detection models  on DAMI dataset"""
        data_path = 'E:\\project\\ODdata\\arff2\\Arrhythmia\\' # change the data name here
        k = 10
        with open(f'{RESULT_DIR}/others_auc.csv', 'w') as f:
            for file in os.listdir(data_path):
                point_set, labels, outlier_num = load_arff2(os.path.join(data_path, file))
                od_rate = outlier_num / len(point_set)
                
                for model_name in ['MO_GAAL', 'COPOD', 'SUOD', 'ECOD']:
                    try:
                        labels, scores = self.execute_model(model_name, 'arrhythmia')
                        fpr, tpr, roc_auc = self.calculate_auc(labels, scores)
                        f.write(f"{file},{k},{model_name},{roc_auc}\n")
                        print(f"{model_name} {file} {roc_auc}")
                    except Exception as e:
                        print(f"{model_name} {file} error: {str(e)}")

    def execute_all_experiments(self):
        """execute all experiments"""
        for model_name in self.models:
            for dataset_name in self.datasets:
                try:
                    fpr, tpr, roc_auc = self.run_model(model_name, dataset_name)
                    self.save_results(model_name, dataset_name, fpr, tpr)
                except Exception as e:
                    print(f"{model_name} not work on {dataset_name} : {str(e)}")

def main():
    """main function"""
    experiment = OutlierExperiment()
    experiment.execute_all_experiments()

if __name__ == "__main__" :
	main()
	