import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils import metrics_util
from utils import data_utils

def write_result(file, str):
    f = open(file, "a+")
    f.write(str)
    f.close()


class XGB:
    def __init__(self, df,fold,m,dataset_index):
        self.df=df
        self.fold=fold
        self.m=m
        self.dataset_index=dataset_index

    def run(self):
        global metric
        X_train, X_test, y_train, y_test = data_utils.load_train_test_data(self.df,self.fold,self.m)
        del X_train['project_id']
        del X_test['project_id']
        if self.m=="Logistic_regression":
            model = LogisticRegression(max_iter=10000)
        elif self.m=="Random_forest":
            model = RandomForestClassifier(n_estimators=10, criterion='gini')
        else:
            model = XGBClassifier()    
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        y_score=model.predict_proba(X_test)[:,1]

        auc,precision, recall, f1, average_precision, fpr, tpr,ndcg=metrics_util.get_all_metrics(y_test,y_pred,y_score)
        if self.m in ["Logistic_regression","Random_forest"]:
            acc=model.score(X_test,y_test)
        else:
            acc=accuracy_score(y_test,y_pred)
        metric = np.array(metric) + np.array([auc,acc,precision, recall, f1])
        print(auc,acc,precision, recall, f1,list(y_test).count(0),list(y_test).count(1))

if __name__ == '__main__':
    global metric
    logging.basicConfig(
        format="%(asctime)s (Process %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO)
    datasetindlist=[0]
    m=["Complete_XGB","Logistic_regression","Random_forest","only_corecap","only_corewill","only_other_cap","only_other_will","only_pro_contri","only_pro_new","ab_corecap","ab_corewill","ab_other_cap", "ab_other_will","ab_pro_contri","ab_pro_new"]
    dataset_index=0
    df=data_utils.load_raw_data(["/data/projectdata/ghtall30.json","/data/projectdata/ghtall31.json","/data/projectdata/ghtall32.json","/data/projectdata/ghtall33.json"],dataset_index)
    
    printlist=[]
    for model_name in m:
        print(model_name)
        metric=[0,0,0,0,0]
        for fold in range(10):
            model = XGB(df,fold,model_name,dataset_index)
            model.run()
        metric = [x/10 for x in metric]
        print(metric)
        printlist.append(metric)
    for i in range(len(m)):
        print(m[i],printlist[i])