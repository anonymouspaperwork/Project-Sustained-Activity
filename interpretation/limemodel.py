import logging
import pandas as pd
import lime
import lime.lime_tabular
from utils import data_utils
from xgboost import XGBClassifier
from statistics import mean, median

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
logging.basicConfig(
    format="%(asctime)s (Process %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    level=logging.INFO)

df=data_utils.load_raw_data(["/data/xwx/ght/pro_lifetime/ghtall30.json","/data/xwx/ght/pro_lifetime/ghtall31.json","/data/xwx/ght/pro_lifetime/ghtall32.json","/data/xwx/ght/pro_lifetime/ghtall33.json"],0)
X_train, X_test, y_train, y_test = data_utils.load_train_test_data(df,9,"Complete_XGB")


trainproid=X_train[['project_id']]
testproid=X_test[['project_id']]
del X_train['project_id']
del X_test['project_id']
model = XGBClassifier()
model.fit(X_train,y_train)
c=list(X_test.columns)

explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values,mode="regression",feature_names=c,verbose=True)
lime_lst=[]
indlist=range(len(X_test))
for i in indlist:
    def p(data):
        data=pd.DataFrame(data)
        data.columns=X_train.columns
        return model.predict_proba(data)[:,1]
    exp = explainer.explain_instance(data_row=X_test.iloc[i],num_features=len(c), predict_fn=p)

    lime_lst.append(exp.as_list())
    logging.info(i)

res = pd.DataFrame({"lime_lst":lime_lst})
res.to_pickle("limeres.pkl")
logging.info("finish")