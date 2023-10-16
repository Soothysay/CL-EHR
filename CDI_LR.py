# Code used for Downstream task. Uncomment classifiers and change df2 to include the relevant task label
import pandas as pd
import numpy as np
df1=pd.read_csv('data/data/DECEnt_3M_auto_v4/c42/df_patient_embedding_per_day.csv')
print(len(df1))
df1=df1.drop_duplicates()
print(len(df1))
df2=pd.read_csv('data/data/c4/patient_label/df_MICU_transfer.csv') #df_MICU_transfer
df2=df2.drop_duplicates(subset=['user_id','timestamp'])
#df2['timestamp']=df2['timestamp']+(2*50)
#df2['timestamp']=df2['timestamp']+(53*1)
#print(df2['label'].value_counts())
df3=pd.merge(left=df1,right=df2,on=['user_id','timestamp'],how='inner')
print(len(df3))
#df3=df3.fillna(0)
#print(len(df3))
print(df3.columns)
print(df3['label'].value_counts())
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
skf = StratifiedKFold(n_splits=3)
y=df3['label']
print(df3.columns)
X=df3.drop(['label','timestamp','user_id'],axis=1).values
n_repetition=30
#clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=500)
clf = SVC(gamma='auto',probability=True) #RandomForestClassifier(n_estimators=1000, max_depth=2)
eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[]}
#undersample = RandomUnderSampler(sampling_strategy='majority')
for rep in range(n_repetition):
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        #X_train, y_train = undersample.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        #pred = clf.predict(X_test)
        pred_prob = clf.predict_proba(X_test)
        probs = pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
        auc = metrics.auc(fpr, tpr)
        eval_results["test_auc"].append(auc)
        eval_results["test_fpr"].append(fpr)
        eval_results["test_tpr"].append(tpr)
        train_pred = clf.predict(X_train)
        train_pred_prob = clf.predict_proba(X_train)
        train_probs = train_pred_prob[:,clf.classes_ == True].flatten()
        fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)

        auc = metrics.auc(fpr, tpr)
        eval_results["train_auc"].append(auc)
print(np.std(eval_results['test_auc']))
print(np.mean(eval_results['test_auc']))
