
import pandas as pd
import numpy as np

df = pd.read_csv('./online_shoppers_intention.csv')

X = df.drop('Revenue', axis=1)
y = df['Revenue']


#%%전처리
#Label Encoder
from sklearn.preprocessing import LabelEncoder
X_enc = X.copy()
y_enc = y.copy()
for col in X.columns:
    if X[col].dtypes == 'object' or X[col].dtypes == 'bool':
        lb = LabelEncoder()
        X_enc[col] = lb.fit_transform(X[col].values)
        y_enc = lb.fit_transform(df['Revenue'])
        
df['Month'] = X_enc['Month']
df['VisitorType'] = X_enc['VisitorType']
df['Weekend'] = X_enc['Weekend']
df['Revenue'] = y_enc


#이상치 제거
from scipy import stats
z=np.abs(stats.zscore(df))
filtered_entries = (z < 3).all(axis=1)
df = df[filtered_entries]

X_new = df.drop('Revenue', axis=1)
y_new = df['Revenue']

#Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=.3, random_state=1)

#SelectKBest
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
sb = SelectKBest(score_func=f_classif, k=5)
x_new = sb.fit_transform(X_new, y_new)
all_names = X_new.columns
mask = sb.get_support()
selected_names = all_names[mask]
#print("선택: ", selected_names)
#print("선택X: ", all_names[~mask])

df2 = df[selected_names]  #선택된 변수만 들어 있는 df 새로 생성


#SMOTE / ADASYN
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
sm = SMOTE(random_state=0)
ada = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_sm,y_train_sm = sm.fit_resample(X_train, y_train)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)

#표준화
X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(X_enc, y_enc, test_size=.3, random_state=1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_enc_train)
X_train_std = sc.transform(X_enc_train)
X_test_std = sc.transform(X_enc_test)

#SMOTE / ADASYN
sm = SMOTE(random_state=0)
ada = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_std_sm,y_train_std_sm = sm.fit_resample(X_train_std, y_train)
X_train_std_ada, y_train_std_ada = ada.fit_resample(X_train_std, y_train)


























=======
X_train_std_ada, y_train_std_ada = ada.fit_resample(X_train_std, y_train)
>>>>>>> 6eda3ed900a05b873099a37ff473913eba86c3a9
