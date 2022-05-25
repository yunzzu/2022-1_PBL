



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./online_shoppers_intention.csv')
df.shape
df.describe()
df.isnull().sum()


#%%EDA(Exploratory Data Analysis)
#1. VisitorType vs Duration 
fig, axes = plt.subplots(3,1, figsize=(13,13))

ax1 = sns.boxplot(x="VisitorType",y="Administrative_Duration", data=df,hue='Revenue',palette=['#d66354','#5486d6'], ax = axes[0])
ax1.set_xlabel("VisitorType", fontsize=12)
ax1.set_ylabel("Administrative Duration", fontsize=12)
ax1.set_title("Administrative Duration by VisitorType", fontsize=14)
ax1.set_yscale('log')

ax2 = sns.boxplot(x="VisitorType",y="Informational_Duration", data=df,hue='Revenue',palette=['#d66354','#5486d6'], ax = axes[1])
ax2.set_xlabel("VisitorType", fontsize=12)
ax2.set_ylabel("Informational Duration", fontsize=12)
ax2.set_title("Informational Duration by VisitorType", fontsize=14)
ax2.set_yscale('log')

ax3 = sns.boxplot(x="VisitorType",y="ProductRelated_Duration", data=df,hue='Revenue',palette=['#d66354','#5486d6'], ax = axes[2])
ax3.set_xlabel("VisitorType", fontsize=12)
ax3.set_ylabel("ProductRelated Duration", fontsize=12)
ax3.set_title("ProductRelated Duration by VisitorType", fontsize=14)
ax3.set_yscale('log')

plt.subplots_adjust(wspace = 0.2, hspace = 0.5, top = 0.9)

#2. Revenue by Exit Rates 
ax1 = sns.boxplot(x="Revenue",y="ExitRates", data=df,palette=['#d66354','#5486d6'])
ax1.set_xlabel("Revenue", fontsize=12)
ax1.set_ylabel("Exit Rates", fontsize=12)
ax1.set_title("Revenue by Exit Rates", fontsize=14)

#3. Weekend/Month vs Revenue
plt.figure(figsize = (13,10))

ax = plt.subplot(221)
ax = sns.countplot(x="Weekend", data=df, hue="Revenue")
ax.set_xlabel("Weekend", fontsize=12)
ax.set_ylabel("Count of customers", fontsize=12)
ax.set_title("Weekend by Revenue", fontsize=14)

ax2 = plt.subplot(212)
ax2 = sns.countplot(x="Month", data=df, hue="Revenue")
ax2.set_xlabel("Month", fontsize=12)
ax2.set_ylabel("Count of customers", fontsize=12)
ax2.set_title("Month by Revenue", fontsize=14)

plt.subplots_adjust(wspace = 0.6, hspace = 0.4, top = 0.9)

#4. GA 변수 비 (BounceRates, ExitRates, PageValues)
fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(12,10))
sns.distplot(df['BounceRates'], hist=False, ax=ax[0])
sns.distplot(df['ExitRates'], hist=False, ax=ax[1])
sns.distplot(df['PageValues'], hist=False, ax=ax[2])
#sns.distplot(df, x='BounceRates', hue='Revenue') #외않됌?

def distplot_with_hue(data=None, x=None, hue=None, row=None, col=None, legend=True, **kwargs):
    _, bins = np.histogram(data[x].dropna())
    g = sns.FacetGrid(data, hue=hue, row=row, col=col)
    g.map(sns.distplot, x, **kwargs)
    if legend and (hue is not None) and (hue not in [x, row, col]):
        g.add_legend(title=hue) 
distplot_with_hue(data=df, x='BounceRates', hue='Revenue', hist=False)
distplot_with_hue(data=df, x='ExitRates', hue='Revenue', hist=False)
distplot_with_hue(data=df, x='PageValues', hue='Revenue', hist=False)

#5. PageValues vs Revenue
sns.barplot(x='Revenue', y='PageValues', data=df)
sns.jointplot(data=df, x="BounceRates", y="ExitRates", hue="Revenue", palette = "Set2")

ax = sns.scatterplot(x="BounceRates", y="ExitRates",hue = 'Revenue',palette = "Set2", data=df)
ax.set_title('ExitRates vs BounceRates')


#%% 이상치 
#1. 범주형 자료의 수치화
#각 변수의 값(unique value)과 변수형 출력
df.info()
df['Month'].unique()
df['VisitorType'].unique()
df['Weekend'].unique()
df['Revenue'].unique()

for col in df.columns:
    print("{} have {} unique values: {}".format(col, df[col].nunique(), df[col].dtypes))
    if df[col].dtypes == 'int64' or df[col].dtypes == 'bool':
        print("{} values: {}".format(col,df[col].unique()))
        
#(VisitorType, Weekend, Month, Revenue) -교재 p.125
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Change bool and object value in label encoder
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

df['Month'].unique()
df['VisitorType'].unique()
df['Weekend'].unique()
df['Revenue'].unique()

#2. 이상치 제거 전 전체 변수 확인 
fig, axes = plt.subplots(nrows=3, ncols=6,figsize=(20,10))
for i, col in enumerate(df.columns):
    ax = axes.flatten()
    sns.boxplot(data=df, y=col, ax = ax[i])

#zscore 적용 
from scipy import stats
z=np.abs(stats.zscore(df))
filtered_entries = (z < 1.8).all(axis=1)
df = df[filtered_entries]
df.shape
#df1_new = df[filtered_entries]
#df1_new.shape

X_new = df.drop('Revenue', axis=1)
y_new = df['Revenue']


#%% Target Variable
#84% of values in Revenue is false
df['Revenue'].value_counts(normalize=True)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Revenue')
plt.title('Revenue', fontsize=15, fontweight='bold')
plt.show()


#%% 상관계수 파악 (heatmap)
corr = df.corr()
g = sns.heatmap(corr, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()
g.figure.set_size_inches(25,9)
plt.show()


#%% 특성변수의 선택 
#Revenue와 가장 corrleated된 변수를 k개 고르기 
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
sb = SelectKBest(score_func=f_classif, k=5)
x_new = sb.fit_transform(X_enc, y_enc)
#x_new.shape

all_names = X_new.columns
mask = sb.get_support()
selected_names = all_names[mask]
print("선택: ", selected_names)
print("선택X: ", all_names[~mask])

#결과해석 필요함 
sb.scores_
sb.pvalues_

#선택된 변수만 들어 있는 df 새로 생성
df2 = df[selected_names]


#%% 불균형자료 처리 
#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=.3, random_state=1)

#SMOTE / ADASYN
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
sm = SMOTE(random_state=0)
ada = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_sm,y_train_sm = sm.fit_resample(X_train, y_train)
X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)

#print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_sm).value_counts())
print('처리 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('처리 전 레이블 값 분포: %s' % Counter(y_train))
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_sm.shape, y_train_sm.shape)
print('SMOTE 적용 후 레이블 값 분포: %s' % Counter(y_train_sm))
print('ADASYN 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_ada.shape, y_train_ada.shape)
print('ADASYN 적용 후 레이블 값 분포: %s' % Counter(y_train_ada))


#%%표준화 -> 불균형자료 처리
#이상치 제거 전 데이터
df = pd.read_csv('./online_shoppers_intention.csv')
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Change bool and object value in label encoder
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

#data split
from sklearn.model_selection import train_test_split
X = df.drop('Revenue', axis=1)
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)

#표준화
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#SMOTE / ADASYN
sm = SMOTE(random_state=0)
ada = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_std_sm,y_train_std_sm = sm.fit_resample(X_train_std, y_train)
X_train_std_ada, y_train_std_ada = ada.fit_resample(X_train_std, y_train)

print('처리 전 학습용(표준화) 피처/레이블 데이터 세트: ', X_train_std.shape, y_train.shape)
print('처리 전 레이블(표준화) 값 분포: %s' % Counter(y_train))
print('SMOTE 적용 후 학습용(표준화) 피처/레이블 데이터 세트: ', X_train_std_sm.shape, y_train_sm.shape)
print('SMOTE 적용 후 레이블(표준화) 값 분포: %s' % Counter(y_train_sm))
print('ADASYN 적용 후 학습용(표준화) 피처/레이블 데이터 세트: ', X_train_std_ada.shape, y_train_ada.shape)
print('ADASYN 적용 후 레이블(표준화) 값 분포: %s' % Counter(y_train_ada))