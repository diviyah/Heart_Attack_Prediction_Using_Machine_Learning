# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:19:43 2022

# =============================================================================
# ABOUT DATA 
# =============================================================================
Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type
    Value 1: typical angina
    Value 2: atypical angina
    Value 3: non-anginal pain
    Value 4: asymptomatic

trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results
Value 0: normal
Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach : maximum heart rate achieved

output : 0= less chance of heart attack, 
         1= more chance of heart attack

credits for the info and dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset


###############################################################################

Dataset Description (+medical definitions)

Age : Age of the patient

Sex : Sex of the patient

cp : Chest Pain type

Value 0: typical angina

Value 1: atypical angina

Value 2: non-anginal pain

Value 3: asymptomatic

trtbps : resting blood pressure (in mm Hg)

chol: cholesterol in mg/dl fetched via BMI sensor

fbs: (fasting blood sugar > 120 mg/dl)

1 = true

0 = false

rest_ecg: resting electrocardiographic results
Value 0: normal

Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach: maximum heart rate achieved

exang: exercise induced angina

1 = yes

0 = no

old peak: ST depression induced by exercise relative to rest

slp: the slope of the peak exercise ST segment

0 = unsloping

1 = flat

2 = downsloping

caa: number of major vessels (0-3)

thall : thalassemia

0 = null

1 = fixed defect

2 = normal

3 = reversable defect

output: diagnosis of heart disease (angiographic disease status)
0: < 50% diameter narrowing. less chance of heart disease

1: > 50% diameter narrowing. more chance of heart disease

Credits for the info: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/329925

From this website that, we know of the 0 value existence in 'thall' is null --> NaN


# =============================================================================
# STEP-BY_STEP 
# =============================================================================
#EDA
1. Data Loading
2. Data Inspection
3. Data Cleaning
4. Feature Selection
5. Data Pre-processing
Model Development
Model Evaluation
Discussion

@author: ASUS
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

#%%    #### Confusion Matrix

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% STATIC 
DATA_PATH = os.path.join(os.getcwd(), 'dataset','heart.csv')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'model','best_pipeline_model.pkl')
BEST_GSCVMODEL_PATH = os.path.join(os.getcwd(),'model','best_GSCV_model.pkl')

#%% STEP 1 - Data Loading

df = pd.read_csv(DATA_PATH)

# Back up file
df_backup =df.copy()


#%% STEP 2 - Data Inspection

df.info() # No object dtypes found
          # only oldpeak is float64 and others are int64
          # int64 doesnt determine its continuity or categorical trait
          # Diving deeper...

df.describe().T # Mean and median are almost similar 
                # Comparatively, chol, thalachh has larger difference for mean and median compared to others 
                # Could be an indicates of fewer presence of outliers

plt.figure(figsize=(20,10))
df.boxplot()
plt.show()  #trtbps, chol has more outliers than fbs, thalachh,oldpeak,caa and thall
            #Especially chol has wider dispersion of data compared to other variables

msno.matrix(df) # no missing values as well


column_names = df.columns

# Check for NaNs

df.isna().sum()  #no NaN present in the data

# Check for duplicated data
df.duplicated().sum()  #has 1 duplicated data
df[df.duplicated()] 

# Check for funny values
df_backup[df_backup['age'] < 0] # No negative values
df_backup[df_backup['trtbps'] < 0] # No negative values
df_backup[df_backup['chol'] < 0] # No negative values
df_backup[df_backup['thalachh'] < 0] # No negative values
df_backup[df_backup['oldpeak'] < 0] # No negative values


# thall has 0 eventhough there is no explanation for it
# thall = 0 ==> indicates null >> NaN
# Hence, need to impute it

# Check for presence of extreme values
#    trtbps ==> (94,200) #dont have to remove extreme values, none to remove
#    chol ==> (126,564) # 564 is huge value since anything more than 200 is to be concerned
                        # But its also depends on the machine's calibration as well
                        # not just due to funny readings
#    thalachh (71,202) # blood pressure reading. # dont have to remove extreme values
#    oldpeak (0,6.2)

cat_columns = ['sex','cp', 'fbs','restecg','exng','slp','caa','thall','output']
con_columns = ['age','trtbps','chol','thalachh','oldpeak']

# Categorical - Data Distribution

for cat in cat_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

for con in con_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

#%% STEP 3 - Data Cleaning

# TO DO  -> 1. Drop duplicated data 
#           2. Impute the data if got NaN #Impute thall
#           3. Check for presence of extreme values since this is a medical data 
#              (continuous data only!) - cholesterol can be considered as having extreme value of 500.
#           4. Label Encoder if the data is not numeric for categorical (pass)
     
#1.     
    # Initial data size --> (303,14)
    # After remove the duplicated data --> (302,14) since there was only 1 duplicated data
df = df.drop_duplicates()  
df.duplicated().sum() #Hence, 0 duplicated data now

#2. Data imputation using Simple Imputer
df['thall']=df['thall'].replace(0,np.nan) #So now has been replaced with NaNs
df.isna().sum() #There is 2 NaNs in thall

df['thall'] = df['thall'].fillna(df['thall'].mode()[0])
df.isna().sum() #There is no NaNs in the data
                # Data has been imputed

#%% STEP 4 - Feature Selection

y = df['output']
X = df.drop(labels=['output'],axis=1)

# Continuous versus categorical

for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis = 1), df['output'])
    print(lr.score(np.expand_dims(df[con], axis = 1), df['output']))

# we can take all of the continuous variable since more than 50%

# Categorical versus Categorical
for cat in cat_columns:
    print(cat)
    confussion_mat = pd.crosstab(df[cat], df['output']).to_numpy()   
    print(cramers_corrected_stat(confussion_mat))

# Choosing variable with more than 50% accuracy
# cp, thall

# CONCLUSION :
    # Selected features -> ['age','trtbps','chol','thalachh','oldpeak','cp','thall']

#%% STEP 5 - Data Preprocessing

X = df.loc[:,['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
y = df['output']


X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Pipeline

# STEPS
# 1. Determine whether MMS or SS is better in this case
# 2. Determine which classifier works the best
#         a - Logistic Regression
#         b - Random Forest
#         c - Decision Tree
#         d - KNN
#         e - SVC [ Support Vector Classifier ]

# Logistic Regression Pipeline
pl_ss_lr = Pipeline([('StandardScaler', StandardScaler()),
                      ('LogisticClassifier', LogisticRegression())])


pl_mms_lr = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('LogisticClassifier', LogisticRegression())])


# Random Forest Pipeline
pl_ss_rf = Pipeline([('StandardScaler', StandardScaler()),
                      ('RFClassifier', RandomForestClassifier())])


pl_mms_rf = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('RFClassifier', RandomForestClassifier())])


# Decision Tree Pipeline
pl_ss_dt = Pipeline([('StandardScaler', StandardScaler()),
                      ('DTClassifier', DecisionTreeClassifier())])


pl_mms_dt = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('DTClassifier', DecisionTreeClassifier())])


# KNN Pipeline
pl_ss_knn = Pipeline([('StandardScaler', StandardScaler()),
                       ('KNNClassifier', KNeighborsClassifier())])


pl_mms_knn = Pipeline([('MinMaxScaler', MinMaxScaler()),
                       ('KNNClassifier', KNeighborsClassifier())])


# SVC Pipeline
pl_ss_svc = Pipeline([('StandardScaler', StandardScaler()),
                      ('SVCClassifier', SVC())])

pl_mms_svc = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('SVCClassifier', SVC())])


pipelines = [pl_ss_lr,
             pl_mms_lr,
             pl_ss_rf,
             pl_mms_rf,
             pl_ss_dt,
             pl_mms_dt,
             pl_ss_knn,
             pl_mms_knn,
             pl_ss_svc,
             pl_mms_svc]

for pipeline in pipelines:
    pipeline.fit(X_train,y_train)
    
pipeline_scored = []

for i, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    pipeline_scored.append(pipeline.score(X_test,y_test))
    
best_pipeline = pipelines[np.argmax(pipeline_scored)]
best_accuracy = pipeline_scored[np.argmax(pipeline_scored)]
print('The best combination of the pipeline is {} with accuracy of {}'
      .format(best_pipeline.steps,best_accuracy))
  
# best model + accuracy ==>  MMS + LR with 0.7692307692307693 [76.9231%]     

     
# GridSearch CV - to find tune the model
# from the pipeline above, it is deduced that pipeline with MMS + LR
# acheived the highest accuracy when tested against test dataset

pl_mms_lr = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('LogisticClassifier', LogisticRegression())])

#estimator is the pipeline
# Always include the default value for grid_param dictionary

grid_param = [{'LogisticClassifier__random_state':[100,1000,None],
               'LogisticClassifier__C': [0.001,0.01,0.1,1,10]}]

grid_search = GridSearchCV(pl_mms_lr,grid_param,cv=5,n_jobs=2)
best_model = grid_search.fit(X_train,y_train)           

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

print(best_model.score(X_test,y_test))  #0.7692307692307693
print(best_model.best_index_)
print(best_model.best_params_)

# The best model after fine tuning gives the same accuracy using pipeline


#%% Retrain the model with the selected parameters


pl_mms_lr = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('LogisticClassifier', LogisticRegression(random_state=100,C=1))])

pl_mms_lr.fit(X_train,y_train)


#%% Model Saving 

with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_pipeline,file)

with open(BEST_GSCVMODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)
    
    
#%%  Model Evaluation

y_true = y_test
y_pred = best_pipeline.predict(X_test)

print(classification_report(y_true,y_pred)) #f1-score - 80%
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))  #Accuracy is 80%


#%% Discussion

# =============================================================================
# Problem Statement

# Given medical dataset, can we predict the chance of a person having heart attack?
# =============================================================================
# =============================================================================
# Exploratory Data Analysis
# Questions:
    # 1. What kind of data are we dealing with?
        # Can refer the "About Data" Section stated above.
        # our target variable is the 'output' variable

    # 2. Do we have missing values?
        # Variable thal had 0 as its value which represents null value. 
        # Hence, had it replaced with NaN to be imputed using mode of the particular variable        
     
    # 3. Do we have duplicated datas?
        # We had 1 and had it removed. 
        
    # 4. Do we have extreme values?
        # The data didnt possess any negative values which is logical.
        # However, chol variable do possess the maximum value up to 564 which is quite concerning. 
        
    # 5. How to choose the features to make the best out of the provided data?
        # Used Logistic Regression to select continuous features with more than 50% accuracy
        # Used Cramers'V to select categorical features with more than 50% accuracy as well

# =============================================================================
# =============================================================================
# MODEL DEVELOPMENT
    # Built machine learning model using both pipeline and GridSearchCV methods.
    # The model is first deployed with removing the 0 value in 'thall'
    # Achieved accuracy for both models were 76.9231%
    
    # The 0 values then has been imputed by using Simple Imputer with the feature's mode value
    # Then, the model has heightened it's accuracy to 80.2198%
    
    # Both the pipeline and GridSearchCV model had the same accuracy. 
    # Thus, pipeline model is chosen to be executed for the app deployment. 
# =============================================================================





