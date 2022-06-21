
<a><img alt='kg' src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white"></a>
<a><img alt='kg' src="https://img.shields.io/badge/Spyder%20Ide-FF0000?style=for-the-badge&logo=spyder%20ide&logoColor=white"></a>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
![info](https://user-images.githubusercontent.com/105897390/174782470-20d6e2e2-454b-4199-8e36-e1fd187df16a.jpg)


# Heart_Attack_Prediction_Using_Machine_Learning
 Trained machine learning model to solve the heart attack classification problem using selected features. 


According to World Health Organisation (WHO), every year around 17.9 million 
deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming 
the leading cause of death globally. CVDs are a group of disorders of the heart 
and blood vessels, if left untreated it may cause heart attack. Heart attack occurs 
due to the presence of obstruction of blood flow into the heart. The presence of 
blockage may be due to the accumulation of fat, cholesterol, and other substances. 
Despite treatment has improved over the years and most CVD’s pathophysiology 
have been elucidated, heart attack can still be fatal. 
Thus, clinicians believe that prevention of heart attack is always better than curing 
it. After many years of research, scientists and clinicians discovered that, the 
probability of one’s getting heart attack can be determined by analysing the
patient’s age, gender, exercise induced angina, number of major vessels, chest 
pain indication, resting blood pressure, cholesterol level, fasting blood sugar, 
resting electrocardiographic results, and maximum heart rate achieved.


## Dataset Link 
Data is obtained from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/329925

## Additional Info 
Additional informations were obtained from these websites:
1. http://archive.ics.uci.edu/ml/datasets/Heart+Disease
2. https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/329925

## Discussions


### Problem Statement
Given medical dataset, can we predict the chance of a person having heart attack?

### Exploratory Data Analysis
Questions:
     1. What kind of data are we dealing with?
         Can refer the to Additional Link info attached stated above.
         our target variable is the 'output' variable

     2. Do we have missing values?
        Variable thal had 0 as its value which represents null value. 
        Hence, had it replaced with NaN to be imputed using mode of the particular variable        
     
     3. Do we have duplicated datas?
        We had 1 and had it removed. 
        
     4. Do we have extreme values?
        The data didnt possess any negative values which is logical.
        However, chol variable do possess the maximum value up to 564 which is quite concerning.  
        
     5. How to choose the features to make the best out of the provided data?
        Used Logistic Regression to select continuous features with more than 50% accuracy
        Used Cramers'V to select categorical features with more than 50% accuracy as well

![boxplot_heart](https://user-images.githubusercontent.com/105897390/174783642-8982636d-4a8f-4a23-86ea-070a3cbd5df6.png)
![output](https://user-images.githubusercontent.com/105897390/174786416-25cd8162-a380-4287-8780-540f8331ba28.png)
The target variable namely output depicts that we have 165 person with heart disease and 138 person without heart disease, so our problem is considered balance.


![msno](https://user-images.githubusercontent.com/105897390/174786918-90b60b39-ebfd-42a3-8c91-cf264c8ba74c.png)
And no null values! :)

### MODEL DEVELOPMENT
     Built machine learning model using both pipeline and GridSearchCV methods.
     The model is first deployed with removing the 0 value in 'thall'
     Achieved accuracy for both models were 76.9231%
    
     The 0 values then has been imputed by using Simple Imputer with the feature's mode value
     Then, the model has heightened it's accuracy to 80.2198%



     Both the pipeline and GridSearchCV model had the same accuracy. 
     Thus, pipeline model is chosen to be executed for the app deployment. 
![classification_report](https://user-images.githubusercontent.com/105897390/174783216-cf0f4aaf-e325-4139-8d59-a5419da339c1.png)



