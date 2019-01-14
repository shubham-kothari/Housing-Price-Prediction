
# coding: utf-8

# # House Price Prediction
# ---------------------------
# 
# 
# This is a copy of UCI ML housing dataset.
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
# 
# 
# This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
# 
# The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
# prices and the demand for clean air', J. Environ. Economics & Management,
# vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
# ...', Wiley, 1980.   N.B. Various transformations are used in the table on
# pages 244-261 of the latter.
# 
# The Boston house-price data has been used in many machine learning papers that address regression
# problems.   
#      
# .. topic:: References
# 
#    - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
#    - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
# 
# 
# 

# **Data Set Characteristics:**  
# 
#     :Number of Instances: 506 
# 
#     :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
# 
#     :Attribute Information (in order):
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per $10,000
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#         - LSTAT    % lower status of the population
#         - MEDV     Median value of owner-occupied homes in $1000's
# 
#     :Missing Attribute Values: None
# 
#     :Creator: Harrison, D. and Rubinfeld, D.L.
# 

# # Libraries Import
# 

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting the Data
# 

# In[2]:

X = pd.read_csv('train.csv')


# In[3]:

X_test = pd.read_csv('test.csv')


# In[4]:

X.head()


# In[5]:

X.describe()


# In[6]:

X.info()


# # Exploratory Data Analysis of Dataset
# 
# Let's do some data visualization! I'll be using seaborn and matplotlib library.
# 
# 

# ** Creating a heatmap of correlation between all the columns of boston housing dataset

# In[7]:

plt.figure(figsize=(15,15))
sns.heatmap(X.corr(),annot=True,cmap='coolwarm')


# ** Creating a scatterplot between target and 'RM','PTRATIO','LSTAT' in the dataset to analyse the trend between them.

# In[8]:

sns.set_style('whitegrid')
for i in ['rm','ptratio','lstat']:
    sns.lmplot(i,'medv',data=X,aspect=1,height=10,fit_reg=True)


# ## Train Test Split
# 
# Now its time to split the data into a training set and a testing set!
# 

# In[9]:

X_train = X[['ID','crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
       'tax', 'ptratio', 'black', 'lstat']]
y_train = X['medv']


# In[11]:

X_test = X_test


# ## Training a Random Forest Regressor Model
# 
# 
# ** Import RandomForestRegressor**

# In[12]:

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(X_train,y_train)
prediction = RFR.predict(X_test)


# ** Creating a submission.csv file to submit in kaggle https://inclass.kaggle.com/c/boston-housing **

# In[15]:

submission = pd.DataFrame({"ID": X_test["ID"],"medv": prediction})


# In[16]:

submission.set_index('ID',inplace=True)


# In[17]:

submission.to_csv('submission.csv')

