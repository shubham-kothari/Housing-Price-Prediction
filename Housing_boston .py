
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

from sklearn.datasets import load_boston


# In[3]:

boston = load_boston()


# In[4]:

boston.keys()


# In[5]:

print(boston['DESCR'])


# In[6]:

print(boston['feature_names'])


# In[7]:

boston_df = pd.DataFrame(boston['data'],columns=boston['feature_names'])


# In[8]:

boston_df['target'] = boston['target']


# In[9]:

boston_df.describe()


# In[10]:

boston_df.info()


# # Exploratory Data Analysis of Dataset
# 
# Let's do some data visualization! I'll be using seaborn and matplotlib library.
# 
# 

# ** Creating a heatmap of correlation between all the columns of boston housing dataset

# In[11]:

plt.figure(figsize=(15,15))
sns.heatmap(boston_df.corr(),annot=True,cmap='coolwarm')


# ** Creating a scatterplot between target and 'RM','PTRATIO','LSTAT' in the dataset to analyse the trend between them.

# In[12]:

sns.set_style('whitegrid')
for i in ['RM','PTRATIO','LSTAT']:
    sns.lmplot(i,'target',data=boston_df,aspect=1,height=10,fit_reg=True)


# ## Train Test Split
# 
# Now its time to split the data into a training set and a testing set!
# 
# ** Using sklearn to split data into a training set and a testing set.**

# In[13]:

X = boston_df[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y = boston_df['target']


# In[14]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)


# ## Training a Linear Regression Model
# 
# 
# ** Import LinearRegression**

# In[15]:

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train,y_train)
prediction = lm.predict(X_test)


# ** Creating a scatterplot of the real test values versus the predicted values. **

# In[16]:

plt.figure(figsize=(10,7))
plt.scatter(y_test,prediction)
plt.ylabel("Prediction")


# ### Evaluating the Linear Regression Model
# 
# Evaluating the linear model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[17]:

from sklearn import metrics
d = {}


# In[18]:

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
d['LinearRegression'] = np.sqrt(metrics.mean_squared_error(y_test, prediction))


# ## Training a Decision Tree Regressor Model
# 
# 
# ** Import DecisionTreeRegressor**

# In[19]:

from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(X_train,y_train)
prediction = DTR.predict(X_test)


# ** Creating a scatterplot of the real test values versus the predicted values. **

# In[20]:

plt.scatter(y_test,prediction)
plt.ylabel("Prediction")


# ### Evaluating the Decision Tree Regressor Model
# 
# Evaluating the Decision Tree Regressor model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[21]:

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
d['DecisionTreeRegressor'] = np.sqrt(metrics.mean_squared_error(y_test, prediction))



# ## Training a Random Forest Regressor Model
# 
# 
# ** Import RandomForestRegressor**

# In[22]:

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(X_train,y_train)
prediction = RFR.predict(X_test)


# ** Creating a scatterplot of the real test values versus the predicted values. **

# In[23]:

plt.scatter(y_test,prediction)
plt.ylabel("Prediction")


# ### Evaluating the Random Forest Regressor Model
# 
# Evaluating the  Random Forest Regressor model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[24]:

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
d['RandomForestRegressor'] = np.sqrt(metrics.mean_squared_error(y_test, prediction))


# # Comparing the R.M.S.E ( Root Mean Square Error ) of different models.

# In[25]:

pd.DataFrame(data=list(d.values()),index=d.keys(),columns=['RMSE'])


# # Random Forest Regressor Model gave the best result !
# 
