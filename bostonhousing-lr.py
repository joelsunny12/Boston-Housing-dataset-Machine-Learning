#!/usr/bin/env python
# coding: utf-8

# In[14]:


import math

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as stats

import statsmodels.api as sm
import numpy as np

from sklearn import preprocessing


# In[15]:


df = pd.read_csv('../Downloads/BostonHousing.csv')
df


# In[16]:


df.corr(method='pearson')


# In[17]:


# scikit learn
independent_variables = df.drop('MEDV', axis=1)

x = independent_variables.values
y = df['MEDV'].values
x_scaled = preprocessing.scale(x)
lr = LinearRegression(fit_intercept = True)
lr.fit(x_scaled, y)
y_pred = lr.predict(x_scaled)


# In[18]:


print('Coefficients = ', lr.coef_)


# In[19]:


print('Intercept = ', lr.intercept_)


# In[20]:


print('R^2 = ', lr.score(x_scaled, y))


# In[21]:


print('Root MSE = ', math.sqrt(metrics.mean_squared_error(y, y_pred)))


# In[22]:


x = independent_variables
y = df['MEDV']

x2 = sm.add_constant(x_scaled)
ols = sm.OLS(y, x2)
est = ols.fit()

est.summary()


# In[23]:


#The model after normalisation has an R^2 value of 0.7406426641094095
#F-Statistic is 108.1
#RMSE = 4.679191295697281

