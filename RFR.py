#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data = pd.read_csv(r'C:\Users\ony\Downloads\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 9 - Random Forest Regression\Random_Forest_Regression\Position_Salaries.csv')


# In[3]:


print(data)
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2:].values


# In[11]:


reg = RandomForestRegressor(n_estimators=300,random_state =0)
a = reg.fit(x,y)


# In[12]:


x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,reg.predict(x_grid))
plt.show()


# In[ ]:




