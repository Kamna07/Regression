#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# In[3]:


data = pd.read_csv(r'C:\Users\ony\Downloads\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression\Polynomial_Regression\Position_Salaries.csv')


# In[4]:


print(data)


# In[34]:


x = data.iloc[:, 1:2].values
y = data.iloc[:, 2:].values
plt.scatter(x,y)


# In[47]:


#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 0)
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
#x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# In[49]:


#reg = SVR(kernel = 'linear',degree = 1)
#pr = reg.fit(x_train,y_train)
#pred = reg.predict(x_test)
#print(reg.score(x_test,y_test))
#print(r2_score(y_test,pred))


# In[50]:


reg1 = SVR(kernel = 'rbf')
pr1 = reg1.fit(x,y)
#pred1 = pr1.predict(x_test)
#print(reg1.score(x_test,y_test))
#print(r2_score(y_test,pred1))


# In[51]:


plt.scatter(x,y)
plt.plot(x,pr1.predict(x))


# In[55]:


pred = sc_y.inverse_transform(reg1.predict(sc_x.transform([[6.5]])))
print(pred)

