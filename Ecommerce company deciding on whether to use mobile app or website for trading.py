#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[3]:


Ecom = pd.read_csv('Ecommerce Customers')


# In[4]:


Ecom.head()


# In[5]:


Ecom.info()


# In[7]:


Ecom.describe()


# In[8]:


Ecom.columns


# In[9]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data = Ecom)


# In[10]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data = Ecom)


# In[11]:


sns.jointplot(x='Time on App',y='Length of Membership',data = Ecom,kind='hex')


# In[12]:


sns.pairplot(Ecom)


# In[13]:


sns.distplot(Ecom['Yearly Amount Spent'])


# In[16]:


sns.lmplot(y='Yearly Amount Spent',x='Length of Membership',data=Ecom)


# In[17]:


Ecom.columns


# In[19]:


x=Ecom[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[20]:


y=Ecom['Yearly Amount Spent']


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


train_test_split


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[24]:


lm = LinearRegression()


# In[25]:


lm.fit(x_train,y_train)


# In[26]:


lm.coef_


# In[27]:


predictions = lm.predict(x_test)


# In[28]:


predictions


# In[29]:


plt.scatter(y_test,predictions)


# In[30]:


from sklearn import metrics


# In[31]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[32]:


sns.distplot((y_test-predictions))


# In[33]:


plt.hist((y_test-predictions))


# In[34]:


cdf = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])


# In[35]:


cdf


# In[ ]:




