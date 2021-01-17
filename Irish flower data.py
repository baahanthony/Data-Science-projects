#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


iris = sns.load_dataset('iris')


# In[5]:


iris.head()


# In[24]:


sns.pairplot(iris,hue='species',palette='Dark2')


# In[12]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)


# In[14]:


sns.distplot(iris['sepal_width'])


# In[16]:


sns.jointplot(x='sepal_width',y='petal_width',data=iris)


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=101)


# In[26]:


from sklearn.svm import SVC


# In[27]:


svc_model = SVC()


# In[28]:


svc_model.fit(X_train,y_train)


# In[29]:


predictions = svc_model.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[31]:


print(confusion_matrix(y_test,predictions))


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


from sklearn.model_selection import GridSearchCV


# In[34]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# In[35]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[36]:


grid_predictions = grid.predict(X_test)


# In[37]:


print(confusion_matrix(y_test,grid_predictions))


# In[38]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




