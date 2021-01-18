#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv('College_Data',index_col = 0)


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[15]:


sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
import warnings
warnings.filterwarnings("ignore")


# In[22]:


sns.lmplot(x='Outstate',y='F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[23]:


df['Outstate'].plot(kind='hist')


# In[24]:


sns.set_style('darkgrid')
h = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
h = h.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# In[25]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[26]:


df[df['Grad.Rate']>100]


# In[28]:


df[df['Grad.Rate']<20]


# In[29]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[30]:


df[df['Grad.Rate'] > 100]


# In[31]:


sns.set_style('darkgrid')
y = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
y = y.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[32]:


from sklearn.cluster import KMeans


# In[33]:


kmeans = KMeans(n_clusters=2)


# In[34]:


kmeans.fit(df.drop('Private',axis=1))


# In[35]:


kmeans.cluster_centers_


# In[41]:


kmeans.labels_


# In[36]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[37]:


df['Cluster'] = df['Private'].apply(converter)


# In[38]:


df.head()


# In[40]:


df.tail()


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# In[ ]:




