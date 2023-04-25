#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import layer,Dense,Dropout


# In[9]:


data=pd.read_csv("D:\\NMDS\\drug200.csv")
data.head()


# In[10]:


data.columns


# In[12]:


data.Drug


# In[13]:


data.isnull() 


# In[14]:


data.isnull().sum()


# In[16]:


data.info()


# In[18]:


import seaborn as sns 
sns.heatmap(data.corr(),annot=True)


# In[23]:


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(data['Age'])
plt.subplot(1,2,2)
sns.distplot(data['Na_to_K'])


# In[26]:


sns.pairplot(data=data,markers=["^","v"],palette="Inferno")


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_resamble,y_resamble,test_size=0.2,random_state=0)


# In[33]:


import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensample import RandomForestClassifier
rfr1=RandomForestClassifier().fit(x_os,y_os.values.ravel())
y_pred=rfr1.pred(x_test_os)


# In[34]:


from xgboost import XGBClassifier
xgb1=XGBClassifier()
xgb1.fit(x_os,y_os)


# In[35]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
sv=SVC
sv.fit(x_bal,y_bal)


# In[ ]:




