#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('Churn_Modelling.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull()


# In[8]:


data.isnull().sum()


# In[9]:


data.columns


# In[10]:


data=data.drop(['RowNumber','CustomerId','Surname'],axis=1)


# In[11]:


data.head()


# In[12]:


# Encoding Categorical Data


# In[13]:


data=pd.get_dummies(data,drop_first=True)


# In[14]:


data.head()


# In[15]:


data['Exited'].value_counts()


# In[16]:


import seaborn as sns


# In[17]:


colors = ['green','red']
sns.countplot(data['Exited'])


# In[18]:


X= data.drop('Exited',axis=1)
y=data['Exited']


# In[19]:


print(y)


# In[20]:


# Here we using SMOTE to handle imbalanced Dataset


# In[21]:


get_ipython().system('pip install imbalanced-learn')


# In[22]:


from imblearn.over_sampling import SMOTE


# In[23]:


X_res,y_res=SMOTE().fit_resample(X,y)


# In[24]:


y_res.value_counts()


# In[25]:


# Split the dataset into training and testing


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=44)


# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


sc=StandardScaler()


# In[30]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:





# In[31]:


# Applying Different machine learning model


# In[32]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score


# In[33]:


log=LogisticRegression()


# In[34]:


log.fit(X_train,y_train)


# In[35]:


y_pred1=log.predict(X_test)


# In[36]:


accuracy_score(y_test,y_pred1)


# In[37]:


precision_score(y_test,y_pred1)


# In[38]:


recall_score(y_test,y_pred1)


# In[39]:


f1_score(y_test,y_pred1)


# In[ ]:





# In[40]:


#Support Vector Classifier
from sklearn import svm


# In[41]:


svm=svm.SVC()


# In[42]:


svm.fit(X_train,y_train)


# In[43]:


y_pred2=svm.predict(X_test)


# In[44]:


accuracy_score(y_test,y_pred2)


# In[45]:


precision_score(y_test,y_pred2)


# In[ ]:





# In[60]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier


# In[61]:


gbc=GradientBoostingClassifier()


# In[62]:


gbc.fit(X_train,y_train)


# In[63]:


y_pred3=gbc.predict(X_test)


# In[64]:


accuracy_score(y_test,y_pred3)


# In[66]:


precision_score(y_test,y_pred3)


# In[ ]:


# On the basis of accuracy and precision score we select svm which is goog suited with this dataset


# In[68]:


# Final model
X_res=sc.fit_transform(X_res)


# In[69]:


svm.fit(X_res,y_res)


# In[70]:


import joblib


# In[71]:


joblib.dump(svm,'Bank_customer_churn_prediction_model')


# In[72]:


model=joblib.load('Bank_customer_churn_prediction_model')


# In[73]:


data.columns


# In[77]:


model.predict([[608,41,1,83807.86,1,0,1,112542.58,0,0,0]])


# In[ ]:




