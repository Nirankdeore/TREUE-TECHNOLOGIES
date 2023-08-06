#!/usr/bin/env python
# coding: utf-8

# In[317]:


import numpy as np
import pandas as pd


# In[318]:


Data=pd.read_csv("BHP.csv")


# In[319]:


Data.head()


# In[320]:


Data.shape


# In[321]:


Data.info()


# In[322]:


for column in Data.columns: 
    print(Data[column].value_counts())
    print("*"*20)


# In[323]:


Data.isna()


# In[324]:


Data.isna().sum()


# In[325]:


Data.drop(columns=['area_type','availability','society','balcony'],inplace=True)


# In[326]:


Data.describe()


# In[327]:


Data.info()


# In[328]:


#Here we fill the missing value
Data['location']=Data['location'].fillna('Sarjapur Road')


# In[329]:


Data['size'].value_counts()


# In[330]:


# Here we feeling missing value with 2 bhk as mode
Data['size']=Data['size'].fillna('2 BHK')


# In[331]:


Data['bath']=Data['bath'].fillna(Data['bath'].median())


# In[332]:


Data.info()


# In[333]:


Data['bhk']=Data['size'].str.split().str.get(0).astype(int)


# In[334]:


Data[Data.bhk>20]


# In[335]:


def convertRange(x):
    y=x.split('-')
    if len(y)==2:
        return(float(y[0])+float(y[1]))/2
    try:
        return float(x)
    except:
        return None


# In[336]:


Data['total_sqft']=Data['total_sqft'].apply(convertRange)


# In[337]:


Data.head()


# In[338]:


# Price Per Square Feet
Data['price_per_sqft']=Data['price']*1000000/Data['total_sqft']


# In[339]:


Data['price_per_sqft']


# In[340]:


Data.describe()


# In[341]:


Data['location'].value_counts()


# In[342]:


Data['location']=Data['location'].apply(lambda x:x.strip())
location_count=Data['location'].value_counts()


# In[343]:


location_count_less_10=location_count[location_count<=10]
location_count_less_10


# In[344]:


Data['location']=Data['location'].apply(lambda x:'other' if x in location_count_less_10 else x)


# In[345]:


Data['location'].value_counts()


# In[ ]:





# Outlier detection and removal

# In[346]:


Data.describe()


# In[347]:


(Data['total_sqft']/Data['bhk']).describe()


# In[348]:


(Data['total_sqft']/Data['bhk']).describe()


# In[349]:


Data=Data[((Data['total_sqft']/Data['bhk'])>=300)]


# In[350]:


Data.shape


# In[351]:


Data.price_per_sqft.describe()


# In[352]:


def remove_outliers_sqft(df):
    df_output=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        
        gen_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_output=pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
Data=remove_outliers_sqft(Data)


# In[353]:


Data.describe()


# In[357]:


def bhk_outlier_remover(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
    
    for bhk,bhk_df in location_df.groupby('bhk'):
        stats=bhk_stats.get(bhk-1)
        if stats and stats['count']>5:
            exclusive_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return Data.drop(exclude_indices,axis='index')


# In[358]:


Data=bhk_outlier_remover(Data)


# In[359]:


Data.shape


# In[360]:


Data


# In[361]:


Data.drop(columns=['size','price_per_sqft'],inplace=True)


# In[ ]:





# In[362]:


# Clean Data


# In[363]:


Data.head()


# In[364]:


Data.to_csv('BHK_Data.csv')


# In[365]:


X=Data.drop(columns=['price'])
y=Data['price']


# In[366]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[367]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[368]:


print(X_train.shape)
print(X_test.shape)


# In[369]:


#Applying Linear Regression


# In[370]:


column_trans=make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')


# In[371]:


scaler=StandardScaler()


# In[372]:


lr=LinearRegression(normalize=True)


# In[373]:


pipe=make_pipeline(column_trans,scaler,lr)


# In[374]:


pipe.fit(X_train,y_train)


# In[375]:


y_pred_lr=pipe.predict(X_test)


# In[377]:


print(y_pred_lr)
print(y_test)


# In[380]:


print('Coefficient of determination:',r2_score(y_test,y_pred_lr))

