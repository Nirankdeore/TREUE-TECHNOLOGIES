#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('spam.csv',encoding='ISO-8859-1')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


# Data cleaning


# In[7]:


# drop last 3 columns
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


data.head()


# In[9]:


# renaming the columns
data.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[10]:


data.head()


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[12]:


data['target']=encoder.fit_transform(data['target'])


# In[13]:


data.head()


# In[14]:


# missing values
data.isnull().sum()


# In[15]:


# check for duplicate values
data.duplicated().sum()


# In[16]:


# remove duplicates
data=data.drop_duplicates(keep='first')


# In[17]:


data.duplicated().sum()


# In[18]:


data.shape


# In[19]:


# Exploretry data Analysis


# In[20]:


data['target'].value_counts()


# In[21]:


import matplotlib.pyplot as plt
colors = ['#DD7596', '#8EB897']
plt.pie(data['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f',labeldistance=1.15, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },colors=colors)
plt.show()


# In[22]:


# Data is inbalaced


# In[23]:


import nltk


# In[24]:


get_ipython().system('pip install nltk')


# In[25]:


nltk.download('punkt')


# In[26]:


data['num_characters']=data['text'].apply(len)


# In[27]:


data.head()


# In[28]:


#num of words
data['num_words']=data['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[29]:


data['num_sentences']=data['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


data.head()


# In[31]:


data[['num_characters','num_words','num_sentences']].describe()


# In[32]:


#ham
data[data['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[33]:


#spam
data[data['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[34]:


import seaborn as sns


# In[35]:


sns.histplot(data[data['target']==0]['num_characters'],color='green')
sns.histplot(data[data['target']==1]['num_characters'],color='red')


# In[36]:


sns.histplot(data[data['target']==0]['num_words'],color='Green')
sns.histplot(data[data['target']==1]['num_words'],color='red')


# In[ ]:





# In[37]:


sns.heatmap(data.corr(),annot=True)


# In[38]:


# Data Preprocessing
#lower case, Tokenization, Removing special characters, removing stop words and punchuatiom, Stemming


# In[39]:


import string
string.punctuation


# In[40]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[41]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('liking')


# In[42]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


# In[43]:


transform_text('Okay name ur price as long as its legal! Wen can I pick them up? Y u ave x ams xx')


# In[44]:


data['text'][100]


# In[45]:


data['transformed_text']=data['text'].apply(transform_text)


# In[46]:


data.head()


# In[47]:



#wc=WordCloud(width=60,height=60,min_font_size=10,background_color='Black')


# In[48]:


spam_corpus=[]
for msg in data[data['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[49]:


len(spam_corpus)


# In[50]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(10))[0],pd.DataFrame(Counter(spam_corpus).most_common(10))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[51]:


ham_corpus=[]
for msg in data[data['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[52]:


len(ham_corpus)


# In[53]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(10))[0],pd.DataFrame(Counter(ham_corpus).most_common(10))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


# Model Building


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[56]:


X=tfidf.fit_transform(data['transformed_text']).toarray()


# In[57]:


X.shape


# In[58]:


X


# In[59]:


y=data['target'].values


# In[60]:


y


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[63]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[64]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[69]:


mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print("accuracy_score:-",accuracy_score(y_test,y_pred2))
print("confusion_matrix:-",confusion_matrix(y_test,y_pred2))
print("precision_score:-",precision_score(y_test,y_pred2))


# In[70]:


bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print("accuracy_score:-",accuracy_score(y_test,y_pred3))
print("confusion_matrix:-",confusion_matrix(y_test,y_pred3))
print("precision_score:-",precision_score(y_test,y_pred3))


# In[74]:


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print("accuracy_score:-",accuracy_score(y_test,y_pred1))
print("confusion_matrix:-",confusion_matrix(y_test,y_pred1))
print("precision_score:-",precision_score(y_test,y_pred1))

