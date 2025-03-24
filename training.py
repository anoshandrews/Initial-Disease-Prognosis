#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# In[3]:


df = pd.read_csv('Training.csv')


# In[4]:


df.head()


# In[5]:


df.head(10)


# In[6]:


df.size


# In[56]:


columns = df.columns.to_list()
columns


# In[54]:


df.columns


# In[10]:


df = df.drop(columns = 'Unnamed: 133')


# In[12]:


diseases_count = df['prognosis'].value_counts()
diseases_count


# In[14]:


temp_df = pd.DataFrame()
temp_df['Disease'] = diseases_count.index
temp_df['Count'] = diseases_count.values


# In[16]:


temp_df.head()


# In[21]:


plt.figure(figsize = (18,8))
sns.barplot(x = 'Disease', y = 'Count', data = temp_df)
plt.xticks(rotation =90)


# In[22]:


encoder = LabelEncoder()
df['prognosis'] = encoder.fit_transform(df['prognosis'])


# In[23]:


df['prognosis']


# In[30]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[36]:


X_train.shape,y_train.shape


# In[38]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state = 18)


# In[39]:


model.fit(X_train,y_train)


# In[42]:


X_test.shape, y_test.shape


# In[43]:


prediction = model.predict(X_test)


# In[44]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[45]:


accuracy = accuracy_score(prediction, y_test)


# In[46]:


accuracy


# In[50]:


plt.figure(figsize = (10,6))
c_matrix = confusion_matrix(y_test, prediction)
sns.heatmap(c_matrix, annot = True,)


# In[51]:


import joblib

joblib.dump(model,'disease_prediction.pkl')


# In[ ]:




