#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


titanic_data = pd.read_csv('C:/Users/omkol/Desktop/titanic/train.csv')


# In[5]:


titanic_data.head()


# In[6]:


titanic_data.shape


# In[7]:


titanic_data.info()


# In[8]:


titanic_data.isnull().sum()


# In[9]:



titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[10]:


titanic_data['Age'].fillna(titanic_data['Age'].mean, inplace=True)


# In[11]:


print (titanic_data['Embarked'].mode())


# In[12]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[13]:


titanic_data.isnull().sum()


# In[14]:


titanic_data.describe()


# In[15]:


titanic_data['Survived'].value_counts()


# In[16]:


sns.set()


# In[17]:


sns.countplot('Survived', data=titanic_data)


# In[18]:


sns.countplot('Sex', data=titanic_data)


# In[19]:


sns.countplot('Sex', data=titanic_data)


# In[20]:


sns.countplot('Sex', hue = 'Survived', data=titanic_data)


# In[21]:


titanic_data['Pclass'].value_counts()


# In[22]:


sns.countplot('Pclass', data=titanic_data)


# In[23]:


sns.countplot('Pclass', hue = 'Survived', data=titanic_data)


# In[24]:


titanic_data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[25]:


titanic_data.head()


# In[27]:


X = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[29]:


X.shape, X_train.shape, X_test.shape


# In[37]:


print(X_train['Age'].dtypes)


# In[51]:


X_train['Age'] = pd.to_numeric(X_train['Age'].fillna(-1), errors='coerce')
X_test['Age'] = pd.to_numeric(X_train['Age'].fillna(-1), errors='coerce')


# In[52]:


model = LogisticRegression()


# In[53]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X_train['Fare'] = label_encoder.fit_transform(X_train['Fare'])
X_train['Age'] = label_encoder.fit_transform(X_train['Age'])
X_test['Fare'] = label_encoder.fit_transform(X_test['Fare'])
X_test['Age'] = label_encoder.fit_transform(X_test['Age'])



# In[56]:


model.fit(X_train, Y_train)


# In[57]:


X_train_prediction = model.predict(X_train)


# # 0 = SURVIVED  and 1 = NOT - SURVIVED
# 

# In[44]:


print(X_train_prediction) # 0 means SURVIVED and 1 means NOT SURVIVED


# In[58]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy Score of Training data:", training_data_accuracy)


# In[59]:


X_test_prediction = model.predict(X_test)


# In[60]:


print(X_test_prediction)


# In[61]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy Score of Test data:", test_data_accuracy)


# In[ ]:





# In[ ]:




