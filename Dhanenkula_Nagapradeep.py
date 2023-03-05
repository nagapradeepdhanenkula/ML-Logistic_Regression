#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[48]:


#Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix


# In[49]:


# Reading the Dataset into Pandas


# In[50]:


df = pd.read_csv('winequality-red.csv')
df.head()


# In[51]:


#To check No. of Rows and Columns


# In[52]:


df.shape


# In[53]:


#Check for Null Values


# In[54]:


df.isnull().values.any()


# In[55]:


#Summarized Descriptive Statistics
df.describe()


# In[56]:


##Correlation Matrix using Heatmap


# In[57]:


fig = plt.figure(figsize = [10,7])
sns.heatmap(df.corr(),annot = True, cmap = 'RdBu_r', center = 0)
plt.title("Correlation Matrix")
plt.show()


# In[90]:


def correlation(dataset, threshold):
    correlated_Col = set()
    correlated_matrix = dataset.corr()
    for n in range(len(correlated_matrix.columns)):
        for p in range(n):
            if abs(correlated_matrix.iloc[n, p]) > threshold:
                column_name = correlated_matrix.columns[n] 
                correlated_Col.add(column_name)
    return correlated_Col


# In[94]:


correlated_features = correlation(X_train, 0.6)
len(set(correlated_features))


# In[ ]:


# Top four Features


# In[97]:


print(correlated_features)


# In[ ]:


#Relation b/w the attributes vs Quality


# In[ ]:


sns.pairplot(df,hue='quality')
plt.figure(figsize =(12,8)) 


# In[ ]:


#Checking the Quality-wine with a condition


# In[60]:


df['quality'] = df.quality.apply(lambda x : 'High' if x > 6.5 else 'Low')
df.head()


# In[ ]:


#% of Quality_wines


# In[61]:


df['quality'].value_counts()


# In[ ]:


#Bar plots for Quality Variable


# In[62]:


sns.countplot(data = df,x='quality')
plt.title('Count vs Quality')
plt.show()


# In[ ]:


#Assigning the Dependent and independent variables


# In[67]:


X = df.drop('quality', axis=1)
y = df['quality']
print(X.head())
y.head()


# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

#Independent variables on a normal scale

scaler = StandardScaler()
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
X_train[columns] = scaler.fit_transform(X_train[columns])
X_test[columns] = scaler.fit_transform(X_test[columns])
X_train.head()


# In[ ]:


#Training the Logistic regression model


# In[78]:


log_reg = LogisticRegression(class_weight='balanced') #Model Hyperparameter
log_reg.fit(X_train, y_train)

# predicting the Dependent variable 
y_preds = log_reg.predict(X_test)

#Calculating the accuracy of the model
print(accuracy_score(y_test, y_preds))


# In[ ]:


#To quantify the logistic regression model


# In[86]:


cf_matrix=confusion_matrix(y_test, y_preds)
print(cf_matrix)


# In[104]:


#Visualizing the Confusion Matrix
Cf_m = sns.heatmap(cf_matrix, annot=True, cmap='coolwarm',fmt='g')

Cf_m.set_title('Seaborn Confusion Matrix \n\n');
Cf_m.set_xlabel('\nActual Values')
Cf_m.set_ylabel('predicted Values ');
Cf_m.xaxis.set_ticklabels(['positive','negative'])
Cf_m.yaxis.set_ticklabels(['positive','negative'])
plt.show()


# In[ ]:




