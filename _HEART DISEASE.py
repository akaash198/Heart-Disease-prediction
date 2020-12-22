#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # READING DATA

# In[2]:


data = pd.read_csv('heart_disease_dataset.csv')
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values


# In[3]:


data.head()


# # EDA

# In[4]:


sns.countplot(data.num)
plt.show()
print(data.num.value_counts())


# # DATA WRANGLING

# In[5]:


data.isnull().sum()


# In[6]:


sns.heatmap(data.isnull())


# In[7]:


data.describe()


# In[8]:


data.hist(figsize=(20,20))
plt.show()


# In[9]:


cor = data.corr()

mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))

with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[13]:


feature_importance = classifier.feature_importances_
feat_importances = pd.Series(classifier.feature_importances_, index=data.columns[:-1])
feat_importances = feat_importances.nlargest(13)

feature = data.columns.values.tolist()[0:-1]
importance = sorted(classifier.feature_importances_.tolist())


x_pos = [i for i, _ in enumerate(feature)]

plt.barh(x_pos, importance , color='dodgerblue')
plt.ylabel("feature")
plt.xlabel("importance")
plt.title("feature_importances")

plt.yticks(x_pos, feature)

plt.show()


# In[14]:


y_pred = classifier.predict(X_test)


# In[15]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[16]:


cm

