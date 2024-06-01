#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[32]:


df = pd.read_csv('heart.csv')
df = df.drop_duplicates()


# -1 hingga 1
# 

# In[33]:


df.head()


# In[34]:


df.info()


# In[4]:


df['oldpeak'].value_counts().reset_index().sort_values('index')


# In[5]:


df.describe()


# In[19]:


plot_target = df['target'].value_counts().reset_index()
plot_target


# In[20]:


sns.set_theme(palette = 'coolwarm', style = 'whitegrid')
sns.barplot(plot_target, x = 'target', y = 'count', palette = ['yellow', 'red'])
plt.title('Heart Disease Distribution', fontweight = 'bold', fontsize = 14)


# In[22]:


df.head()


# In[25]:


sns.countplot(data = df, x = 'sex')
plt.title('Sex')


# In[27]:


for i in df.columns:
    sns.countplot(x = df[i])
    plt.title(i)
    plt.show()


# In[29]:


for i in df.columns:
    sns.boxplot(x = df[i])
    plt.title(i)
    plt.show()


# In[8]:


for i in df.columns:
    sns.histplot(df[i])
    plt.title(i)
    plt.show()


# In[40]:


plt.figure(figsize = (15,15))
mask = np.triu(np.ones_like(df.corr(method='pearson')), k=1)
sns.heatmap(df.corr(method = 'pearson').apply(lambda x: round(x,2)), annot = True, mask = mask)
plt.title('Pearson', fontweight = 'bold')
plt.show()

plt.figure(figsize = (15,15))
mask = np.triu(np.ones_like(df.corr(method='spearman')), k=1)
sns.heatmap(df.corr(method = 'spearman').apply(lambda x: round(x,2)), annot = True, mask = mask)
plt.title('spearman', fontweight = 'bold')
plt.show()

plt.figure(figsize = (15,15))
mask = np.triu(np.ones_like(df.corr(method='kendall')), k=1)
sns.heatmap(df.corr(method = 'kendall').apply(lambda x: round(x,2)), annot = True, mask = mask)
plt.title('kendall', fontweight = 'bold')
plt.show()


# age kita bagi persentil 4
# 
# trestps bagi 4
# < 120
# 120 - 129 1
# 130 - 139 2
# 140 - 180 3
# 180 > 4
# 
# chols
# < 200 0 
# 200 - 239 1
# > 239 2
# 
# thalac
# thalach*100%/(220 - age) 
# 50-60 kat 0
# 60-70 kat 1
# 70-80 kat 2
# 80-90 kat 3
# 90-100 kat 4
# 
# oldpeak
# 0 = 0
# >0 abnormal 1

# In[ ]:





# In[49]:


df['age_conv'] = df['age'].apply(lambda x: 
                                 0 if x < np.percentile(df['age'], q = 25) else(
                                 1 if np.percentile(df['age'], q = 25) <= x < np.percentile(df['age'], q = 50) else(
                                 2 if np.percentile(df['age'], q = 50) <= x < np.percentile(df['age'], q = 75) else(3)))
                                )

df['trestbps_conv'] = df['trestbps'].apply(lambda x:
                                          0 if x < 120 else(
                                          1 if 120 <= x < 130 else(
                                          2 if 130 <= x < 140 else(
                                          3 if 140 <= x < 180 else
                                          4)))
                                          )

df['oldpeak_conv'] = df['oldpeak'].apply(lambda x:
                                          0 if x == 0 else 1)
                                        
                                          

df['chol_conv'] = df['chol'].apply(lambda x:
                                          0 if x < 200 else(
                                          1 if 200 <= x < 239 else(2)
                                          )
                                          )

df['thalach_conv'] = df[['thalach', 'age']].apply(lambda x: 
                                           0 if (x[0]*100)/(220 - x[1]) < 60 else(
                                           1 if 60 <= (x[0]*100)/(220 - x[1]) < 70 else(
                                           2 if 70 <= (x[0]*100)/(220 - x[1]) < 80 else(
                                           3 if 80 <= (x[0]*100)/(220 - x[1]) < 90 else(
                                           4 if (x[0]*100)/(220 - x[1]) >=90 else 9))))
                                           ,axis = 1)


# In[77]:


df.head()


# In[50]:


df2 = df.drop(columns = ['age','trestbps', 'chol', 'thalach', 'oldpeak'])


# In[51]:


df2


# In[9]:


from sklearn.model_selection import train_test_split


# In[58]:


X = df2.drop(columns = 'target')
y = df2['target']

X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42, stratify = y)
X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)


# In[68]:


import catboost as cb
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# In[72]:


cat_model = cb.CatBoostClassifier()
cat_model.fit(X_train, y_train)
y_predict_train = cat_model.predict(X_train)
print('-----train')
print(classification_report(y_train, y_predict_train))
y_predict_test = cat_model.predict(X_test)
print('CatBoost')
print(classification_report(y_test, y_predict_test))

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_predict_test)

# Membuat heatmap confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('CatBoost Confusion Matrix')
plt.show()


# In[73]:


xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_predict_train = xgb_model.predict(X_train)
print('-----train')
print(classification_report(y_train, y_predict_train))
y_predict_test = xgb_model.predict(X_test)
print('XGBoost')
print(classification_report(y_test, y_predict_test))

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_predict_test)

# Membuat heatmap confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('XGBoost Confusion Matrix')
plt.show()


# In[74]:


rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)
y_predict_train = rfc_model.predict(X_train)
print('-----train')
print(classification_report(y_train, y_predict_train))
y_predict_test = rfc_model.predict(X_test)
print('Random Forest')
print(classification_report(y_test, y_predict_test))

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_predict_test)

# Membuat heatmap confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Random Forest Confusion Matrix')
plt.show()


# In[75]:


# Inisialisasi dan melatih model SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Memprediksi nilai target untuk data uji
y_pred = svm_model.predict(X_test)

# Mencetak laporan klasifikasi untuk data latih dan data uji
print('-----train')
print(classification_report(y_train, svm_model.predict(X_train)))
print('SVM')
print(classification_report(y_test, y_pred))

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Membuat heatmap confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('SVM Confusion Matrix')
plt.show()


# In[60]:


y_test


# In[76]:


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_predict_train = lr_model.predict(X_train)
print('-----train')
print(classification_report(y_train, y_predict_train))
y_predict_test = lr_model.predict(X_test)
print('Logistic Regression')
print(classification_report(y_test, y_predict_test))

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_predict_test)

# Membuat heatmap confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# In[ ]:




