#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[49]:


df = pd.read_csv('train.csv')
df.head()


# In[50]:


df.info()


# In[51]:


df.describe()


# # EDA (Exploratory Data Analysis)

# In[52]:


gender_survived = df.groupby(['Sex','Survived'])['PassengerId'].count().reset_index().rename(columns = {'PassengerId':'count'})
gender_survived


# In[53]:


sns.set_theme(palette = 'cubehelix')
sns.barplot(data = gender_survived, x = 'Survived', y = 'count', hue = 'Sex')
plt.title('Survived People by their Sex', fontweight = 'bold', fontsize = 14)


# In[54]:


data_tabel = df.groupby(['Pclass','Sex','Embarked'])['Survived'].value_counts()
data_tabel


# In[55]:


### 70% gaada bisa di apus aja datanya
df_info = df.count().reset_index().rename(columns = {'index':'columns',0:'count'})
df_info['Percentage (%)']= df_info['count']*100/df_info['count'].max()
df_info


# dari persebaran kelengkapan data, data cabin hanya memilki 22,8% kelengkapan data, bisa dipertimbangin untuk di hapus
# Age dan embarked walaupun ada yang hilang, masih dapat dipertimbangkan untuk digunakan
# 
# ### Handling Missing Values
# -data cabin di handle nya dengan cara di drop
# -data embarked bisa isi asal median
# -data age bisa diisi media/percentile/KNN
# 
# EDA itu ada 2, deskriptif dan visual

# In[56]:


df.describe(percentiles = [.25,.5,.75,.8,.9])


# bisa di asumsikan 75% orang yang naik kapal titanic meninggal
# 50% orang dengan class 3
# umur maksimal 80 tahun, minimal 0,42 tahun, rata rata 29 tahun

# ### plotting
# -numerikal bagusnya mengunakan histogram
# 
# -kategorikal bagusnya menggunakan barplot
# 
# -numerikal : umur, fare
# 
# -kategorikal : gender, pclass, parch, sibsp, survived, embarked

# In[57]:


sns.set_style('white')
sns.histplot(df['Age'], color = 'blue')
plt.title('Data Distribution of Age', fontweight = 'bold', fontsize = 14)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, f'{round(p.get_height(), 2)}', ha='center')


# In[58]:


sex_count = df['Sex'].value_counts().reset_index()
print(sex_count)
sns.barplot(data = sex_count, x = 'Sex', y = 'count', hue = 'Sex', palette = ['red', 'blue'], dodge = False)


# In[59]:


for i in ['Survived', 'Sex', 'Pclass', 'Embarked']:
    table = df[i].value_counts().reset_index()
    print(table)
    sns.barplot(data = table, x = i, y= 'count', hue= i, dodge = False, palette = ['blue', 'red', 'orange'])
    plt.title('Data Distribution of {}'.format(i), fontweight = 'bold')
    plt.show()


# In[60]:


embarked_survived = df.groupby('Embarked').agg(count = ('Survived', 'value_counts')).reset_index()
display(embarked_survived.sort_values(['Embarked', 'Survived']))
sns.barplot(data = embarked_survived, x = 'Embarked', y = 'count', hue= 'Survived')
plt.title('Survived People by their Embarked', fontweight= 'bold', fontsize = 14)
plt.show()


# In[61]:


embarked_survived1 = df.groupby('Embarked')['Survived'].value_counts().unstack()
embarked_survived1['percentage_survived'] = (embarked_survived1[1] / (embarked_survived1[1] + embarked_survived1[0]))*100
embarked_survived1


# In[62]:


Pclass_survived = df.groupby('Pclass')['Survived'].value_counts().reset_index()
Pclass_survived
# Pclass_survived.sort_values(['Pclass', 'Survived'])
# Pclass_survived['precentage_survived'] = (Pclass_survived[1] / (Pclass_survived[0] + Pclass_survived[1]))* 100
# Pclass_survived


# In[63]:


sns.barplot(data = Pclass_survived, x='Pclass', y = 'count', hue = 'Survived', palette = ['blue', 'red'])
plt.title('Survived People by their Pclass', fontweight= 'bold', fontsize = 14)


# In[64]:


fig, ax = plt.subplots(2,1, figsize = (8, 10))
ax[0].hist(df['SibSp'], color = 'b')
ax[0].set_title('Data Distribution of SibSp')
ax[0].set_ylabel('count')
ax[1].hist(df['Parch'], color = 'r')
ax[1].set_title('Data Distribution of Parch')
ax[1].set_ylabel('count')

plt.xlabel('Number of Siblings/Spouses (SibSp) and Parents/Children (Parch)')
plt.tight_layout()
plt.show()


# In[65]:


sns.histplot(df['Parch'], discrete = True, color = 'red')
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, f'{(p.get_height())}', ha='center')


plt.title ('Data Distribution of Parch', fontweight = 'bold')
plt.xticks(range(0,7))
plt.show()


# In[66]:


sns.histplot(df['SibSp'], discrete = True, color = 'blue')
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, f'{(p.get_height())}', ha='center')


plt.title ('Data Distribution of SibSP', fontweight = 'bold')
plt.xticks(range(0,9))
plt.show()


# In[67]:


df.loc[df['Age'].isna(), 'Age'] = np.mean(df['Age'])
df.loc[df['Embarked'].isna(), 'Embarked'] = 'S'
df_use = df.drop(columns = ['Cabin', 'Name', 'PassengerId', 'Ticket'])
df_use


# In[68]:


df_use.info()


# In[69]:


df_use['Sex_conv'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)
df_use['Embarked_conv'] = df['Embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'S' else 2))


# In[70]:


df_use.head()


# In[71]:


def pf (x):
    return np.percentile(df_use['Fare'], x)
def Fare_convert (x):
    if x < pf(25):
        return 0
    elif pf(25) <= x < pf(50):
        return 1
    elif pf(50) <= x < pf(75):
        return 2
    elif pf(75) <= x < pf(90):
        return 3
    else:
        return 4
    
def pa (x):
    return np.percentile(df_use['Age'], x)
def Age_convert(x):
    if x < pa(25):
        return 0
    elif pa(25) <= x < pa(50):
        return 1
    elif pa(50) <= x < pa(75):
        return 2
    else:
        return 3

df_use['Fare_conv'] = df_use['Fare'].apply(Fare_convert)
df_use['Age_conv'] = df_use['Age'].apply(Age_convert)


# In[72]:


ac = df_use['Age_conv'].value_counts().reset_index().sort_values(by = 'Age_conv')
sns.barplot(data = ac, x = 'Age_conv', y = 'count')
plt.title('Distribution of Age_conv')


# In[73]:


fc = df_use['Fare_conv'].value_counts().reset_index().sort_values(by = 'Fare_conv')
display(fc)
sns.barplot(data = fc, x = 'Fare_conv', y = 'count')
plt.title('Distribution of Fare_conv')


# In[74]:


df_use['Parch_conv'] = df_use['Parch'].apply(lambda x: 0 if x == 0 else(1 if x == 1 else 2))
df_use['SibSp_conv'] = df_use['SibSp'].apply(lambda x: 0 if x == 0 else(1 if x == 1 else 2))
# df_use['Pclass_conv'] = df_use['Pclass'].apply(lambda x: 0 if x == 1 else(1 if x == 2 else 2))


# In[75]:


display(df_use['SibSp_conv'].value_counts().reset_index())
sns.barplot(data = df_use['SibSp_conv'].value_counts().reset_index(), x = 'SibSp_conv', y = 'count')
plt.title('Distribution of SibSp_conv')
plt.show()


# In[76]:


display(df_use['Parch_conv'].value_counts().reset_index())
sns.barplot(data = df_use['Parch_conv'].value_counts().reset_index(), x = 'Parch_conv', y = 'count')
plt.title('Distribution of Parch_conv')
plt.show()


# In[77]:


df_use.head()


# In[78]:


df_fix = df_use.drop(columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass'])
df_fix.head()


# In[79]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[81]:


#Membuat model train_test_split dulu

X = df_fix.drop(columns = 'Survived')
y = df_fix['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state = 42 , stratify = y)
X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)


# In[87]:


#coba model logistik regression untuk melihat nilai dari akurasi model
LR = LogisticRegression() ##deklarasi
titanic_lr = LR.fit(X_train, y_train)  #model membuat rumus atau mempelajari

y_train_predict = titanic_lr.predict(X_train)
print('Model Logistic Regression')
print('------train')
print(classification_report(y_train, y_train_predict))
y_test_predict = titanic_lr.predict(X_test)
print('------test')
print(classification_report(y_test, y_test_predict))


# In[86]:


#coba model CatBoost
CB = CatBoostClassifier(verbose = False)
titanic_cb = CB.fit(X_train, y_train) 
y_train_predict = titanic_cb.predict(X_train)
print('Model CatBoost')
print('------train')
print(classification_report(y_train, y_train_predict))
y_test_predict = titanic_cb.predict(X_test)
print('------test')
print(classification_report(y_test, y_test_predict))



# In[85]:


#coba model xgboost
GB = xgb.XGBClassifier()
titanic_gb = GB.fit(X_train, y_train) 
y_train_predict = titanic_gb.predict(X_train)
print('Model XGBoost')
print('------train')
print(classification_report(y_train, y_train_predict))
y_test_predict = titanic_gb.predict(X_test)
print('------test')
print(classification_report(y_test, y_test_predict))


# In[88]:


#coba model random forrest 
RF = RandomForestClassifier()
titanic_rf = RF.fit(X_train, y_train) 
y_train_predict = titanic_rf.predict(X_train)
print('Model Random Forest Classifier')
print('------train')
print(classification_report(y_train, y_train_predict))
y_test_predict = titanic_rf.predict(X_test)
print('------test')
print(classification_report(y_test, y_test_predict))


# ### Hyper Parameter Tuning

# In[89]:


#coba model CatBoost
CB = CatBoostClassifier(verbose = False)
titanic_cb = CB.fit(X_train, y_train) 
y_train_predict = titanic_cb.predict(X_train)
print('Model CatBoost')
print('------train')
print(classification_report(y_train, y_train_predict))
y_test_predict = titanic_cb.predict(X_test)
print('------test')
print(classification_report(y_test, y_test_predict))


# In[90]:


a =[1,0.1,0.001,0.0001, 0.00001]

train_accuracy_score = []
test_accuracy_score = []
for i in a:
    CB = CatBoostClassifier(learning_rate = i, verbose = False)
    titanic_cb = CB.fit(X_train, y_train) 
    
    y_train_predict = titanic_cb.predict(X_train)
    train_accuracy_score.append(accuracy_score(y_train, y_train_predict))
    
    y_test_predict = titanic_cb.predict(X_test)
    test_accuracy_score.append(accuracy_score(y_test, y_test_predict))

par_learning_rate = pd.DataFrame({'parameter_learning_rate' :a, 'train_acc': train_accuracy_score, 'test_acc': test_accuracy_score})  
display(par_learning_rate)

sns.lineplot(data = par_learning_rate, x = 'parameter_learning_rate', y = 'train_acc', label = 'train_acc')
sns.lineplot(data = par_learning_rate, x = 'parameter_learning_rate', y = 'test_acc', label = 'test_acc')
plt.title('Parameter Learning Rate of CatBoost Method')
plt.show()


# In[91]:


d =[1,3,5,7,9,11,13]

train_accuracy_score = []
test_accuracy_score = []
for i in d:
    CB = CatBoostClassifier(depth = i, verbose = False)
    titanic_cb = CB.fit(X_train, y_train) 
    
    y_train_predict = titanic_cb.predict(X_train)
    train_accuracy_score.append(accuracy_score(y_train, y_train_predict))
    
    y_test_predict = titanic_cb.predict(X_test)
    test_accuracy_score.append(accuracy_score(y_test, y_test_predict))

par_depth = pd.DataFrame({'parameter_depth' :d, 'train_acc': train_accuracy_score, 'test_acc': test_accuracy_score})  
display(par_depth)

sns.lineplot(data = par_depth, x = 'parameter_depth', y = 'train_acc', label = 'train_acc')
sns.lineplot(data = par_depth, x = 'parameter_depth', y = 'test_acc', label = 'test_acc')
plt.title('Parameter Depth of CatBoost Method')
plt.show()


# In[92]:


e =[1, 50, 100, 200, 300]

train_accuracy_score = []
test_accuracy_score = []
for i in e:
    CB = CatBoostClassifier(n_estimators = i, verbose = False)
    titanic_cb = CB.fit(X_train, y_train) 
    
    y_train_predict = titanic_cb.predict(X_train)
    train_accuracy_score.append(accuracy_score(y_train, y_train_predict))
    
    y_test_predict = titanic_cb.predict(X_test)
    test_accuracy_score.append(accuracy_score(y_test, y_test_predict))

par_n_estimators = pd.DataFrame({'parameter_n_estimators' :e, 'train_acc': train_accuracy_score, 'test_acc': test_accuracy_score})  
display(par_n_estimators)

sns.lineplot(data = par_n_estimators, x = 'parameter_n_estimators', y = 'train_acc', label = 'train_acc')
sns.lineplot(data = par_n_estimators, x = 'parameter_n_estimators', y = 'test_acc', label = 'test_acc')
plt.title('Parameter n_estimators of CatBoost Method')
plt.show()


# In[ ]:




