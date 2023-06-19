#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


#Sklearn imports 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# In[154]:


#Reading dataset
df = pd.read_csv('train.csv')
df.drop('Unnamed: 0', axis = 1, inplace=True)


# In[4]:


X_valid = pd.read_csv('test.csv')


# In[30]:


#Correlation map 
sns.heatmap(df.corr())
plt.show


# In[8]:


df


# In[179]:


df.isna().sum()


# In[155]:


#Handeling missing values of position 
from sklearn.compose import ColumnTransformer
column_trans = ColumnTransformer([('imp_lap', SimpleImputer(missing_values = 0.000000,strategy='median'), ['latitude'])])
df['latitude'] = column_trans.fit_transform(df)
column_trans2 = ColumnTransformer([('imp_long', SimpleImputer(missing_values = 0.000000,strategy='median'), ['longitude'])])
df['longitude'] = column_trans2.fit_transform(df)


# In[156]:


#features and target
X = df.drop('status', axis=1)
Y = df['status']


# In[131]:


binarizer = make_column_transformer((OrdinalEncoder(),['status']))
df['status'] = binarizer.fit_transform(df)
df


# In[113]:


X


# In[104]:


df['date'] = pd.to_datetime(df['date'], unit='ms')


# In[157]:


class handledate(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform (self, X):
        X['date'] = pd.to_datetime(X['date'], unit='ms')
        X['weekday'] = X.date.dt.weekday
        X['month'] = X.date.dt.month
        X['hour'] = X.date.dt.hour
       
        return X



# In[184]:


X.columns


# In[185]:


def calc_mean(df,n,column):
    np_df = df.loc[(df[column] != 0.000000) & (df['deliveryPrice']==n)]
    mean = np_df[column].mean()
    return mean
   


# In[158]:


#Droppers
class droppcolumns(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform (self, X):
        return X.drop(['city','name','date'], axis=1)
    
#dc = droppcolumns()
#dc.fit_transform(X)


# In[160]:


Y


# In[159]:


hd = handledate()
hd.fit_transform(X)


# In[116]:


X


# In[161]:


# handle date
#hd = handledate()
#hd.fit_transform(X)


#Encodage
categorial_features = ['product_id','paid','storeUID','clientUID','deliveryUID']
numerique_features = ['unit_Price','qte','deliveryPrice','latitude','longitude','order_price','weekday','month','hour']

#piplines for categories 
numerique_piplines= make_pipeline(StandardScaler())
categorial_piplines = make_pipeline(SimpleImputer(missing_values=np.NaN, strategy='most_frequent'),OrdinalEncoder(unknown_value=-1,handle_unknown='use_encoded_value'),StandardScaler())
#categorial_piplines = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

#tobin_piplines = make_pipeline(LabelBinarizer(),StandardScaler())

preprocess = make_column_transformer((numerique_piplines,numerique_features),(categorial_piplines,categorial_features))

dc = droppcolumns()
sc = StandardScaler()
#from sklearn.preprocessing import PolynomialFeatures
#pf = PolynomialFeatures(2)

process = make_pipeline(dc, preprocess)


# In[162]:


X_transformed = process.fit_transform(X)


# In[163]:


#X_transformed = process.fit_transform(X)
#X_transformed = pd.DataFrame.sparse.from_spmatrix(X_transformed)
X_transformed = pd.DataFrame(X_transformed)

X_transformed.rename({
    0 : 'unit_Price',
    1 : 'qte',
    2 : 'deliveryPrice',
    3 : 'latitude',
    4 : 'longitude',
    5 : 'order_price',
    6 : 'weekday',
    7 : 'month',
    8 : 'hour',
    9 : 'product_id',
    10 : 'paid',
    11 : 'storeUID',
    12 : 'clientUID',
    13 : 'deliveryUID',
},axis=1,inplace=True)


# In[164]:


X_transformed


# In[165]:


#creating the split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.33, random_state=42)


# In[41]:


rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [700,1000,1100],
    'max_features': ['auto'],
    'max_depth' : [10,12,15,20],
    'criterion' :['gini']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[42]:


CV_rfc.best_params_


# In[166]:


Fmodel = RandomForestClassifier(random_state=42,criterion = 'gini',max_depth = 20,max_features ='auto',n_estimators =1000)


# In[167]:


Fmodel.fit(X_train,y_train)
#Scoring


# In[168]:


from sklearn.metrics import f1_score
print(f'the accuracy of the model on the training set is {Fmodel.score(X_train, y_train)}')
print(f'the accuracy of the model on the validation set is {Fmodel.score(X_test, y_test)}')


# In[146]:


print(f'the accuracy of the model on the training set is {Fmodel.score(X_train, y_train)}')
print(f'the accuracy of the model on the validation set is {Fmodel.score(X_test, y_test)}')


# In[152]:


XT = pd.DataFrame(X_transformed)
XT


# In[69]:


X_transformed['deliveryPrice'].value_counts()


# In[70]:


X['deliveryPrice'].value_counts()


# In[169]:


Fmodel.fit(X_transformed,Y)


# In[60]:


#Prediction with old model
X_valid = pd.read_csv('test.csv')
X_valid = process.transform(X_valid)
y_valid_pred = Fmodel.predict(X_valid)
#Submition
sub = pd.DataFrame(data={'ID': range(len(y_valid_pred)), 'status': y_valid_pred}).set_index("ID")
sub.to_csv('six_out.csv')


# In[89]:


data


# In[86]:


data


# In[147]:


data = pd.concat([X_transformed, pd.DataFrame(Y)],axis=1)
#Correlation map 
plt.figure(figsize=(16, 6))
sns.heatmap(data.corr(),annot=True)
plt.show


# In[170]:


#features importance 
importances = Fmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in Fmodel.estimators_], axis=0)
forest_importances = pd.Series(importances, index=X_transformed.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# In[171]:


#Update models on Feautures importance
columns_to_drop = ['qte','unit_Price','deliveryPrice','storeUID','weekday']
X_transformed_f = pd.DataFrame(X_transformed).drop(columns_to_drop, axis=1)
X_transformed_f
Ftmodel = RandomForestClassifier(random_state=42,criterion = 'gini',max_depth = 15,max_features ='auto',n_estimators =700)


Ftmodel.fit(X_transformed_f, Y)
#creating the split
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_transformed_f, Y, test_size=0.33, random_state=42)
#Calculing the score 
print(f'the accuracy of the model on the training set is {Ftmodel.score(Xf_train, yf_train)}')
print(f'the accuracy of the model on the validation set is {Ftmodel.score(Xf_test, yf_test)}')


# In[ ]:


the accuracy of the model on the training set is 0.9983471074380166
the accuracy of the model on the validation set is 0.9463687150837988


# In[173]:


#prediction with new model 
X_valid = pd.read_csv('test.csv')
#X_valid['cat'] = X_valid['unit_Price']
X_valid = hd.transform(X_valid)
X_valid = process.transform(X_valid)
X_valid = pd.DataFrame(X_valid)

X_valid.rename({
    0 : 'unit_Price',
    1 : 'qte',
    2 : 'deliveryPrice',
    3 : 'latitude',
    4 : 'longitude',
    5 : 'order_price',
    6 : 'weekday',
    7 : 'month',
    8 : 'hour',
    9 : 'product_id',
    10 : 'paid',
    11 : 'storeUID',
    12 : 'clientUID',
    13 : 'deliveryUID',
},axis=1,inplace=True)

X_valid = pd.DataFrame(X_valid).drop(columns_to_drop, axis=1)
y_valid_pred = Ftmodel.predict(X_valid)

sub = pd.DataFrame(data={'ID': range(len(y_valid_pred)), 'status': y_valid_pred}).set_index("ID")
sub.to_csv('last2_out.csv')


# In[200]:


get_ipython().system('pip install xgboost')


# In[222]:


#Changing to xgboost
import xgboost as xgb

lab = LabelBinarizer()
ny_train = lab.fit_transform(y_train)

dtrain = xgb.DMatrix(data=X_train, label=ny_train)
num_parallel_tree = 4
num_boost_round = 16
# total number of built trees is num_parallel_tree * num_classes * num_boost_round

# We build a boosted random forest for classification here.
booster = xgb.train({
    'num_parallel_tree': 10, 'subsample': 0.5, 'num_class': 5},
                    num_boost_round=num_boost_round, dtrain=dtrain)


# In[227]:


modelx = xgb.XGBClassifier(nthread=4,
    seed=42)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

CV_rfc = GridSearchCV(estimator=modelx, param_grid=parameters, cv= 3,verbose=True)
CV_rfc.fit(X_train, ny_train)

#modelx.fit(X_train, ny_train)
#expected_y  = ny_test
#predicted_y = modelx.predict(X_test)


# In[228]:


CV_rfc.best_params_


# In[229]:


modelx = xgb.XGBClassifier(learning_rate=0.1, max_depth= 9, n_estimators= 140,nthread=4,
    seed=42)
modelx.fit(X_train, ny_train)
expected_y  = ny_test
predicted_y = modelx.predict(X_test)


# In[230]:


ny_test = lab.fit_transform(y_test)


# In[231]:


print(f'the accuracy of the model on the training set is {modelx.score(X_train, ny_train)}')
print(f'the accuracy of the model on the validation set is {modelx.score(X_test, ny_test)}')


# In[ ]:


0.9474860335195531


# In[232]:


nY = lab.fit_transform(Y)
modelx.fit(X_transformed, nY)


# In[233]:


#prediction with xgb model 
X_valid = pd.read_csv('test.csv')
#X_valid['cat'] = X_valid['unit_Price']
X_valid = process.transform(X_valid)
#X_valid = pd.DataFrame(X_valid).drop([1,2,7,0], axis=1)
y_valid_pred = modelx.predict(X_valid)


# In[234]:


y_valid_pred = lab.inverse_transform(y_valid_pred)
sub = pd.DataFrame(data={'ID': range(len(y_valid_pred)), 'status': y_valid_pred}).set_index("ID")
sub.to_csv('ten_out.csv')


# In[ ]:




