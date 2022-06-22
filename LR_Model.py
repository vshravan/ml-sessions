# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:15:25 2020

@author: shrav
"""

#1. Importing the required libraries for EDA

import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
  
sns.set(color_codes=True)


#2. Loading the data into the data frame

df = pd.read_csv("Housing_Price_170720_test.csv")

df.columns

#df.head(5)

df.dtypes

df.isnull().mean()*100

df.shape

'''
#Identifying Numerical Variables

num_var_list=['Length','Width', 'Floor', 'Balcony_Area']

#Identifying Categorical Variables

cat_var_list=['Corner','Special_Feature']

def impute_na_num(df, variable):

    return df[variable].fillna(df[variable].mean())


for var in num_var_list:
    print(var)
    df[var] = impute_na_num(df, var)
    
df.isnull().mean()


def impute_na_cat(df, variable):

    return df[variable].fillna(df[variable].mode().iloc[0])
    #return df[variable].fillna(df[variable].mode())

for var in cat_var_list:
    print(var)
    df[var] = impute_na_cat(df, var)
    
    
df.isnull().mean()

'''

df=df.dropna()

df.shape

df.isnull().mean()*100

df.boxplot()

sns.boxplot(x=df['Length'])

sns.boxplot(x=df['Width'])

sns.boxplot(x=df['Floor'])

sns.boxplot(x=df['Corner'])

sns.boxplot(x=df['Balcony_Area'])

list_out=["Length", "Width"]

'''

lb = df_out.quantile(0.1)
ub = df_out.quantile(0.9)

df_out=df_out.clip(lower=df_out.quantile(0.1), upper=df_out.quantile(0.9), axis=1)

'''

'''

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df.shape

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

'''

def remove_outlier(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out

for var in list_out:
    df=remove_outlier(df,var)
    
df.shape

########Feature engineering##################
df['Area']=df['Length']*df['Width']

df.head()

df=df.drop(['Length','Width'],axis=1)

df=df.drop(['Id'],axis=1)

df

df.shape

dummy=pd.get_dummies(df['Special_Feature'], drop_first=True)
#dummy=pd.get_dummies(df['Special_Feature'])

df=pd.concat([df,dummy], axis=1)
df=df.drop(["Special_Feature"], axis=1)

#
#from sklearn.preprocessing import LabelEncoder 
#
#le = LabelEncoder() 
#
#df['Special_Feature']= le.fit_transform(df['Special_Feature']) 

df.head()

df.columns



df.shape

###########splitting data to train & test#############

from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 2892356)


#from sklearn.preprocessing import MinMaxScaler
#
#scaler = MinMaxScaler()
#
#num_vars = ['Floor',  'Special_Feature', 'Balcony_Area', 'Price', 'Area']
#
#df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
#
#type(df_train)
#
#df_test[num_vars] = scaler.transform(df_test[num_vars])


#
#
#
### Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
##num_vars = ['Floor','Balcony_Area', 'Area']
##
##df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
##
##df_test[num_vars] = scaler.transform(df_test[num_vars])


y_train = df_train.pop('Price')
X_train = df_train

####sklearn########

## Fitting Simple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
#regressor.coef_
#regressor.intercept_
#
#print('Slope:' ,regressor.coef_)
#print('Intercept:', regressor.intercept_)


###############second method################################

import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train)

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()

lr.params

print(lr.summary())


y_test = df_test.pop('Price')
X_test = df_test

X_test_lm = sm.add_constant(X_test)

y_pred_lm = lr.predict(X_test_lm)

r2_test = r2_score(y_test, y_pred_lm)

plt.figure(figsize=(10,5))
c= X_train_lm.corr()
sns.heatmap(c,cmap="BrBG",annot=True)


#####################VIF Calculations#######################

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_lm.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

###################removing Balcony_Area n BUilding Model#############################

X_train_lm_1 = X_train_lm.drop(["Balcony_Area"], axis=1)

lr_1 = sm.OLS(y_train, X_train_lm_1).fit()

#lr_1.params

print(lr_1.summary())

X_test_lm_1 = X_test_lm.drop(["Balcony_Area"], axis=1)
y_pred_test_lm_1 = lr_1.predict(X_test_lm_1)

r2_test_1 = r2_score(y_test, y_pred_test_lm_1)

#####################VIF Calculations#######################

vif = pd.DataFrame()
vif['Features'] = X_train_lm_1.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm_1.values, i) for i in range(X_train_lm_1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

###################removing Corner n BUilding Model#############################

X_train_lm_2 = X_train_lm_1.drop(["Corner"], axis=1)

lr_2 = sm.OLS(y_train, X_train_lm_2).fit()

lr_2.params

print(lr_2.summary())

X_test_lm_2 = X_test_lm_1.drop(["Corner"], axis=1)
y_pred_test_lm_2 = lr_2.predict(X_test_lm_2)

r2_test_2 = r2_score(y_test, y_pred_test_lm_2)

y_pred_train_lm_2 = lr_2.predict(X_train_lm_2)


###################removing Pool Facing n BUilding Model#############################


X_train_lm_3 = X_train_lm_2.drop(["Pool Facing"], axis=1)

lr_3 = sm.OLS(y_train, X_train_lm_3).fit()

lr_3.params

print(lr_3.summary())

X_test_lm_3 = X_test_lm_2.drop(["Pool Facing"], axis=1)
y_pred_test_lm_3 = lr_3.predict(X_test_lm_3)
type(y_pred_test_lm_3)

r2_test_3 = r2_score(y_test, y_pred_test_lm_3)

y_pred_train_lm_3 = lr_3.predict(X_train_lm_3)

###################removing Mountain Facing n BUilding Model#############################


X_train_lm_4 = X_train_lm_3.drop(["Mountain Facing"], axis=1)

lr_4 = sm.OLS(y_train, X_train_lm_4).fit()

lr_4.params

print(lr_4.summary())

X_test_lm_4 = X_test_lm_3.drop(["Mountain Facing"], axis=1)
y_pred_test_lm_4 = lr_4.predict(X_test_lm_4)

r2_test_4 = r2_score(y_test, y_pred_test_lm_4)

y_pred_train_lm_4 = lr_4.predict(X_train_lm_4)

###################removing Floor n BUilding Model#############################


X_train_lm_5 = X_train_lm_4.drop(["Floor"], axis=1)

lr_5 = sm.OLS(y_train, X_train_lm_5).fit()

lr_5.params

print(lr_5.summary())

X_test_lm_5 = X_test_lm_4.drop(["Floor"], axis=1)
y_pred_test_lm_5 = lr_5.predict(X_test_lm_5)

r2_test_5 = r2_score(y_test, y_pred_test_lm_5)

y_pred_train_lm_5 = lr_5.predict(X_train_lm_5)

import pickle

# Saving model to disk
pickle.dump(lr_5, open('D:/ikigai/SLRnMLR/lr_5.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('D:/ikigai/SLRnMLR/lr_5.pkl','rb'))
print(model.predict([[6, 90, 8]]))


###################Final Model is lr_5###############################

####################Prediction Model####################################

#1. Importing the required libraries for EDA

import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
  
sns.set(color_codes=True)


#2. Loading the data into the data frame

df = pd.read_csv("Housing_Price_231120.csv")

df.columns

#df.head(5)

df.dtypes

df.isnull().mean()*100

df.shape

'''
#Identifying Numerical Variables

num_var_list=['Length','Width', 'Floor', 'Balcony_Area']

#Identifying Categorical Variables

cat_var_list=['Corner','Special_Feature']

def impute_na_num(df, variable):

    return df[variable].fillna(df[variable].mean())


for var in num_var_list:
    print(var)
    df[var] = impute_na_num(df, var)
    
df.isnull().mean()


def impute_na_cat(df, variable):

    return df[variable].fillna(df[variable].mode().iloc[0])
    #return df[variable].fillna(df[variable].mode())

for var in cat_var_list:
    print(var)
    df[var] = impute_na_cat(df, var)
    
    
df.isnull().mean()

'''

df=df.dropna()

df.shape

df.isnull().mean()*100

df.boxplot()

sns.boxplot(x=df['Length'])

sns.boxplot(x=df['Width'])

sns.boxplot(x=df['Floor'])

sns.boxplot(x=df['Corner'])

sns.boxplot(x=df['Balcony_Area'])

list_out=["Length", "Width"]

'''

lb = df_out.quantile(0.1)
ub = df_out.quantile(0.9)

df_out=df_out.clip(lower=df_out.quantile(0.1), upper=df_out.quantile(0.9), axis=1)

'''

'''

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df.shape

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

'''

def remove_outlier(df, col_name):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    return df_out

for var in list_out:
    df=remove_outlier(df,var)
    
df.shape

########Feature engineering##################
df['Area']=df['Length']*df['Width']

df.head()

df=df.drop(['Length','Width'],axis=1)

df=df.drop(['Id'],axis=1)

df

df.shape

dummy=pd.get_dummies(df['Special_Feature'], drop_first=True)
#dummy=pd.get_dummies(df['Special_Feature'])

df=pd.concat([df,dummy], axis=1)
df=df.drop(["Special_Feature"], axis=1)

#
#from sklearn.preprocessing import LabelEncoder 
#
#le = LabelEncoder() 
#
#df['Special_Feature']= le.fit_transform(df['Special_Feature']) 

df.head()

df.columns



df.shape

del df['Floor']

del df['Floor']
del df['Corner']
del df['Balcony_Area']
del df['Mountain Facing']
del df['Pool Facing']
del df['Floor']

y=df.pop('Price')
X=df

X_c = sm.add_constant(X)

y_pred_231120=lr_5.predict(X_c)










#