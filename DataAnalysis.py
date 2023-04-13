import os
import pandas as pd 
import numpy as np
import seaborn as sns

train_data=pd.read_csv('train_v9rqX0R.csv')
test_data=pd.read_csv('test_AbJTz2l.csv')

#keeping a copy for reference

train_data1=train_data.copy()
test_data1=test_data.copy()

#Dataset Structure

train_data.info()
test_data.info()

#summarising data
pd.set_option('display.float_format',lambda x:'%.3f' % x) #limiting describe 
print(train_data.describe()) 
print(test_data.describe()) 

#FINDING NULL VALUES IN DATA

print(train_data.isnull().sum())

print(test_data.isnull().sum())



#REPLACING EMPTY VALUES WITH MODE BECAUSE CATEGORICAL VARIABLE

mode=train_data['Outlet_Size'].mode().values[0]
train_data['Outlet_Size']=train_data['Outlet_Size'].replace(np.nan, mode)

mode=test_data['Outlet_Size'].mode().values[0]
test_data['Outlet_Size']=test_data['Outlet_Size'].replace(np.nan, mode)

print(train_data.isnull().sum())

print(test_data.isnull().sum())


mean1=train_data['Item_Weight'].mean()
mean2=test_data['Item_Weight'].mean()

median1=train_data['Item_Weight'].median()
median2=test_data['Item_Weight'].median()

train_data['Item_Weight']=train_data['Item_Weight'].replace(np.nan, mean1)
test_data['Item_Weight']=test_data['Item_Weight'].replace(np.nan, mean2)

print(train_data.isnull().sum())

print(test_data.isnull().sum())

#ALL NULL VALUES HAVE BEEN FILLED

sns.boxplot(x='Item_MRP', y='Outlet_Size', data=train_data)

#SHOWS WE CAN PUT MORE MRP FOR BIGGER STORE SIZE

pd.crosstab(index=train_data['Item_Fat_Content'], columns='count')
pd.crosstab(index=test_data['Item_Fat_Content'], columns='count')


train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('LF', 'low fat')
train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('Low Fat', 'low fat')
train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('reg', 'Regular')

test_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('LF', 'low fat')
test_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('Low Fat', 'low fat')
test_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('reg', 'Regular')













