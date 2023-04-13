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

#OMITTING MISSING VALUES

train_data_omit=train_data.dropna(axis=0)

train_data_omit=pd.get_dummies(train_data_omit, drop_first=True)


test_data_omit=train_data.dropna(axis=0)

test_data_omit=pd.get_dummies(train_data_omit, drop_first=True)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


x1=train_data_omit.drop(['Item_Outlet_Sales'], axis='columns', inplace=False)
y1=train_data_omit['Item_Outlet_Sales']
x2=test_data_omit.drop(['Item_Outlet_Sales'], axis='columns', inplace=False)
y2=test_data_omit['Item_Outlet_Sales']

#finding mean of test data value
base_pred=np.mean(y2)
print(base_pred)

base_pred=np.repeat(base_pred, len(y2))

#LINEAR REGRESSION WITH OMITTED DATA

lgr=LinearRegression(fit_intercept=True)

#model
model_lin1=lgr.fit(x1,y1)

#predicting model on test set

Sales_prediction_lin1=lgr.predict(x2)

#MSE AND RMSE to measure the accuracy of the model

lin_mse1 = mean_squared_error(y2, Sales_prediction_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

#R-squared value

r2_lin_test1=model_lin1.score(x2,y2)
r2_lin_train1=model_lin1.score(x1,y1)
print(r2_lin_test1,r2_lin_train1)

residuals1=y2-Sales_prediction_lin1
sns.regplot(x=Sales_prediction_lin1, y=residuals1, scatter=True, 
            fit_reg=False)
residuals1.describe()


#RANDOM FOREST REGRESSION 

rf = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

# Model
model_rf1=rf.fit(x1,y1)

# Predicting model on test set
cars_predictions_rf1 = rf.predict(x2)

# Computing MSE and RMSE
rf_mse1 = mean_squared_error(y2, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1=model_rf1.score(x2,y2)
r2_rf_train1=model_rf1.score(x1,y1)
print(r2_rf_test1,r2_rf_train1) 

#BUILDING MODEL WITH IMPUTED DATA

train_data_imputed=train_data.apply(lambda x:x.fillna(x.mean()) \
                  if x.dtype=='float' else \
                  x.fillna(x.value_counts().index[0]))
train_data_imputed.isnull().sum()

test_data_imputed=test_data.apply(lambda x:x.fillna(x.mean()) \
                  if x.dtype=='float' else \
                  x.fillna(x.value_counts().index[0]))
test_data_imputed.isnull().sum()

#CATEGORICAL VARIABLE TO DUMMY VARIABLES

train_data_imputed=pd.get_dummies(train_data_imputed,drop_first=True) 
test_data_imputed=pd.get_dummies(test_data_imputed,drop_first=True) 

x3=train_data_imputed.drop(['Item_Outlet_Sales'] , inplace=False)
y3=test_data_imputed['Item_Outlet_Sales']
x4=train_data_imputed.drop(['Item_Outlet_Sales'] , inplace=False)
y4=test_data_imputed['Item_Outlet_Sales']



#finding mean of test data value
base_pred2=np.mean(y4)
print(base_pred2)

base_pred=np.repeat(base_pred, len(y4))

#LINEAR REGRESSION WITH OMITTED DATA

lgr2=LinearRegression(fit_intercept=True)

#model
model_lin2=lgr.fit(x3,y3)

#predicting model on test set

Sales_prediction_lin2=lgr.predict(x4)

#MSE AND RMSE to measure the accuracy of the model

lin_mse2 = mean_squared_error(y4, Sales_prediction_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

#R-squared value

r2_lin_test2=model_lin2.score(x4,y4)
r2_lin_train2=model_lin2.score(x3,y3)
print(r2_lin_test2,r2_lin_train2)

#RANDOM FOREST WITH IMPUTED DATA
# Model parameters
rf2 = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

# Model
model_rf2=rf2.fit(x3,y3)

# Predicting model on test set
cars_predictions_rf2 = rf2.predict(x4)

# Computing MSE and RMSE
rf_mse2 = mean_squared_error(y4, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)

# R squared value
r2_rf_test2=model_rf2.score(x4,y4)
r2_rf_train2=model_rf2.score(x3,y4)
print(r2_rf_test2,r2_rf_train2)     


print("Metrics for models built from data where missing values were omitted")
print("R squared value for train from Linear Regression=  %s"% r2_lin_train1)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test1)
print("R squared value for train from Random Forest=  %s"% r2_rf_train1)
print("R squared value for test from Random Forest=  %s"% r2_rf_test1)

print("RMSE value for test from Linear Regression=  %s"% lin_rmse1)
print("RMSE value for test from Random Forest=  %s"% rf_rmse1)
print("\n\n")
print("Metrics for models built from data where missing values were imputed")
print("R squared value for train from Linear Regression=  %s"% r2_lin_train2)
print("R squared value for test from Linear Regression=  %s"% r2_lin_test2)
print("R squared value for train from Random Forest=  %s"% r2_rf_train2)
print("R squared value for test from Random Forest=  %s"% r2_rf_test2)

print("RMSE value for test from Linear Regression=  %s"% lin_rmse2)
print("RMSE value for test from Random Forest=  %s"% rf_rmse2)



























