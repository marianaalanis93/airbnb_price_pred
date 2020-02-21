import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import xgboost as xgb

#ignoring some warnings for libraries versions
import warnings
warnings.filterwarnings("ignore")

# import train_test_split function
from sklearn.model_selection import train_test_split
# import metrics
from sklearn.metrics import mean_squared_error, r2_score

#Getting raw data
df = pd.read_csv('data/listings_reduced.csv')

#This is for checking the general structure of the data

print(df.head(5))
print(df.columns)
print(df.dtypes)

# checking shape
print("Dataset has {} rows and {} features.".format(*df.shape))

#Data analysis and data cleaning
print("It contains {} duplicates.".format(df.duplicated().sum()))

#dropping irrelevant columns 'host_id', 'host_name'
final_columns =['id', 'name', 'description',
       'host_about', 'host_response_time', 'host_is_superhost',
       'host_listings_count', 'host_identity_verified', 'latitude',
       'longitude', 'room_type', 'bathrooms', 'bedrooms',
       'beds', 'square_feet', 'price', 'guests_included', 'minimum_nights',
       'maximum_nights', 'availability_90', 'number_of_reviews',
       'review_scores_rating']

df = df[final_columns].set_index('id')
print("The final dataset has {} rows and {} columns - after dropping irrelevant columns.".format(*df.shape))

print(df.head(5))

#Cleaning Price Column

print(df[['price']].head(3))


#first check if there are any null values

# checking Nan's in "price" column
print('{} Null values'.format(df.price.isna().sum()))


#dropping $ symbol

# clean up the columns (by method chaining)
df.price = df.price.str.replace('$', '').str.replace(',', '').astype(float)

#checking the distribution of the price
print(df['price'].describe())

#we can see that the minimum is 0 and the maximum is 7000 in the price, lets plot it to see it better

square = dict(markerfacecolor='b', markeredgecolor='b', marker='.')
df['price'].plot(kind='box', xlim=(0, 7000), vert=False, flierprops=square, figsize=(16,2));
plt.show()

#some prices are setted at 0 and only two or three are up to 1000, let's drop the one which are setted at 0 and
#let's see more closer

df.drop(df[ (df.price == 0) ].index, axis=0, inplace=True)

print(df['price'].describe())

square = dict(markerfacecolor='b', markeredgecolor='b', marker='.')
df['price'].plot(kind='box', xlim=(0, 1000), vert=False, flierprops=square, figsize=(16,2));
plt.show()


#let's keep the data in which we have more prices

df.drop(df[ (df.price > 450) ].index, axis=0, inplace=True)

print(df['price'].describe())


print("The dataset has {} rows and {} columns - after price being preprocessed.".format(*df.shape))


#Dealing with Missing Values

#First and most easier approach, dropping all the Nan values

print(df.isna().sum())

# drop columns with too many Nan's
df = df.drop(columns=['host_about','host_response_time', 'square_feet','name'])

# int the other columns drop rows with NaN's 
df = df.dropna(subset=['host_is_superhost','host_listings_count',
	'host_identity_verified','bathrooms', 'bedrooms','beds','review_scores_rating'])

print('NAns dropped')
print(df.isna().sum())


print("The dataset has {} rows and {} columns - after NAns being dropped.".format(*df.shape))


#Second approach, predict nan values with regression (Simply replacing it with the mean or median makes no sense)


# extract numbers 
df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
df['size'] = df['size'].str.replace("\D", "")


# change datatype of size into float
df['size'] = df['size'].astype(float)
df['size'] = df['size'].fillna(0)

# drop description column
df.drop(['description'], axis=1, inplace=True)


print(df.info())

#Modeling the Data
#Preparing Target and Features

#Now let's convert all string columns into categorical ones:

for col in ['room_type','host_is_superhost','host_identity_verified']:
    df[col] = df[col].astype('category')

print(df.info())
# define our target
target = df[["price"]]

# define our features 
features = df.drop(["price"], axis=1)

#not using - one-hot encoding, which creates a new column for each unique category in a categorical variable. 
#instead label encoding

num_feats = features.select_dtypes(include=['float64', 'int64', 'bool']).copy()

# one-hot encoding of categorical features
cat_feats = features.select_dtypes(include=['category']).copy()
from sklearn import preprocessing 

for col in cat_feats.columns:
    cat_feats[col] = preprocessing.LabelEncoder().fit_transform(cat_feats[col])

features_recoded = pd.concat([num_feats, cat_feats], axis=1)

print(features_recoded.shape)
print(features_recoded.head(2))


#Splitting and Scaling the Data

# split our data
X_train, X_test, y_train, y_test = train_test_split(features_recoded, target, test_size=0.2)


# scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

#Training an XGBoost Regressor

# create a baseline
booster = xgb.XGBRegressor()

from sklearn.model_selection import GridSearchCV

# create Grid
param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)


# instantiate xgboost with best parameters
booster = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, 
                           max_depth=6, n_estimators=200, random_state=4)

# train
booster.fit(X_train, y_train)

# predict
y_pred_train = booster.predict(X_train)
y_pred_test = booster.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE: {round(RMSE, 4)}")

r2 = r2_score(y_test, y_pred_test)

print(f"r2: {round(r2, 4)}")

xg_train = xgb.DMatrix(data=X_train, label=y_train)

params = {'colsample_bytree':0.6, 'gamma':0.2, 'learning_rate':0.05, 'max_depth':6}

cv_results = xgb.cv(dtrain=xg_train, params=params, nfold=3,
                    num_boost_round=200, early_stopping_rounds=10, 
                    metrics="rmse", as_pandas=True)


#Saves the trained vectorizer for future use.

path='model/XGBRegressor.pkl'


with open(path, 'wb') as f:
    pickle.dump(booster, f)
    print("Pickled boster at {}".format(path))

