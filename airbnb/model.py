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
from sklearn import preprocessing 
# scale data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#for saving best params
from sklearn.externals import joblib


class RegressionModel(object):
  def __init__(self):
    """Regression model
        Attributes:
            xgb: xgboost
    """
    self.booster = xgb.XGBRegressor()
    self.sc = StandardScaler()

  def getData(self):
    #Getting raw data
    df = pd.read_csv('data/listings_reduced.csv')
    return df

  def preprocess(self, df):
    #dropping irrelevant columns 'host_id', 'host_name'
    if "price" in df.columns:
      final_columns =['id', 'name', 'description',
             'host_about', 'host_response_time', 'host_is_superhost',
             'host_listings_count', 'host_identity_verified', 'latitude',
             'longitude', 'room_type', 'bathrooms', 'bedrooms',
             'beds', 'square_feet', 'price', 'guests_included', 'minimum_nights',
             'maximum_nights', 'availability_90', 'number_of_reviews',
             'review_scores_rating']
    else:
      final_columns =['id', 'name', 'description',
             'host_about', 'host_response_time', 'host_is_superhost',
             'host_listings_count', 'host_identity_verified', 'latitude',
             'longitude', 'room_type', 'bathrooms', 'bedrooms',
             'beds', 'square_feet', 'guests_included', 'minimum_nights',
             'maximum_nights', 'availability_90', 'number_of_reviews',
             'review_scores_rating']

    df = df[final_columns].set_index('id')

    #Cleaning Price Column

    #dropping $ symbol

    # clean up the columns (by method chaining)
    if "price" in df.columns:
      df.price = df.price.str.replace('$', '').str.replace(',', '').astype(float)

      #some prices are setted at 0 and only two or three are up to 1000, let's drop the one which are setted at 0 and
      df.drop(df[ (df.price == 0) ].index, axis=0, inplace=True)

      #let's keep the data in which we have more prices
      df.drop(df[ (df.price > 450) ].index, axis=0, inplace=True)

    #Dealing with Missing Values

    # drop columns with too many Nan's
    df = df.drop(columns=['host_about','host_response_time', 'square_feet','name'])

    # int the other columns drop rows with NaN's 
    df = df.dropna(subset=['host_is_superhost','host_listings_count',
      'host_identity_verified','bathrooms', 'bedrooms','beds','review_scores_rating'])

    # extract numbers for the airbnb size (we dropped 'square_feet' because it has to many nans values)
    df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
    df['size'] = df['size'].str.replace("\D", "")

    # change datatype of size into float
    df['size'] = df['size'].astype(float)
    df['size'] = df['size'].fillna(0)

    #set types of numbers
    df[['host_listings_count','latitude','longitude','bathrooms','bedrooms','beds','review_scores_rating','size']] = df[['host_listings_count','latitude','longitude','bathrooms','bedrooms','beds','review_scores_rating','size']].astype(float)
    df[['guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews']] = df[['guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews']].astype(int)


    # after that, drop description column
    df.drop(['description'], axis=1, inplace=True)

    #Modeling the Data
    #Preparing Target and Features

    #Now let's convert all string columns into categorical ones:

    for col in ['room_type','host_is_superhost','host_identity_verified']:
        df[col] = df[col].astype('category')

    # define our features 
    if "price" in df.columns:
      features = df.drop(["price"], axis=1)
    else: 
      features = df

    #not using - one-hot encoding, which creates a new column for each unique category in a categorical variable. 
    #instead label encoding

    num_feats = features.select_dtypes(include=['float64', 'int64', 'bool']).copy()

    # one-hot encoding of categorical features
    cat_feats = features.select_dtypes(include=['category']).copy()

    for col in cat_feats.columns:
        cat_feats[col] = preprocessing.LabelEncoder().fit_transform(cat_feats[col])

    features_recoded = pd.concat([num_feats, cat_feats], axis=1)

    print(features_recoded.info())

    # define our target
    if "price" in df.columns:
      target = df[["price"]]
      return features_recoded,target
    else:
      return features_recoded


  def splitData(self, df, target):
    #return X_train, X_test, y_train, y_test
    return train_test_split(df, target, test_size=0.2)

  def scaleDataTrain(self, X_train, X_test):
    # scale data
    X_train = self.sc.fit_transform(X_train)
    X_test  = self.sc.transform(X_test)
    return X_train, X_test

  def scaleDataPredict(self, df):
    array = df.to_numpy()
    return array

  def parameterOptimizer(self, X_train, y_train):
    # create Grid
    param_grid = {'n_estimators': [100, 150, 200],
                  'learning_rate': [0.01, 0.05, 0.1], 
                  'max_depth': [3, 4, 5, 6, 7],
                  'colsample_bytree': [0.6, 0.7, 1],
                  'gamma': [0.0, 0.1, 0.2]}

    # instantiate the tuned random forest
    booster_grid_search = GridSearchCV(self.booster, param_grid, cv=3, n_jobs=-1)

    # train the tuned random forest
    booster_grid_search.fit(X_train, y_train)
    # save best estimator parameters found during the grid search
    joblib.dump(booster_grid_search, 'model/best_params/best_booster_params.pkl', compress = 1) # Only best parameters

  def train(self, X_train, y_train, X_test):
    # Load best parameters
    booster_params = joblib.load('model/best_params/best_booster_params.pkl')
    self.booster = booster_params.best_estimator_
    # train
    self.booster.fit(X_train, y_train)

    # predict
    y_pred_train = self.booster.predict(X_train)
    y_pred_test = self.booster.predict(X_test)

    return self.booster, y_pred_train, y_pred_test

  def prediction(model, df):
    return model.predict(df)


  def metrics(self, y_test, y_pred_test, X_train, y_train):
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2 = r2_score(y_test, y_pred_test)

    xg_train = xgb.DMatrix(data=X_train, label=y_train)

    params = {'colsample_bytree':0.6, 'gamma':0.2, 'learning_rate':0.05, 'max_depth':6}

    cv_results = xgb.cv(dtrain=xg_train, params=params, nfold=3,
                        num_boost_round=200, early_stopping_rounds=10, 
                        metrics="rmse", as_pandas=True)

    return RMSE, r2, cv_results

  def pickleModel(self, model):
    #Saves the trained vectorizer for future use.

    path='model/XGBRegressor.pkl'


    with open(path, 'wb') as f:
        pickle.dump(model, f)
        print("Pickled booster at {}".format(path))



if __name__ == "__main__":

    model = RegressionModel()
    print('Building model...')
    print('Getting data...')
    df = model.getData()
    print('Preprocessing...')
    features,target = model.preprocess(df)
    print('Splitting data...')
    X_train, X_test, y_train, y_test = model.splitData(features, target)
    print('Scaling data...')
    X_train, X_test = model.scaleDataTrain(X_train, X_test)
    print('Optimizing data and pickle it...')
    #model.parameterOptimizer(X_train, y_train)
    print('Training and testing data...')
    xgb_model, y_pred_train, y_pred_test = model.train(X_train, y_train, X_test)
    print('Obtaining metrics...')
    RMSE, r2, cv_results = model.metrics(y_test, y_pred_test, X_train, y_train)
    print('RMSE: ',RMSE)
    print('R2: ', r2)
    print('CV Results: ',cv_results)
    print('Pickle model...')
    model.pickleModel(xgb_model)
    print('Building model finished...')




