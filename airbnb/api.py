import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

import io

def cleanDF(df):
    #dropping irrelevant columns 'host_id', 'host_name'
    final_columns =['id', 'name', 'description',
           'host_about', 'host_response_time', 'host_is_superhost',
           'host_listings_count', 'host_identity_verified', 'latitude',
           'longitude', 'room_type', 'bathrooms', 'bedrooms',
           'beds', 'square_feet', 'guests_included', 'minimum_nights',
           'maximum_nights', 'availability_90', 'number_of_reviews',
           'review_scores_rating']

    df = df[final_columns].set_index('id')

    # drop columns with too many Nan's
    df = df.drop(columns=['host_about','host_response_time', 'square_feet','name'])

    # int the other columns drop rows with NaN's 
    df = df.dropna(subset=['host_is_superhost','host_listings_count',
        'host_identity_verified','bathrooms', 'bedrooms','beds', 'review_scores_rating'])

    # extract numbers 
    df['size'] = df['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
    df['size'] = df['size'].str.replace("\D", "")


    # change datatype of size into float
    df['size'] = df['size'].astype(float)
    df['size'] = df['size'].fillna(0)

    # drop description column
    df.drop(['description'], axis=1, inplace=True)

    for col in ['room_type','host_is_superhost','host_identity_verified']:
        df[col] = df[col].astype('category')
    #one-hot encoding, which creates a new column for each unique category in a categorical variable. 
    df[['host_listings_count','latitude','longitude','bathrooms','bedrooms','beds','review_scores_rating','size']] = df[['host_listings_count','latitude','longitude','bathrooms','bedrooms','beds','review_scores_rating','size']].astype(float)
    df[['guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews']] = df[['guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews']].astype(int)
    
    num_feats = df.select_dtypes(include=['float64', 'int64', 'bool']).copy()

    # one-hot encoding of categorical features
    cat_feats = df.select_dtypes(include=['category']).copy()
    from sklearn import preprocessing 

    for col in cat_feats.columns:
        cat_feats[col] = preprocessing.LabelEncoder().fit_transform(cat_feats[col])

    features_recoded = pd.concat([num_feats, cat_feats], axis=1)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_fit = sc.fit(features_recoded)
    X_std = X_fit.transform(features_recoded)

    return X_std


app = Flask(__name__)
model = pickle.load(open('model/XGBRegressor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    values = [x for x in request.form.values()]
    print(type(values))
    df=pd.DataFrame(values)
    df = df.T
    df.columns = ['id','name','description','host_id','host_name','host_since','host_about','host_response_time','host_is_superhost',
    'host_listings_count','host_identity_verified','latitude','longitude','room_type','accommodates','bathrooms','bedrooms','beds','square_feet',
    'guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews','first_review','last_review','review_scores_rating','review_scores_accuracy',
    'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','reviews_per_month']

    print(df)
    clean_features = cleanDF(df)

    prediction = model.predict(clean_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)