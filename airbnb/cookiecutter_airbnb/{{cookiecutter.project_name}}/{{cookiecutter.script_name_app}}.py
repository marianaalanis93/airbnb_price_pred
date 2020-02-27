import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

import io

#import build model methods
from model import RegressionModel

def {{cookiecutter.def_cleanDF}}(df):

    model = {{cookiecutter.class_name}}()

    df = model.{{cookiecutter.def_preprocess}}(df)

    return model.{{cookiecutter.def_scaleDataPredict}}(df)


app = Flask(__name__)
model = pickle.load(open({{cookiecutter.path_model_pkl}}, 'rb'))

@app.route('/')
def {{cookiecutter.def_home}}():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def {{cookiecutter.def_predict}}():

    values = [x for x in request.form.values()]
    df=pd.DataFrame(values)
    df = df.T
    df.columns = ['id','name','description','host_id','host_name','host_since','host_about','host_response_time','host_is_superhost',
    'host_listings_count','host_identity_verified','latitude','longitude','room_type','accommodates','bathrooms','bedrooms','beds','square_feet',
    'guests_included','minimum_nights','maximum_nights','availability_90','number_of_reviews','first_review','last_review','review_scores_rating','review_scores_accuracy',
    'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','reviews_per_month']

    clean_features = cleanDF(df)

    prediction = model.predict(clean_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def {{cookiecutter.def_results}}():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)