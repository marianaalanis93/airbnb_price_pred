# AirBnB Price Prediction

This is a complete production project which makes a prediction based in airbnb_listings dataset.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Folder structure

* airbnb
  * api.py: Flask API application
  * request.py: The request for Flask API
  * build_model.py: iinitiates a new model, trains the model, and pickle
  * model
      * XGBRegressor.pkl
  * requirements.txt: list of packages that the app will import
  * data: directory that contains the data file
      * listings_reduced.csv
  * template: the initial page where the new data to be predicted will be added
      * index.html

### Prerequisites

What things you need to install the software and how to install them

```
requirements.txt
```

This is a text file that holds all the library versioning requirements. It consists of all the external (non pre-installed Python libraries) libraries used to execute the code within the pipeline.

### Installing

```
# Install libraries in requirements.txt
pip install -r requirements.txt
```

```
# If you want to run, build and create the pickle file again just:
python build_model.py
If not, it's ok, a version of the pickle model is included in: model/XGBRegressor.pkl
```

End with an example of getting some data out of the system or using it for a little demo

## Run the application

For start running the complete application just:

```
#Run the next command in your cmd/terminal
python api.py
```
Open the url that it shows you, it would be:
```
http://127.0.0.1:5000/
```
And a window, where all fields must be filled in, must be opened (btw sorry for my poor front-end skills
After fill all the files, just press the button "Predict price" and the predicted price will appear.


## Authors

* **Mariana Alanis** - *Initial work* - 


