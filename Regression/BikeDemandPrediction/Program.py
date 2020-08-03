import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import TypeConverter
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.ensemble import FastTreesRegressor

# load the file
dataFrame = pd.read_csv("bikedemand.csv", 
                        sep=',', 
                        header=0)

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"], result_type = "R4"),
    ColumnConcatenator() << {"Feature":["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]},
    FastTreesRegressor() << {Role.Label:"cnt", Role.Feature:"Feature"}
])

# train the model
pipeline.fit(trainData)

# evaluate the model and report metrics
metrics, _ = pipeline.test(testData)
print("\nEvaluation metrics:")
print("  RMSE: ", metrics["RMS(avg)"][0])
print("  MSE: ", metrics["L2(avg)"][0])
print("  MAE: ", metrics["L1(avg)"][0])

# set up a sample
sample = pd.DataFrame(  [[3.0, 1.0, 8.0, 10.0, 0.0, 4.0, 1.0, 1.0, 0.8, 0.7576, 0.55, 0.2239]],
                            columns = ["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"])

# create prediction for sample
prediction = pipeline.predict(sample)
print("\nSingle trip prediction:")
print("  Number of bikes:", prediction["Score"][0])
