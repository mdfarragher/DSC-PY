import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import TypeConverter
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.ensemble import FastTreesRegressor

# load the file
dataFrame = pd.read_csv("yellow_tripdata_2018-12.csv", 
                        sep=',', 
                        header=0)

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["passenger_count", "trip_distance"], result_type = "R4"),
    OneHotVectorizer() << ["VendorID", "RatecodeID", "payment_type"],
    ColumnConcatenator() << {"Feature":["VendorID", "RatecodeID", "payment_type", "passenger_count", "trip_distance"]},
    FastTreesRegressor() << {Role.Label:"total_amount", Role.Feature:"Feature"}
])

# train the model
pipeline.fit(trainData)

# evaluate the model and report metrics
metrics, _ = pipeline.test(testData)
print("\nEvaluation metrics:")
print("  RMSE: ", metrics["RMS(avg)"][0])
print("  MSE: ", metrics["L2(avg)"][0])
print("  MAE: ", metrics["L1(avg)"][0])

# set up a trip sample
tripSample = pd.DataFrame(  [[1, 1, 1, 1.0, 3.75]],
                            columns = ["VendorID", "RatecodeID", "payment_type", "passenger_count", "trip_distance"])

# predict fare for trip sample
prediction = pipeline.predict(tripSample)
print("\nSingle trip prediction:")
print("  Fare:", prediction["Score"][0])
