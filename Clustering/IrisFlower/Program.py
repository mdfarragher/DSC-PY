import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import TypeConverter
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.cluster import KMeansPlusPlus

# load the file
dataFrame = pd.read_csv("iris-data.csv", 
                        sep=',', 
                        header=None,
                        names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Label"])

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], result_type = "R4"),
    ColumnConcatenator() << {"Feature":["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]},
    KMeansPlusPlus(n_clusters = 3, feature = ["Feature"])
])

# train the model
pipeline.fit(trainData)

# evaluate the model and report metrics
metrics, _ = pipeline.test(testData, "Label")
print("\nEvaluation metrics:")
print("  Normalized Mutual Information: ", metrics["NMI"][0])
print("  Average distance to centroid:  ", metrics["AvgMinScore"][0])

# set up a sample
sample = testData.sample(n=5)

# create predictions for sample
prediction = pipeline.predict(sample)

# merge the sample and the predictions
sample.reset_index(drop=True, inplace=True)
prediction.reset_index(drop=True, inplace=True)
results = pd.concat([sample["Label"], prediction], axis=1)

# print results
pd.options.display.float_format = "{:,.4f}".format
print("\nSingle flower predictions:")
print(results)
