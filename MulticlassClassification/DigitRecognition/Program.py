import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing import FromKey, ToKey
from nimbusml.preprocessing.schema import TypeConverter, ColumnConcatenator
from nimbusml.ensemble import LightGbmClassifier

# load the train and test files
trainData = pd.read_csv("mnist_train.csv", sep=',')
testData = pd.read_csv("mnist_test.csv", sep=',')

# get a list of pixel column names
pixelColumns = trainData.columns[1:785].values.tolist()

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns=pixelColumns, result_type = "R4"),
    ColumnConcatenator() << {"Feature":pixelColumns},
    LightGbmClassifier() << {Role.Label:"label", Role.Feature:"Feature"}
])

# train the model
pipeline.fit(trainData)

# evaluate the model and report metrics
metrics, _ = pipeline.test(testData)

print("\nEvaluation metrics:")
print("  MicroAccuracy:    ", metrics["Accuracy(micro-avg)"][0])
print("  MacroAccuracy:    ", metrics["Accuracy(macro-avg)"][0])
print("  LogLoss:          ", metrics["Log-loss"][0])
print("  LogLossReduction: ", metrics["Log-loss reduction"][0])

# set up a sample
sample = testData.sample(n=5)

# predict diagnosis for sample
prediction = pipeline.predict(sample)

# merge the sample and the predictions
sample.reset_index(drop=True, inplace=True)
prediction.reset_index(drop=True, inplace=True)
results = pd.concat([sample["label"], prediction], axis=1)

# print results
pd.options.display.float_format = "{:,.4f}".format
print("\nSingle digit predictions:")
print(results)
