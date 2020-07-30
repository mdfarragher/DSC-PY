import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import TypeConverter
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.ensemble import FastTreesBinaryClassifier

# load the file
dataFrame = pd.read_csv("processed.cleveland.data.csv", 
                        sep=',', 
                        names=["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal", "Diagnosis"],
                        header=None)

# convert the final diagnosis code to a boolean value
dataFrame["Diagnosis"] = dataFrame["Diagnosis"].apply(lambda x: 1.0 if x > 0 else 0.0)

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"], result_type = "R4"),
    ColumnConcatenator() << {"Feature":["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"]},
    FastTreesBinaryClassifier() << {Role.Label:"Diagnosis", Role.Feature:"Feature"}
])

# train the model
pipeline.fit(trainData)

# evaluate the model and report metrics
metrics, _ = pipeline.test(testData)

print("\nEvaluation metrics:")
print("  Accuracy:          ", metrics["Accuracy"][0])
print("  Auc:               ", metrics["AUC"][0])
print("  Auprc:             ", metrics["AUPRC"][0])
print("  F1Score:           ", metrics["F1 Score"][0])
print("  LogLoss:           ", metrics["Log-loss"][0])
print("  LogLossReduction:  ", metrics["Log-loss reduction"][0])
print("  PositivePrecision: ", metrics["Positive precision"][0])
print("  PositiveRecall:    ", metrics["Positive recall"][0])
print("  NegativePrecision: ", metrics["Negative precision"][0])
print("  NegativeRecall:    ", metrics["Negative recall"][0])

# set up a patient sample
sample = pd.DataFrame(  [[36.0, 1.0, 4.0, 145.0, 210.0, 0.0, 2.0, 148.0, 1.0, 1.9, 2.0, 1.0, 7.0]],
                            columns = ["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"])

# predict diagnosis for sample
prediction = pipeline.predict(sample)

print("\nSingle patient prediction:")
print("  Diagnosis:  ", "Sick" if prediction["PredictedLabel"][0] == 1.0 else "Healthy")
print("  Probability:", prediction["Probability"][0])
