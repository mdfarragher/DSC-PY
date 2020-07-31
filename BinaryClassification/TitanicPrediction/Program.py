import pandas as pd
import numpy as np
import math

from nimbusml import Pipeline, Role
from nimbusml.preprocessing.missing_values import Handler
from nimbusml.feature_extraction.categorical import OneHotVectorizer
from nimbusml.preprocessing.schema import TypeConverter, ColumnConcatenator
from nimbusml.ensemble import FastTreesBinaryClassifier

# load the data files
trainData = pd.read_csv("train_data.csv", sep=',') 
testData = pd.read_csv("test_data.csv", sep=',') 

# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"], result_type = "R4"),
    Handler(replace_with = "Mean") << ["Age"],
    OneHotVectorizer() << ["Sex", "Ticket", "Cabin", "Embarked"],
    ColumnConcatenator() << {"Feature":["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Ticket", "Cabin", "Embarked"]},
    FastTreesBinaryClassifier() << {Role.Label:"Survived", Role.Feature:"Feature"}
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

# set up a sample
sample = pd.DataFrame(  [[1.0, "male", 48, 0.0, 0.0, "B", 70.0, "123", "S"]],
                            columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])

# predict diagnosis for sample
prediction = pipeline.predict(sample)

print("\nSingle prediction:")
print("  Prediction:  ", "Survived" if prediction["PredictedLabel"][0] == 1.0 else "Perished")
print("  Probability: ", prediction["Probability"][0])
