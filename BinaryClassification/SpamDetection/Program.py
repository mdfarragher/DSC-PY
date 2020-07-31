import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.feature_extraction.text import NGramFeaturizer
from nimbusml.feature_extraction.text.extractor import Ngram
from nimbusml.linear_model import LogisticRegressionBinaryClassifier

# load the file
dataFrame = pd.read_csv("spam.tsv", 
                        sep='\t', 
                        names=["Verdict", "Message"],
                        header=None)

# convert verdict code to a boolean value
dataFrame["Verdict"] = dataFrame["Verdict"].apply(lambda x: 1.0 if x == "spam" else 0.0)

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# build a machine learning pipeline
pipeline = Pipeline([
    NGramFeaturizer(word_feature_extractor = Ngram(weighting = 'Tf')) << ["Message"],
    LogisticRegressionBinaryClassifier() << {Role.Label:"Verdict", Role.Feature:"Message"}
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
sample = pd.DataFrame([ ["Woohoo! I have a date on sunday with melanie from work!!"],
                        ["URGENT! You have won a FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 555123987"], 
                        ["Hi, how's your friday going? Just texting to see if you'd decided to do anything tomorrow?"],
                        ["We tried to call you re your reply to our SMS for a free nokia mobile + free camcorder"]],
                        columns = ["Message"])

# predict diagnosis for sample
prediction = pipeline.predict(sample)

print("\nMessage prediction:")
for i in range(0, len(sample.index)):
    print("  Message:     ", sample["Message"][i])
    print("  Prediction:  ", "Spam" if prediction["PredictedLabel"][i] == 1.0 else "Ham")
    print("  Probability: ", prediction["Probability"][i])
    print()
