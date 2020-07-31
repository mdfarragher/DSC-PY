# Assignment: Predict heart disease risk

In this assignment you're going to build an app that can predict the heart disease risk in a group of patients.

The first thing you will need for your app is a data file with patients, their medical info, and their heart disease risk assessment. We're going to use the famous [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) which has real-life data from 303 patients.

Download the [Processed Cleveland Data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) file and save it as **processed.cleveland.data.csv**.

The data file looks like this:

![Processed Cleveland Data](./assets/data.png)

It’s a CSV file with 14 columns of information:

* Age
* Sex: 1 = male, 0 = female
* Chest Pain Type: 1 = typical angina, 2 = atypical angina , 3 = non-anginal pain, 4 = asymptomatic
* Resting blood pressure in mm Hg on admission to the hospital
* Serum cholesterol in mg/dl
* Fasting blood sugar > 120 mg/dl: 1 = true; 0 = false
* Resting EKG results: 0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes’ criteria
* Maximum heart rate achieved
* Exercise induced angina: 1 = yes; 0 = no
* ST depression induced by exercise relative to rest
* Slope of the peak exercise ST segment: 1 = up-sloping, 2 = flat, 3 = down-sloping
* Number of major vessels (0–3) colored by fluoroscopy
* Thallium heart scan results: 3 = normal, 6 = fixed defect, 7 = reversible defect
* Diagnosis of heart disease: 0 = normal risk, 1-4 = elevated risk

The first 13 columns are patient diagnostic information, and the last column is the diagnosis: 0 means a healthy patient, and values 1-4 mean an elevated risk of heart disease.

You are going to build a binary classification machine learning model that reads in all 13 columns of patient information, and then makes a prediction for the heart disease risk.

Let's get started by creating a new folder for our application:

```bash
$ mkdir HeartDiseasePrediction
$ cd HeartDiseasePrediction
```

If you haven't done so yet, install the NimbusML package:

```bash
$ pip install nimbusml
```

And now launch the Visual Studio Code editor to start building your app:

```bash
$ code Program.py
```

Now you are ready to start coding. You’ll need a couple of import statements:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from nimbusml import Pipeline, Role
from nimbusml.preprocessing.schema import TypeConverter
from nimbusml.preprocessing.schema import ColumnConcatenator
from nimbusml.ensemble import FastTreesBinaryClassifier

# the rest of the code goes here....
```

We'll use **Pandas** DataFrames to import data from CSV files and process it for training. We'll need **Numpy** too because Pandas depends on it. 

And we'll need the **Pipeline**, **Role**, **TypeConverter**, **ColumnConcatenator**, and **FastTreeBinaryClassifier** classes when we start building the machine learning pipeline. We'll do that in a couple of minutes.

Finally, the **train_test_split** function in the **Sklearn** package is very convenient for splitting a single CSV file dataset into a training and testing partition.  

But first, let's load the training data in memory:

```python
# load the file
dataFrame = pd.read_csv("processed.cleveland.data.csv", 
                        sep=',', 
                        names=["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal", "Diagnosis"],
                        header=None)

# convert the final diagnosis code to a boolean value
dataFrame["Diagnosis"] = dataFrame["Diagnosis"].apply(lambda x: 1.0 if x > 0 else 0.0)

# create train and test partitions
trainData, testData = train_test_split(dataFrame, test_size=0.2, random_state=42, shuffle=True)

# the rest of the code goes here...
```

This code calls **read_csv** from the Pandas package to load the CSV data into a new DataFrame. Note the **names=[...]** argument that explicitly provides the column names. We need to do this because the data file does not contain any column headers.

The next line converts the **Diagnosis** column in the data file which is an integer value between 0-4, with 0 meaning 'no risk' and 1-4 meaning 'elevated risk'. 

But we are building a binary classifier which means the model needs to be trained on boolean labels. We need to somehow convert these labels to True and False values.

So what this line does is convert any diagnosis column value greater than zero (meaning elevated risk) to a value of 1.0, and any column value of zero (meaning no risk) to 0.0.

Finally, we call **train_test_split** to set up a training partition with 80% of the data and a test partition with the remaining 20% of the data. Note the **shuffle=True** argument which produces randomized partitions. 

Now you’re ready to start building the machine learning model:

```python
# build a machine learning pipeline
pipeline = Pipeline([
    TypeConverter(columns = ["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"], result_type = "R4"),
    ColumnConcatenator() << {"Feature":["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"]},
    FastTreesBinaryClassifier() << {Role.Label:"Diagnosis", Role.Feature:"Feature"}
])

# train the model
pipeline.fit(trainData)

# the rest of the code goes here....
```
Machine learning models in ML.NET are built with pipelines, which are sequences of data-loading, transformation, and learning components.

This pipeline has the following components:

* A **TypeConverter** that converts the data type of all input columns to **R4** which means a 32-bit floating point number or a single. We need this conversion because Pandas will load floating point data as R8 (64-bit floating point numbers or doubles), and ML.NET cannot deal with that datatype. 
* A **ColumnConcatenator** which combines all input columns into a single column called 'Feature'. This is a required step because ML.NET can only train on a single input column.
* A final **FastTreeBinaryClassification** learner which will analyze the **Feature** column to try and predict the **Diagnosis**.

With the pipeline fully assembled, we can train the model on the training partition by calling the **fit** pipeline function and providing the **trainData** partition.

You now have a fully- trained model. So next, you'll have to grab the test data, predict the diagnosis for every patient, and calculate the accuracy of your model:

```python
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

# the rest of the code goes here....
```

This code calls the **test** pipeline function and provides the **testData** partition to generate predictions for every single taxi trip in the test partition and compare them to the actual diagnoses. 

The function will automatically calculate the following metrics:

* **Accuracy**: this is the number of correct predictions divided by the total number of predictions.
* **AUC**: a metric that indicates how accurate the model is: 0 = the model is wrong all the time, 0.5 = the model produces random output, 1 = the model is correct all the time. An AUC of 0.8 or higher is considered good.
* **AUPRC**: an alternate AUC metric that performs better for heavily imbalanced datasets with many more negative results than positive.
* **F1Score**: this is a metric that strikes a balance between Precision and Recall. It’s useful for imbalanced datasets with many more negative results than positive.
* **LogLoss**: this is a metric that expresses the size of the error in the predictions the model is making. A logloss of zero means every prediction is correct, and the loss value rises as the model makes more and more mistakes.
* **LogLossReduction**: this metric is also called the Reduction in Information Gain (RIG). It expresses the probability that the model’s predictions are better than random chance.
* **PositivePrecision**: also called ‘Precision’, this is the fraction of positive predictions that are correct. This is a good metric to use when the cost of a false positive prediction is high.
* **PositiveRecall**: also called ‘Recall’, this is the fraction of positive predictions out of all positive cases. This is a good metric to use when the cost of a false negative is high.
* **NegativePrecision**: this is the fraction of negative predictions that are correct.
* **NegativeRecall**: this is the fraction of negative predictions out of all negative cases.

When monitoring heart disease, you definitely want to avoid false negatives because you don’t want to be sending high-risk patients home and telling them everything is okay.

You also want to avoid false positives, but they are a lot better than a false negative because later tests would probably discover that the patient is healthy after all.

To wrap up, You’re going to create a new patient record and ask the model to make a prediction:

```python
# set up a patient sample
sample = pd.DataFrame(  [[36.0, 1.0, 4.0, 145.0, 210.0, 0.0, 2.0, 148.0, 1.0, 1.9, 2.0, 1.0, 7.0]],
                            columns = ["Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"])

# predict diagnosis for sample
prediction = pipeline.predict(sample)

print("\nSingle patient prediction:")
print("  Diagnosis:  ", "Sick" if prediction["PredictedLabel"][0] == 1.0 else "Healthy")
print("  Probability:", prediction["Probability"][0])
```

This code sets up a new **DataFrame** with the details of the patient. Note that I have to provide the data and the column names separately. 

Next, we call the **predict** pipeline function to predict the diagnosis for this patient. The resulting dataframe has three columns:

* A **PredictedLabel** column with the predicted diagnosis: 1.0 for a sick patient and 0.0 for a healthy patient.
* A **Score** column with the predicted score. This is the regression value produced by the binary classifier before thresholding. Note that we don't use this column in our code.
* A **Probability** column with the probability of the prediction. This can be interpreted as the level of confidence the machine learning algorithm has in its prediction.

So what’s the model going to predict?

Time to find out. Go to your terminal and run your code:

```bash
$ python ./Program.py
```

What results do you get? What is your accuracy, precision, recall, AUC, AUCPRC, and F1 value?

Is this dataset balanced? Which metrics should you use to evaluate your model? And what do the values say about the accuracy of your model? 

And what about our patient? What did your model predict?

Think about the code in this assignment. How could you improve the accuracy of the model? What are your best AUC and AUCPRC values? 

Share your results in our group!
