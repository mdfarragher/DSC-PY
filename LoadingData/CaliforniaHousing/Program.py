import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# load the file
dataFrame = pd.read_csv("california_housing.csv", 
                        sep=',', 
                        header=0)

# keep only records with median house values < 500,000
dataFrame = dataFrame[dataFrame.median_house_value < 500000]

# # plot median house value by median income
# plt.xlabel("median income")
# plt.ylabel("median house value")
# plt.scatter(dataFrame.median_income, dataFrame.median_house_value)
# plt.show()

# convert the house value range to thousands
dataFrame.median_house_value /= 1000

# show the transformed data
print(dataFrame)

# # plot median house value by longitude
# plt.xlabel("longitude")
# plt.ylabel("median house value")
# plt.scatter(dataFrame.longitude, dataFrame.median_house_value)
# plt.show()

# bin and one-hot encode longitude
for r in zip(range(-124, -114), range(-123, -113)):
    dataFrame[f"longitude_{r}"] = dataFrame["longitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

# bin and one-hot encode latitude
for r in zip(range(32, 42), range(33, 43)):
    dataFrame[f"latitude_{r}"] = dataFrame["latitude"].apply(
      lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)

# # show the transformed data
# print(dataFrame.info())
# print(dataFrame)

# cross the longitude and latitude
for lo in zip(range(-124, -114), range(-123, -113)):
    for la in zip(range(32, 42), range(33, 43)):
        dataFrame[f"cross_{lo}x{la}"] = dataFrame[f"longitude_{lo}"] * dataFrame[f"latitude_{la}"]

# show the transformed data
print(dataFrame.info())
print(dataFrame)
