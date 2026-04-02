from copy import deepcopy
import sys, os, csv
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pyread import measurement, onset

onset_RevisedOnset = deepcopy(onset[["NR_Name", "birth_date", "rbg200"]])
measurementData = deepcopy(measurement[["NR_Name", "week", "weight", "rbg", "diet"]])

# Load the list of user IDs to keep (single column CSV)
RatToGrab = pd.read_csv("./20260327_GrabRat/RatToGrab.csv")

weightList = ["weight" + str(n) for n in range(4,84,4)]
rbgList = ["rbg" + str(n) for n in range(4,84,4)]
dietList = ["diet" + str(n) for n in range(4,84,4)]

# Grab the first (and only) column as a list
RatList = RatToGrab.iloc[:, 0].tolist()
filteredMeasurement = measurementData[measurementData["NR_Name"].isin(RatList)]
filteredMeasurementLong = filteredMeasurement[["NR_Name", "week", "weight", "rbg", "diet"]].pivot(
    index="NR_Name", columns="week", values=["weight", "rbg", "diet"])
filteredMeasurementWeight = filteredMeasurement[["NR_Name", "week", "weight"]].pivot(
    index="NR_Name", columns="week", values="weight")
filteredMeasurementRBG = filteredMeasurement[["NR_Name", "week", "rbg"]].pivot(
    index="NR_Name", columns="week", values="rbg")
filteredMeasurementDiet = filteredMeasurement[["NR_Name", "week", "diet"]].pivot(
    index="NR_Name", columns="week", values="diet")

filteredMeasurementAll = filteredMeasurement[["NR_Name"]].drop_duplicates().merge(
    pd.concat([filteredMeasurementWeight, filteredMeasurementRBG, filteredMeasurementDiet], axis = 0),
    left_on="NR_Name", right_index=True)

filteredMeasurementLong.columns = weightList + rbgList + dietList
fML_Names = filteredMeasurement[["NR_Name"]].drop_duplicates().merge(filteredMeasurementLong, left_on="NR_Name", right_index=True)

filtered = onset_RevisedOnset[onset_RevisedOnset["NR_Name"].isin(RatList)].merge(fML_Names, left_on="NR_Name", right_on="NR_Name")
# Save the result to a new CSV
filtered.to_csv("./20260327_GrabRat/filteredRatRecord.csv", index=False)
filteredMeasurementAll.to_csv("./20260327_GrabRat/filteredMeasurementLong.csv", index=False)