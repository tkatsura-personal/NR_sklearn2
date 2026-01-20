import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
from copy import deepcopy
import pandas as pd
import csv

onset_RevisedOnset = deepcopy(onset[["NR_Name", "sex", "litter_size", "birth_date", "death_date", "rbg200"]])
onset_RevisedOnset["Event"] = (onset_RevisedOnset["rbg200"] < 80).astype(int)
onset_RevisedOnset["Event_time"] = pd.Series(dtype='int64')
# For each row in onset_RevisedOnset, calculate the event time 
for i in range(len(onset_RevisedOnset)):
    if onset_RevisedOnset["Event"].iloc[i]:
        onset_RevisedOnset["Event_time"].iloc[i] = (onset_RevisedOnset["rbg200"].iloc[i] - 4)
    else:
        try:
            onset_RevisedOnset["Event_time"].iloc[i] = round((onset_RevisedOnset["death_date"].iloc[i] - onset_RevisedOnset["birth_date"].iloc[i]).days / 7 - 4)
        except Exception as e:
            ratID = onset_RevisedOnset["NR_Name"].iloc[i]
            #Get max week of that ratID
            max_week = measurement[measurement["NR_Name"] == ratID]["week"].max()
            onset_RevisedOnset["Event_time"].iloc[i] = max_week - 4
mergedData = measurement[["NR_Name", "week", "weight", "rbg", "diet",]].merge(onset_RevisedOnset[["NR_Name", "sex", "Event_time", "Event"]], on="NR_Name")
#Subtract 4 from the week column
mergedData["week"] = mergedData["week"] - 4

with open('onset_longitudinal.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(mergedData.columns)
    for index, row in mergedData.iterrows():
        writer.writerow(row)