import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from copy import deepcopy
import datetime

# Setup a figure to plot the box plots
plt.figure(figsize=(6, 6))
flierSetting = dict(marker='o', markersize=4, markeredgecolor='none', markerfacecolor='black', linestyle='None')
medianSetting = dict(linestyle='-', linewidth=1, color='black')


for sex in ["M", "F"]:
    # Print sex being processed
    print(f"Processing {"Male" if sex == "M" else "Female"}")
    diabetesRatList = onset[(((onset.sex == sex) & (onset.overall_diet == "Rod") &
                              (onset.weanling_diet == "Rod") & (onset.nursing_diet == "Rod")) & #overall diet must be rodent
                   (((onset['sex'] == "M") & (onset['rbg200'] <= 16)) |
                    ((onset['sex'] == "F") & (onset['rbg200'] <= 36))))]["NR_Name"].to_list()

    noDiabetesRatList = onset[(((onset.sex == sex) & (onset.overall_diet == "Rod") & 
                                (onset.weanling_diet == "Rod") & (onset.nursing_diet == "Rod")) & #overall diet must be rodent
                    (((onset['sex'] == "M") & ((onset['death_date'] - onset['birth_date']) >= datetime.timedelta(days=7*16)) & (onset['rbg200'] > 16)) |
                     ((onset['sex'] == "F") & ((onset['death_date'] - onset['birth_date']) >= datetime.timedelta(days=7*36)) & (onset['rbg200'] > 36))))]["NR_Name"].to_list()

    measurementD = measurement[measurement["NR_Name"].isin(diabetesRatList) & (measurement["week"].isin([4,8]))][["NR_Name", "week", "weight"]].dropna(subset=["weight"])
    measurementND = measurement[measurement["NR_Name"].isin(noDiabetesRatList) & (measurement["week"].isin([4,8]))][["NR_Name", "week", "weight"]].dropna(subset=["weight"])

    # Pivot the 4 and 8 week data
    measurementD = measurementD.pivot(index="NR_Name", columns="week", values="weight").reset_index().dropna(subset=[4, 8])
    measurementND = measurementND.pivot(index="NR_Name", columns="week", values="weight").reset_index().dropna(subset=[4, 8])
    
    # Add a new column for the percent increase from 8 to 4 weeks
    measurementD["percent_increase"] = ((measurementD[8] - measurementD[4]) / measurementD[8] * 100).fillna(0)
    measurementND["percent_increase"] = ((measurementND[8] - measurementND[4]) / measurementND[8] * 100).fillna(0)

    # Print boxplot data
    print(f"Diabetes: {len(measurementD['NR_Name'].unique())} rats")
    print(f"No Diabetes: {len(measurementND['NR_Name'].unique())} rats")

    t_statistic, p_value = stats.ttest_ind(measurementD["percent_increase"], measurementND["percent_increase"], equal_var=False)

    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Check if there are any overlaps between the two lists
    overlap = set(measurementD["NR_Name"]).intersection(set(measurementND["NR_Name"]))
    if overlap:
        print(f"Overlap found: {overlap}")
    else:
        print("No overlap found between diabetes and no diabetes rat lists.")

    # Plot the box plot for each group
    plt.boxplot(measurementD["percent_increase"], positions=[(1 if sex == "M" else 2)+(0.175)], widths=0.25, patch_artist=True,
                flierprops=flierSetting, medianprops=medianSetting, boxprops=dict(facecolor=(189/255, 68/255, 30/255), color = "black"))
    plt.boxplot(measurementND["percent_increase"], positions=[(1 if sex == "M" else 2)-(0.175)], widths=0.25, patch_artist=True,
                flierprops=flierSetting, medianprops=medianSetting, boxprops=dict(facecolor=(151/255, 204/255, 232/255), color = "black"))
    
    # Set x-ticks and labels
    plt.xlabel("")
    plt.ylabel("% Growth")
    plt.xlim(0.5, 2.5)
    plt.xticks(ticks = np.arange(1, 3, 1), labels=["Male", "Female"] )
plt.show()
