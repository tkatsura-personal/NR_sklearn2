import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import datetime

# Setup 2 figure to plot the box plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
flierSetting = dict(marker='o', markersize=4, markeredgecolor='none', markerfacecolor='black', linestyle='None')
medianSetting = dict(linestyle='-.', linewidth=1, color='black')


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

    measurementD = measurement[measurement["NR_Name"].isin(diabetesRatList) & (measurement["week"] % 4 == 0) & (measurement["week"] <= 40)][["NR_Name", "week", "weight"]].dropna(subset=["weight"])
    measurementND = measurement[measurement["NR_Name"].isin(noDiabetesRatList) & (measurement["week"] % 4 == 0) & (measurement["week"] <= 40)][["NR_Name", "week", "weight"]].dropna(subset=["weight"])

    # Print shape of measurementD and measurementND
    print(f"Diabetes: {len(measurementD['NR_Name'].unique())} rats, {len(measurementD['weight'].dropna())} measurements")
    print(f"No Diabetes: {len(measurementND['NR_Name'].unique())} rats, {len(measurementND['weight'].dropna())} measurements")

    # Check if there are any overlaps between the two lists
    overlap = set(measurementD["NR_Name"]).intersection(set(measurementND["NR_Name"]))
    if overlap:
        print(f"Overlap found: {overlap}")
    else:
        print("No overlap found between diabetes and no diabetes rat lists.")

    # Group by week
    measurementD = measurementD.groupby("week")["weight"].apply(list).reset_index()
    measurementND = measurementND.groupby("week")["weight"].apply(list).reset_index()
    pos = np.array(measurementD["week"])

    # Plot the box plot for each group
    axs[(0 if sex == "M" else 1)].boxplot(measurementD["weight"], positions=pos+0.75, widths=1.25, patch_artist=True,
                flierprops=flierSetting, medianprops=medianSetting, boxprops=dict(facecolor=(189/255, 68/255, 30/255), color = "black"))
    axs[(0 if sex == "M" else 1)].boxplot(measurementND["weight"], positions=pos-0.75, widths=1.25, patch_artist=True,
                flierprops=flierSetting, medianprops=medianSetting, boxprops=dict(facecolor=(151/255, 204/255, 232/255), color = "black"))
    for i in range(len(measurementD)):
        print(f"Week: {measurementD["week"][i]}")
        t_statistic, p_value = stats.ttest_ind(measurementD["weight"][i], measurementND["weight"][i], equal_var=False)

        print(f"T-statistic: {t_statistic}")
        print(f"P-value: {p_value}")
    # Set x-ticks and labels
    for ax in axs:
        ax.set_xlabel("Age (weeks)")
        ax.set_ylabel("Weight (g)")
        ax.set_xlim(0, 44)
        ax.set_xticks(np.arange(4, 44, 4))
        ax.set_xticklabels(measurementD["week"])
plt.show()