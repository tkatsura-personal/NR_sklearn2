import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns

# Filter onset such that overall_diet is Rod
dietMatch = deepcopy(onset[((onset.overall_diet == "Rod") | (onset.weanling_diet == "Rab")) & (onset.weanling_diet == onset.overall_diet)]["NR_Name"])
summaryrats = deepcopy(measurement[measurement["NR_Name"].isin(dietMatch) & (measurement["diet"].isin(["Rod", "Rab"]))][["NR_Name", "rbg", "diet"]])
print(summaryrats)

fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = [6, 3])

sns.histplot(summaryrats, binwidth=10, x="rbg", hue="diet", alpha=0.7, label="Summary Rats", multiple="dodge", 
             stat="density", common_norm=False, kde=True, ax = ax[0])
sns.boxplot(summaryrats, x="rbg", hue="diet", ax = ax[1])

plt.show()