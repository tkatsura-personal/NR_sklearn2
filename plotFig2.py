import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime

# Filter onset such that overall_diet is Rod or New
R_onset = deepcopy(onset[(((onset.overall_diet == "Rod") & (onset.weanling_diet == "Rod") & (onset.nursing_diet == "Rod") &
                           (onset.gestational_diet == "Rod")) & #overall diet must be rodent
                               (((onset['death_date'] - onset['birth_date']) >= datetime.timedelta(days=280)) |
                                (onset['rbg200'] <= 40)))])

# For each sex...
# Grab NR_Name and rbg200 from onset table
R_onset['rbg200'] = R_onset['rbg200'].where(R_onset['rbg200'] <= 40, 80)
onset_sex = deepcopy(R_onset[['NR_Name', 'sex', 'rbg200']])

# Group by rbg200 and count occurrences
onset_counts = onset_sex.groupby(['sex', 'rbg200']).size().reset_index(name='count')
onset_counts = onset_counts.sort_values(['sex', 'rbg200'])
onset_counts['cumulative'] = onset_counts.groupby('sex')['count'].cumsum()

max_cumulative = onset_counts.groupby('sex')['cumulative'].transform('max')
onset_counts['percent'] = onset_counts['cumulative'] / max_cumulative * 100
print(onset_counts)

# Create a pivot table for plotting
pivot = onset_counts.pivot(index='rbg200', columns='sex', values='percent').fillna(0)
weeks = pivot.index
x = np.array(pivot.index.to_list())

# Plot the data
plt.figure(figsize=(12, 6))
plt.bar(x - 0.75, pivot.M.to_list(), width=1.5, label='Male', color=(181/256, 215/256, 228/256), hatch="//", edgecolor='black')
plt.bar(x + 0.75, pivot.F.to_list(), width=1.5, label='Female', color=(245/256, 194/256, 203/256), hatch="//", edgecolor='black')

# x and y axis formatting
plt.xlabel('Week')
plt.ylabel('Cumulative Percentage')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.xlim(4, 44)
plt.xticks(np.arange(8, 44, 4))
plt.ylim(0, 100)
plt.yticks(np.arange(0, 100, 25))
plt.show()