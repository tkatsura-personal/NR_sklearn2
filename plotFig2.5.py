import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import onset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import datetime

# Filter onset such that overall_diet is Rod or New
R_onset_rod = deepcopy(onset[(((onset.overall_diet == "Rod") & (onset.weanling_diet == "Rod") & (onset.nursing_diet == "Rod") &
                           (onset.gestational_diet == "Rod")) & #overall diet must be rodent
                               (((onset['death_date'] - onset['birth_date']) >= datetime.timedelta(days=280)) |
                                (onset['rbg200'] <= 40)))])

# For each sex...
# Grab NR_Name and rbg200 from onset table
R_onset_rod['rbg200'] = R_onset_rod['rbg200'].where(R_onset_rod['rbg200'] <= 40, 80)
onset_sex_rod = deepcopy(R_onset_rod[['NR_Name', 'sex', 'rbg200']])

# Rod diet group
# Group by rbg200 and count occurrences
onset_counts_rod = onset_sex_rod.groupby(['sex', 'rbg200']).size().reset_index(name='count')
onset_counts_rod = onset_counts_rod.sort_values(['sex', 'rbg200'])
onset_counts_rod['cumulative'] = onset_counts_rod.groupby('sex')['count'].cumsum()

max_cumulative_rod = onset_counts_rod.groupby('sex')['cumulative'].transform('max')
onset_counts_rod['percent'] = onset_counts_rod['cumulative'] / max_cumulative_rod * 100
onset_counts_rod['nursing'] = 'Rod'

# Create a pivot table for plotting
pivot_rod = onset_counts_rod.pivot(index='rbg200', columns='sex', values='percent').fillna(0)
pivot_rod.nursing = 'Rod'

# Do the same for rab diet group
R_onset_rab = deepcopy(onset[(((onset.overall_diet == "Rod") & (onset.weanling_diet == "Rod") & (onset.nursing_diet == "Rab") &
                           (onset.gestational_diet == "Rab")) &
                               (((onset['death_date'] - onset['birth_date']) >= datetime.timedelta(days=280)) |
                                (onset['rbg200'] <= 40)))])

# For each sex...
# Grab NR_Name and rbg200 from onset table
R_onset_rab['rbg200'] = R_onset_rab['rbg200'].where(R_onset_rab['rbg200'] <= 40, 80)
onset_sex_rab = deepcopy(R_onset_rab[['NR_Name', 'sex', 'rbg200']])

# Group by rbg200 and count occurrences
onset_counts_rab = onset_sex_rab.groupby(['sex', 'rbg200']).size().reset_index(name='count')
onset_counts_rab = onset_counts_rab.sort_values(['sex', 'rbg200'])
onset_counts_rab['cumulative'] = onset_counts_rab.groupby('sex')['count'].cumsum()

max_cumulative_rab = onset_counts_rab.groupby('sex')['cumulative'].transform('max')
onset_counts_rab['percent'] = onset_counts_rab['cumulative'] / max_cumulative_rab * 100
onset_counts_rab['nursing'] = 'Rab'

# Merge onset_counts_rod and onset_counts_rab
onset_counts_combined = pd.concat([onset_counts_rod, onset_counts_rab], ignore_index=True)
# Remove rbg200 = 4
onset_counts_combined = onset_counts_combined[onset_counts_combined['rbg200'] != 4]

# Split onset_counts_combined into two tables by sex
onset_counts_M = onset_counts_combined[onset_counts_combined['sex'] == 'M']
onset_counts_F = onset_counts_combined[onset_counts_combined['sex'] == 'F']
print(onset_counts_M)
print(onset_counts_F)

# Create a pivot table for plotting
pivot_M = onset_counts_M.pivot(index='rbg200', columns='nursing', values='percent').fillna(0)
pivot_F = onset_counts_F.pivot(index='rbg200', columns='nursing', values='percent').fillna(0)
weeks = pivot_F.index.to_list()
print(pivot_M)
print(pivot_F)
# Male and female
fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = [6, 6], figsize=(12, 10))

# Use plt.bar() for grouped bar charts (Green for Rab, Orange for Rod)
ax[0].bar(x = [w + 0.75 for w in weeks], height=pivot_M.Rab.to_list(), width = 1.5, color=(120/256, 166/256, 107/256), label='High-fiber maternal diet', hatch="//", edgecolor='black')
ax[0].bar(x = [w - 0.75 for w in weeks], height=pivot_M.Rod.to_list(), width = 1.5, color=(255/256, 166/256, 28/256), label='Regular chow maternal diet', hatch="//", edgecolor='black')
ax[0].set_ylabel('Cumulative Percentage - Male')
ax[0].legend()
ax[0].grid(axis='y')
ax[0].set_xlim(4, 44)
ax[0].set_xticks(np.arange(8, 44, 4))
ax[0].set_ylim(0, 100)
ax[0].set_yticks(np.arange(0, 100, 25))

ax[1].bar(x = [w + 0.75 for w in weeks], height=pivot_F.Rab.to_list(), width = 1.5, color=(120/256, 166/256, 107/256), label='High-fiber maternal diet', hatch="//", edgecolor='black')
ax[1].bar(x = [w - 0.75 for w in weeks], height=pivot_F.Rod.to_list(), width = 1.5, color=(255/256, 166/256, 28/256), label='Regular chow maternal diet', hatch="//", edgecolor='black')
ax[1].set_xlabel('Week')
ax[1].set_ylabel('Cumulative Percentage - Female')
ax[1].legend()
ax[1].grid(axis='y')
ax[1].set_xlim(4, 44)
ax[1].set_xticks(np.arange(8, 44, 4))
ax[1].set_ylim(0, 100)
ax[1].set_yticks(np.arange(0, 100, 25))

plt.tight_layout()
plt.show()
"""
# Rab diet group
plt.bar(x - 0.75, pivot_M.Rab.to_list(), width=1.5, label='High-fiber', color=(120/256, 166/256, 107/256), hatch="//", edgecolor='black')
plt.bar(x + 0.75, pivot_M.Rod.to_list(), width=1.5, label='High-fiber', color=(245/256, 194/256, 203/256), hatch="//", edgecolor='black')
# Rod diet group
#plt.bar(x - 0.75, pivot_rab.M.to_list(), width=1.5, label='High-Fiber', color=(120/256, 166/256, 107/256), hatch="//", edgecolor='black')

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
plt.show()"""