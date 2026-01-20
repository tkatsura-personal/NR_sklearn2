# -*- coding: utf-8 -*-
"""This used to be a file
"""

import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import auc, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from tkinter import Tk, messagebox, StringVar, IntVar, Checkbutton, Button, Label, OptionMenu

week_cutoff_male = 0
week_cutoff_female = 0

root = Tk()

root.title("Model creation")
root.geometry('400x600+600+200')

weekOptions = range(12,44,4)

maleLabel = Label(root, text="Male", fg = "black")
maleLabel.place(x = 100, y = 20, anchor = "center")
maleWeek = StringVar(root)
maleWeek.set("Select week") # default value
maleChoice = OptionMenu(root, maleWeek, *weekOptions)
maleChoice.place(x = 100, y = 50, anchor = "center")

femaleLabel = Label(root, text="Female", fg = "black")
femaleLabel.place(x = 300, y = 20, anchor = "center")
femaleWeek = StringVar(root)
femaleWeek.set("Select week") # default value
femaleChoice = OptionMenu(root, femaleWeek, *weekOptions)
femaleChoice.place(x = 300, y = 50, anchor = "center")

def show_selection():
    allOptions = []
    if ((maleWeek.get() == "Select week") | (femaleWeek.get() == "Select week")):
        print("Please select a week")
        return
    else: 
        global week_cutoff_male
        week_cutoff_male = maleWeek.get()
        global week_cutoff_female
        week_cutoff_female = femaleWeek.get()
        for name, var in check_vars.items():
            if var.get() == 1: # For each selected option, append it to the list
                allOptions.append(name)
        if not allOptions:
            print("Please select at least one option")
            return
        else:
            message = f"Selected options: {allOptions}\nMale week: {week_cutoff_male} Female week: {week_cutoff_female}"
            if messagebox.askokcancel("Confirm Selection", message):
                print("User confirmed")
                global feature_sub
                feature_sub = allOptions
                root.destroy() # Close the GUI window
            else:
                print("User cancelled")
            return

        
check_vars = {}
options = {"Gestational Diet":'gestational_diet', "Nursing Diet":'nursing_diet', 
           "Weight: week 4":'wt4', "Weight: week 8":'wt8', "Weight: week 12":'wt12', 
           "RBG Value: week 4":'rbg4', "RBG Value: week 8":'rbg8', "RBG Value: week 12":'rbg12'}
optionCoordinates = [(200, 100), (200, 125), (100, 150), (100, 175), (100, 200), (300, 150), (300, 175), (300, 200)]

for text, name in options.items():
    var = IntVar()
    check_vars[name] = var
    check_button = Checkbutton(root, text=text, variable=var)
    check_button.place(x = optionCoordinates[list(options.keys()).index(text)][0],
                        y = optionCoordinates[list(options.keys()).index(text)][1], anchor = "center")

startModelCreation = Button(root, text = "Start model creation", fg = "black",
                            command = show_selection)
startModelCreation.place(x = 200, y = 575, anchor = "center")

root.mainloop()
