# Instruction to run ROC curve on Huishi's computer

First, the pyread.py file is necessary since it is used to read the accdb file content.
Additionally, you need to get the accdb file and put it in the main folder along with the file you want to run and the pyread.py file.

## Scripts to copy-paste (details explained below)
cd ~

cd ./Documents

source python_env/bin/activate

cd ./python_rats

python3.12 RF_ROC_curve.py

## Explantion
### cd ~ 
cd is short for "change directory," navigating you to the said directory.

~ is the default directory (usually the <username> folder that leads you to every important folder).

### cd ./Documents
"." implies the folder you are currently sitting in, and ./Documents says "go to the Documents folder that is inside your current directory folder

### source python_env/bin/activate
Command to setup the environment. Not necessary but it's better to setup environments for different purpose because loading packages takes time, and you want to minimize the # of packages to load.

type "deactivate" to deactivate/leave the environment

### cd ./python_rats
You can still navigate while inside the environment.

### python3.12 DT_ROC_curve.py
python3.12 is the python version to execute the file, and DT_ROC_curve.py can be replaced for any python file (usually ends in .py to indicate python files).

It is possible to run python3.12 ./python_rats/DT_ROC_curve.py and skip the above line, but here I need it to read the main, measurement, and onset.csv files in the same folder so I navigate to the folder first.
