# Estimating Above-Ground Biomass in the Finish Forests using satelite imagery and machine learning

Exam project in the course "Geospatial Data Science" Spring 2023. Project was developed by Aske Schytt Meineche and Viktor Due Pedersen.

## Training folder

Training this model is a process that takes a long time. Therefore, we have included the trained model in the repository under `model_outputs`. For the same reason, we haven't included any data but instead provided instructions on how to download the data used in this project in `biomassters-download-instructions.txt`. In the folder `large_sample`we have included a sample of the data used in this project. Note that training can be done by running the `train_CNN.py` script. This will train the model and save it in the `model_outputs` folder, but it will take a long time, even on the HPC.

## Presentation_notebook

This is the file that takes care of all the evaluation and presentation of the results. It is a jupyter notebook that can be run from top to bottom. It will then produce all the figures and tables used in the report.

## A note on files

Some of the files in this repository are copies of files from other folders in this repository. This is due to the fact that we wanted the `training` folder to be self-contained.
