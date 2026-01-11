# Research Question : Do Space Project Themes Influence Funding? A Machine Learning Approach Using K-Means Clustering and Regression Model Comparison

Author Information:
Name: Maia Mailander
Student Number: 25434101
Course: Data Science and Advanced Programming 2026

## Setup

# Create environment
conda env create -f environment.yml
conda activate final-project

# Project structure
final-project
- data/processed
- results
- src
-- __init__.py
-- data_loader.py
-- evaluation.py
-- feature_engineering.py
-- models.py
-- text_processor.py
- .gitignore
- AI_USAGE.md
- environment.yml
- main.py
- README.md

# Running the Pipeline
python main.py

# Expected output : Thematic cluster analysis, visualization tables in the results folder and comparison of regression models

# Regression Results
- Low R^2 across all modelss indicates we cannot use thematic categories to predict funding amounts, although preferences for certain themes clearly exist at the aggregate level.

# Requirements
Python3.11
scikit-learn, pandas, matplotlib, seaborn, nltk, openpyxl and other packages in environment.yml

