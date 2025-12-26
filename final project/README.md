# Clustering and regression using Machine Learning : Space Sector Investment

# Research Question : Which regression model — Linear, Ridge, Random Forest, or Gradient Boosting — best predicts NASA SBIR project funding amounts from thematic clusters identified through K-Means?

## Setup

# Create environment
conda env create -f environment.yml
conda activate final-project

# Usage
python main.py

# Expected output : Thematic cluster analysis and comparison of regression models

# Project structure
final-project
- data
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

# Results
Linear Regression 
Ridge Regression
Random Forest
Gradient Boosting
- Low R^2 indicates thematic content cannot predict individual award amounts, akthough aggregate differences are clear across themes.

# Requirements
Python3.11
scikit-learn, pandas, matplotlib, seaborn, nltk, openpyxl

