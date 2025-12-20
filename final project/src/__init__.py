# src/__init__.py
"""
NASA SBIR Thematic Funding Analysis

This package contains modules for analyzing NASA SBIR award data
to identify thematic patterns in funding allocation.

Modules:
--------
data_loader
    Load raw Excel data, filter to NASA, clean and save
    
text_processor
    NLP preprocessing: lowercase, stop words, lemmatization
    
feature_engineering
    TF-IDF vectorization and categorical feature creation
    
models
    K-Means clustering and regression analysis

Author: Maia Mailander
Course: Introduction to Data Science and Advanced Programming
"""

__version__ = "1.0.0"
__author__ = "Maia Mailander"