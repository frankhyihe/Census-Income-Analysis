# Income Classification & Segmentation Project

## Overview
This project analyzes U.S. census data to:

1. **Classify income**: Predict whether a person earns more than $50,000 per year.  
2. **Data augmentation**: Use **CTGAN** to handle class imbalance for high-income individuals.  
3. **Segmentation**: Group the population into clusters for marketing insights.

**Key features**:  
- 40 demographic and employment variables.  
- Weighted samples reflecting population distribution.  
- Models: Random Forest, Logistic Regression, XGBoost, Multi-Layer Perceptron (MLP).  
- Clustering: K-Means segmentation with profile visualizations.  

---

## Project Structure
project/
│
├─ census-bureau.data # Raw data
├─ census-bureau.columns # Column headers
├─ census.ipynb # Preprocessing, model training, evaluation for classifiers
├─ segmentation.ipynb # Clustering and profiling for segmentation
├─ requirements.txt # Python dependencies
├─ README.md # Instructions & overview

## Prerequisites

- **Python** 3.10+  
- **Dependencies** (install with `pip install -r requirements.txt`):

