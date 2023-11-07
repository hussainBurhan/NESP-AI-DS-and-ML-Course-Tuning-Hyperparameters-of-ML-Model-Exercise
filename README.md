# Heart Disease Prediction
This repository contains code for training and evaluating different Random Forest models for predicting heart disease.

## Overview
The code provided includes:

- Loading and preprocessing the heart disease dataset.
- Training multiple Random Forest models with different hyperparameters.
- Evaluating the models using various classification metrics.
- Performing hyperparameter tuning using Randomized Search CV and Grid Search CV.

## Learning Outcomes

By working on this project, you will:

- Gain experience in handling and preprocessing real-world datasets.
- Understand how to train and evaluate machine learning models for classification tasks.
- Learn about hyperparameter tuning techniques like Randomized Search CV and Grid Search CV.

## Files

- `heart.csv`: Dataset containing features and target labels for heart disease prediction.
- `main.py`: Python script containing the code for training and evaluating the models.
- `README.md`: This file.

## Usage

1. Clone the repository:

git clone https://github.com/yourusername/heart-disease-prediction.git

2. Ensure you have the necessary dependencies installed:

pip install pandas scikit-learn

3. Run the `main.py` script:

python main.py

## Example Output

First result: {'Accuracy': 0.82, 'Precision': 0.82, 'Recall': 0.82, 'F1': 0.82}
Second result: {'Accuracy': 0.75, 'Precision': 0.75, 'Recall': 0.77, 'F1': 0.76}
Third result: {'Accuracy': 0.78, 'Precision': 0.78, 'Recall': 0.76, 'F1': 0.77}
RSCV best parameters: {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 30}
RSCV result: {'Accuracy': 0.84, 'Precision': 0.84, 'Recall': 0.86, 'F1': 0.85}
GSCV best parameters: {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 100}
GSCV result: {'Accuracy': 0.78, 'Precision': 0.76, 'Recall': 0.8, 'F1': 0.78}

## Results

The script will output evaluation metrics for each model, including accuracy, precision, recall, and F1-score. Additionally, it will perform hyperparameter tuning using both Randomized Search CV and Grid Search CV, and display the best parameters found.

## Acknowledgments:
This program was created as part of the DS, AI and ML course offered by the National Emerging Skills Program (NESP).
