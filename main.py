import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set a seed for reproducibility
np.random.seed(7)

# Define a function to evaluate classification metrics
def evaluate_hyper_metrics(y_test, y_predicted):
    """ This function returns evaluation metrics """
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)

    metrics_dict = {'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1}
    return metrics_dict

# Read the 'heart.csv' file and shuffle the rows
heart = pd.read_csv('heart.csv')
mixed_heart_disease = heart.sample(frac=1)

# Separate features (x) and target (y)
x = mixed_heart_disease.drop('target', axis=1)
y = mixed_heart_disease['target']

# Split the data into training, validation, and testing sets
train_split = round(0.7 * len(mixed_heart_disease))
valid_split = round(train_split + 0.15 * len(mixed_heart_disease))

x_train, y_train = x[:train_split], y[:train_split]
x_valid, y_valid = x[train_split:valid_split], y[train_split:valid_split]
x_test, y_test = x[valid_split:], y[valid_split:]

# Initialize a Random Forest Classifier and train it on the training data
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# Evaluate the model on the validation set and print the metrics
y_predicted = clf.predict(x_valid)
first_result = evaluate_hyper_metrics(y_valid, y_predicted)
print(f'First result: {first_result}')

# Train a second Random Forest Classifier with a different hyperparameter (n_estimators=10)
clf2 = RandomForestClassifier(n_estimators=10)
clf2.fit(x_train, y_train)
y_predicted = clf2.predict(x_valid)
second_result = evaluate_hyper_metrics(y_valid, y_predicted)
print(f'Second result: {second_result}')

# Train a third Random Forest Classifier with additional hyperparameters (n_estimators=10, max_depth=10)
clf3 = RandomForestClassifier(n_estimators=10, max_depth=10)
clf3.fit(x_train, y_train)
y_predicted = clf3.predict(x_valid)
third_result = evaluate_hyper_metrics(y_valid, y_predicted)
print(f'Third result: {third_result}')

# Define a dictionary of hyperparameters for Randomized Search CV
hyper_params = {'max_depth': [10, 20, 30],
                'n_estimators': [10, 100, 500, 1000],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['log2', 'sqrt']}

# Read the 'heart.csv' file again for Randomized Search CV
rscv_heart = pd.read_csv('heart.csv')
rscv_heart = rscv_heart.sample(frac=1)
rscv_x = rscv_heart.drop('target', axis=1)
rscv_y = rscv_heart['target']

# Split data into training and testing sets for Randomized Search CV
rscv_x_train, rscv_x_test, rscv_y_train, rscv_y_test = train_test_split(rscv_x, rscv_y, test_size=0.2)

# Initialize a Random Forest Classifier for Randomized Search CV
clf4 = RandomForestClassifier(n_jobs=1)

# Initialize a Randomized Search CV with 10 iterations and 5-fold cross-validation
from sklearn.model_selection import RandomizedSearchCV
rscv_clf = RandomizedSearchCV(estimator=clf4,
                              param_distributions=hyper_params,
                              n_iter=10,
                              cv=5,
                              verbose=2)

# Fit the Randomized Search CV on the training data
rscv_clf.fit(rscv_x_train, rscv_y_train)
print(f'RSCV best parameters: {rscv_clf.best_params_}')

# Evaluate the best model from Randomized Search CV on the test set and print the metrics
rscv_y_predicted = rscv_clf.predict(rscv_x_test)
fourth_result = evaluate_hyper_metrics(rscv_y_test, rscv_y_predicted)
print(f'RSCV result: {fourth_result}')

# Define a dictionary of hyperparameters for Grid Search CV
gscv_params = {'max_depth': [10, 20, 30],
                'n_estimators': [50, 100, 200],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [2, 4, 6],
                'max_features': ['log2', 'sqrt']}

# Initialize a Random Forest Classifier for Grid Search CV
clf5 = RandomForestClassifier(n_jobs=1)

# Initialize a Grid Search CV with 5-fold cross-validation
from sklearn.model_selection import GridSearchCV
gscv_clf = GridSearchCV(estimator=clf5,
                        param_grid=gscv_params,
                        cv=5,
                        verbose=2)

# Fit the Grid Search CV on the training data
gscv_clf.fit(x_train, y_train)
print(f'GSCV best parameters: {gscv_clf.best_params_}')

# Evaluate the best model from Grid Search CV on the test set and print the metrics
gscv_y_predicted = gscv_clf.predict(x_test)
fifth_result = evaluate_hyper_metrics(y_test, gscv_y_predicted)
print(f'GSCV result: {fifth_result}')
