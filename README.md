# Automated Hyperparameter Optimization (HPO) System

## Problem Statement
The quality of performance of a Machine Learning model heavily depends on its hyperparameter settings. Given a dataset and a task, the choice of the machine learning (ML) model and its hyperparameters is typically performed manually. This project aims to develop an automated hyperparameter optimization (HPO) system using AutoML techniques that can efficiently identify the best hyperparameter configuration for a given machine learning model and dataset.

## Datasets Used
- **Breast Cancer Dataset**: A standard dataset from scikit-learn used for binary classification tasks.
- **application_train Dataset**: A dataset from a Kaggle competition used for binary classification tasks related to loan default prediction.

## Models and Techniques
### 1. Support Vector Machine (SVM)
- **Objective**: Binary classification of breast cancer diagnosis and loan default prediction.
- **Optimization Technique**: Bayesian Optimization.
- **Evaluation Metrics**: ROC AUC, learning rate distribution comparison.

### 2. Random Forest Classifier
- **Objective**: Binary classification of breast cancer diagnosis and loan default prediction.
- **Optimization Technique**: Bayesian Optimization.
- **Evaluation Metrics**: ROC AUC, learning rate distribution comparison.

### 3. Gradient Boosting Classifier
- **Objective**: Binary classification of breast cancer diagnosis and loan default prediction.
- **Optimization Technique**: Bayesian Optimization.
- **Evaluation Metrics**: ROC AUC, learning rate distribution comparison.

## Implementation Details
### Breast Cancer Dataset
- **Objective**: Binary classification of breast cancer diagnosis.
- **Models Evaluated**: SVM, Random Forest, Gradient Boosting Classifier.
- **Optimization Technique**: Bayesian Optimization.

#### Code Snippet (Example for SVM using Bayesian Optimization)
```python
# Example code snippet for SVM optimization using Bayesian Optimization
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from bayesian_optimizer import BayesianOptimizer

# Define the SVM function to optimize
def svm_func(C, kernel, gamma):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    scores = cross_val_score(svm, X_breast, y_breast, cv=5, scoring='roc_auc')
    return np.mean(scores)

# Create an instance of BayesianOptimizer
optimizer = BayesianOptimizer(func=svm_func, float_param_ranges={'C': (0.1, 10), 'gamma': (0.01, 1)}, int_param_candidates={'kernel': ['linear', 'rbf']})

# Perform optimization
optimizer.optimize()

# Get results
results = optimizer.get_results()
print(results)
```
#### Example Plot
```python
# Example code snippet for plotting learning rate distributions
import matplotlib.pyplot as plt

# Plotting code here
plt.figure(figsize=(10, 6))
# Plot learning rate distributions for SVM, Random Forest, Gradient Boosting Classifier
plt.title('Learning Rate Distribution Comparison')
plt.xlabel('Learning Rate')
plt.ylabel('Density')
plt.legend(['SVM', 'Random Forest', 'Gradient Boosting'])
plt.show()
```
### Code Snippet (Example for Random Forest using Bayesian Optimization)
```python
# Example code snippet for Random Forest optimization using Bayesian Optimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayesian_optimizer import BayesianOptimizer

# Define the Random Forest function to optimize
def rf_func(n_estimators, max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
    return np.mean(scores)

# Create an instance of BayesianOptimizer
optimizer = BayesianOptimizer(func=rf_func, int_param_candidates={'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]})

# Perform optimization
optimizer.optimize()

# Get results
results = optimizer.get_results()
print(results)
```
## Conclusion 
This README provides an overview of the automated hyperparameter optimization system developed using AutoML techniques. It covers the datasets used, models evaluated, optimization techniques employed, and evaluation metrics used for performance assessment. For detailed implementation and code examples, refer to the corresponding Jupyter notebooks or Python scripts in the repository.

## Resources

### Bayesian Optimization Library
- **scikit-optimize**: A library for sequential model-based optimization, available at [scikit-optimize](https://scikit-optimize.github.io/stable/).

### AutoML Frameworks
- **Auto-sklearn**: An automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator, available at [Auto-sklearn](https://github.com/automl/auto-sklearn).
- **H2O AutoML**: Automatic machine learning for the H2O platform, providing easy-to-use interfaces for training models, available at [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html).
- **TPOT**: A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming, available at [TPOT](https://github.com/EpistasisLab/tpot).

### Datasets
- **Breast Cancer Dataset**: A standard dataset from scikit-learn used for binary classification tasks, available [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
- **application_train Dataset**: A dataset from a Kaggle competition used for binary classification tasks related to loan default prediction, available [here](https://www.kaggle.com/c/home-credit-default-risk/data).

### Further Links
- [Automated Model Tuning](https://www.kaggle.com/code/willkoehrsen/automated-model-tuning/notebook)
- [AutoML](https://www.automl.org/hpo-overview/?cmplz-force-reload=1716301158840)
- [AutoML Papers](https://github.com/windmaple/awesome-AutoML?tab=readme-ov-file)

