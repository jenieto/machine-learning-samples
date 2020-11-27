import numpy as np

from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.sklearn import SPNClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Prepare data
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=42)

# Create sklearn classifier
classifier = SPNClassifier([Gaussian, Gaussian, Gaussian, Gaussian, Categorical])

# Cross Validate Classifier
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print(f'CV accuracy: {np.mean(scores)},  +/- {np.std(scores)}')

# fit
classifier.fit(X_train, y_train)

# Predict and evaluate accuracy on test set
predicted_values = classifier.predict(X_test)
score = accuracy_score(y_test, predicted_values)
print(f'Accuracy on test set: {score}')
