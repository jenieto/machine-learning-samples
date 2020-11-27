import numpy as np

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.algorithms.MPE import mpe

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare data
iris = datasets.load_iris()
X_train, X_test, y_train, y_test= train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 42)
train_data_with_labels = np.insert(X_train, obj=X_train.shape[1], values=y_train, axis=1)
test_data_with_labels = np.insert(X_test, obj=X_test.shape[1], values=y_test, axis=1)

# Learn SPN
context = Context(parametric_types=[Gaussian, Gaussian, Gaussian, Gaussian, Categorical]).add_domains(train_data_with_labels)
spn_classification = learn_classifier(train_data_with_labels, context, learn_parametric, 4)

# Plot SPN
plot_spn(spn_classification, 'iris_spn.png')

# Predict
true_values = np.array(test_data_with_labels[:,-1])
items_to_predict = test_data_with_labels
items_to_predict[:, 4] = np.nan
predicted_values = mpe(spn_classification, test_data_with_labels)
predicted_labels = predicted_values[:, 4]

acc = accuracy_score(true_values, predicted_labels)
print(acc)


