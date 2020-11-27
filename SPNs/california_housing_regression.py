import numpy as np

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.algorithms.MPE import mpe

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Prepare data
california_housing = datasets.fetch_california_housing()
X_train, X_test, y_train, y_test= train_test_split(california_housing.data, california_housing.target, test_size = 0.3, random_state = 42)
train_data_with_labels = np.insert(X_train, obj=X_train.shape[1], values=y_train, axis=1)
test_data_with_labels = np.insert(X_test, obj=X_test.shape[1], values=y_test, axis=1)

# Learn SPN
parametric_types = [Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian]
target_position = 8
context = Context(parametric_types=parametric_types).add_domains(train_data_with_labels)
spn = learn_classifier(train_data_with_labels, context, learn_parametric, target_position)

# Plot SPN
# plot_spn(spn, 'images/california_housing_spn.png')

# Predict
true_values = np.array(test_data_with_labels[:,-1])
items_to_predict = test_data_with_labels
items_to_predict[:, target_position] = np.nan
predicted_values = mpe(spn, test_data_with_labels)
predicted_labels = predicted_values[:, target_position]

error = mean_squared_error(true_values, predicted_labels)
print(f'MSE test: {error}')


