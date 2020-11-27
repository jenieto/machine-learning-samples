import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from mlxtend.evaluate import mcnemar


iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30, shuffle=True, stratify=iris.target)

# Logistic Regression
regressionModel = LogisticRegression()
regressionModel.fit(X_train, y_train)

score = regressionModel.score(X_test, y_test)
print(score)

# KNN
knnModel = KNeighborsClassifier(n_neighbors=3)
knnModel.fit(X_train, y_train)
score = knnModel.score(X_test, y_test)
print(score)

# Create contingency table
contingency = np.array([[0, 0],
                        [0, 0]])
for x, y in zip(X_test, y_test):
    predicted_regression = regressionModel.predict([x])[0]
    predicted_knn = knnModel.predict([x])[0]
    if predicted_regression == y and predicted_knn == y:
        contingency[0, 0] = contingency[0, 0] + 1
    elif predicted_regression == y and predicted_knn != y:
        contingency[1, 0] = contingency[1, 0] + 1
    elif predicted_regression != y and predicted_knn == y:
        contingency[0, 1] = contingency[0, 1] + 1
    else:
        contingency[1, 1] = contingency[1, 1] + 1
print(contingency)

# Calculate McNemar test
statistic, pvalue = mcnemar(contingency, exact=True)
print('statistic=%.3f, p-value=%.3f' % (statistic, pvalue))
alpha = 0.05
if pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')


