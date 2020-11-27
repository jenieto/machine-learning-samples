from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from mlxtend.evaluate import paired_ttest_5x2cv


iris = datasets.load_iris()

# Logistic Regression
regressionModel = LogisticRegression()

# KNN
knnModel = KNeighborsClassifier(n_neighbors=3)

# Calculate 5x2 paired t test
t, p = paired_ttest_5x2cv(estimator1=regressionModel, estimator2=knnModel, X=iris.data, y=iris.target, random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
print('statistic=%.3f, p-value=%.3f' % (t, p))
alpha = 0.05
if p > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')


