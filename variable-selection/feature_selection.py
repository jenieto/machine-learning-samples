import data_generator as dg
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF

data, labels, columns = dg.read_data()


def mutual_information_selection():
    print('--------------------------------------------------------')
    print('Utilizando tecnica de Informacion Mutua...')
    mutual_information = mutual_info_classif(data, labels, discrete_features=True)
    for i, mi in enumerate(mutual_information):
        print(f'{columns[i]}:{mi}')


def chi2_selection():
    print('--------------------------------------------------------')
    print('Utilizando tecnica de Chi Cuadrado...')
    chi2_values, p_values = chi2(data, labels)
    for i, chi in enumerate(zip(chi2_values, p_values)):
        print(f'{columns[i]}:{chi[0]} - p-value: {chi[1]}')


def relief_selection():
    print('--------------------------------------------------------')
    print('Utilizando tecnica RELIEF...')
    relief = ReliefF(n_neighbors=50, n_features_to_keep=5)
    relief.fit(data, labels)
    print(f'Puntuacion de las variables: {relief.feature_scores}')


def wrapper_selection():
    print('--------------------------------------------------------')
    print('Utilizando tecnica de Envoltura...')
    models = [svm.SVC(), RandomForestClassifier(), GaussianNB(), LogisticRegression(), KNeighborsClassifier()]
    for model in models:
        efs = ExhaustiveFeatureSelector(model,
                                        min_features=1,
                                        max_features=5,
                                        scoring='accuracy',
                                        cv=5)
        efs = efs.fit(data, labels)
        selected_features = columns[list(efs.best_idx_)]
        print(f'Variables seleccionadas utilizando {model}: {selected_features}')


def pca_selection():
    print('--------------------------------------------------------')
    print('Utilizando tecnica de PCA...')
    variance = 0.8
    pca = decomposition.PCA()
    pca.fit(data)
    X = pca.transform(data)

    plt.rcParams["figure.figsize"] = (12, 6)

    number_of_variables = len(data[0]) + 1
    fig, ax = plt.subplots()
    xi = np.arange(1, number_of_variables, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    print(f'Ratio de varianza explicado: {pca.explained_variance_ratio_}')

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Numero de componentes')
    plt.xticks(np.arange(0, number_of_variables, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Varianza acumulada (%)')
    plt.title('Numero de componentes necesarios para explicar la varianza')

    plt.axhline(y=variance, color='r', linestyle='-')
    plt.text(0.5, 0.85, f'{variance*100}% varianza', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

    components = np.arange(1, pca.explained_variance_ratio_.shape[0] + 1)
    plt.bar(components, pca.explained_variance_ratio_)
    plt.xlabel('Componentes')
    plt.ylabel('Varianza explicada');
    plt.show()


wrapper_selection()
pca_selection()
relief_selection()
chi2_selection()
mutual_information_selection()