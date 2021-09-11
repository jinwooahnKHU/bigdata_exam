from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def rf_regressor(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    n_estimators = [int(x) for x in np.linspace(start = 100, stop= 300, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 110, num=11)]
    min_samples_split = [2,5,10,]
    min_samples_leaf = [1,2,4,]
    bootstrap = [True, False]

    random_grid = {'n_estimators' : n_estimators,
                    'max_features' : max_features,
                    'max_depth' : max_depth,
                    'min_samples_split' : min_samples_split,
                    'min_samples_leaf' : min_samples_leaf,
                    'bootstrap' : bootstrap}
    
    # create object, use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
    n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

    #Fit the Base and Random model
    base_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    base_model.fit(X_train, y_train)
    rf_random.fit(X_train, y_train)
    best_random = rf_random.best_estimator_

    return base_model, best_random

def regressor_evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape

    print('Model Performace')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# 독립변수 연속형, 종속변수는 범주형이여야 함
def logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    base_model = LogisticRegression(random_state=42)
    X_train = scaler.fit_transform(X_train)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['none', 'l1', 'l2', 'elasticnet']
    c = [10, 1.0, 0.1, 0.01]
    random_grid = {'solver' : solvers,
                    'penalty' : penalty,
                    'C' : c
    }
    logistic_random = RandomizedSearchCV(estimator=base_model, param_distributions=random_grid,
    n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

    #Fit the Base and Random model
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)
    logistic_random.fit(X_train, y_train)
    best_random = logistic_random.best_estimator_
    
    return base_model, best_random

def logistic_evaluate(model, X_test, y_test):
    X_test = scaler.transform(X_test)
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape

    print('Model Performace')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

def k_nn_classifier(data):
    #class 작업해둬야함
    # accuracies 값을 관찰해서 알맞은 k를 설정하고 다시 해야 함.
    X, y = data.drop('CLASS', axis=1), data['CLASS']
    accuracies = []
    k_values = [range(10)]
    for k in k_values:
        # instantiate kNN with given neighbor size k
        knn = KNeighborsClassifier(n_neighbors=k)
        # run cross validation for a given kNN setup
        # I have setup n_jobs=-1 to use all cpus in my env.
        scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy', n_jobs=-1)
        accuracies.append(scores.mean())

    return accuracies

if __name__ == '__main__':

    # X_train = pd.read_csv('data/processed/X_train.csv', index_col = 0)
    # X_test = pd.read_csv('data/processed/X_test.csv', index_col = 0)
    # y_train = pd.read_csv('data/processed/y_train.csv', index_col = 0)
    # y_test = pd.read_csv('data/processed/y_test.csv', index_col = 0)
    dataset = datasets.load_boston()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #RF classifier
    base_model, best_random_model = rf_regressor(X_train, y_train)
    print(regressor_evaluate(base_model, X_test, y_test))
    print(regressor_evaluate(best_random_model, X_test, y_test))

    # logistic regression
    # 종속 변수가 0, 1 로 이루어져야 함. 연속형 안됨.
    # base_model, best_random_model = logistic_regression(X_train, y_train)
    # print(logistic_evaluate(base_model, X_test, y_test))
    # print(logistic_evaluate(best_random_model, X_test, y_test))


