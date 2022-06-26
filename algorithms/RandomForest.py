import time

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

from util.DataTransformations import neuralNetTransform


def runRandomForest(data, logger):
    """
    :param data:
    :return forest:
    """
    logger.info("Starting Random Forest operations...")

    logger.info(
        "Creating dataframe and splitting data into training and validation set...")
    d0 = time.perf_counter()
    rfData = neuralNetTransform(data, logger)
    # format data for use with pandas
    rfData = pd.DataFrame({'p1': rfData[:, 1],
                           'p2': rfData[:, 2],
                           'p3': rfData[:, 3],
                           'p4': rfData[:, 4],
                           'p5': rfData[:, 5],
                           'p6': rfData[:, 6],
                           'p7': rfData[:, 7],
                           'p8': rfData[:, 8],
                           'p9': rfData[:, 9],
                           'p10': rfData[:, 10],
                           'c1': rfData[:, 11],
                           'c2': rfData[:, 12],
                           'c3': rfData[:, 13],
                           'c4': rfData[:, 14],
                           'c5': rfData[:, 15],
                           'c6': rfData[:, 16],
                           'c7': rfData[:, 17],
                           'c8': rfData[:, 18],
                           'c9': rfData[:, 19],
                           'c10': rfData[:, 20],
                           'result': rfData[:, 0]})
    X = rfData[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']]
    y = rfData['result']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    # Figured I would mess around with PCA here, provides a marginal increase in accuracy
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    # The commented out stuff here is automated hyperparameter tuning
    logger.info("Fitting and creating random forest classifier...")
    d0 = time.perf_counter()
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # max_features = ['auto', 'sqrt']
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]  # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    rf = RandomForestClassifier(n_estimators=600, min_samples_split=10, min_samples_leaf=1,
                                max_features='sqrt', max_depth=40, bootstrap=False)
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
    #                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    # rf_random.fit(X_train, y_train)
    # print(rf_random.best_params_)
    y_pred = rf.predict(X_test)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Creating confusion matrix plot, as well as printing results...")
    d0 = time.perf_counter()
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=[
                                   'Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("All random forest operations complete")
