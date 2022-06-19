import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from util.DataTransformations import clusteringTransform
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


def runSVM(data, logger):
    """
    Utilises a support vector machine classifier to create a decision boundary for determining win loss

    :param data: The big ole data array that gets generated in DataGenerator. This then gets truncated
    and modified using an appropriate function from util.DataTransformations
    :param logger: The logger that we made in ChampionsQueueModel.py
    :return: Nothing, just runs the classifier
    """
    logger.info("Starting SVM operations...")

    logger.info("Splitting data into training and validation set...")
    d0 = time.perf_counter()
    # Using kNNData here transform here as they are the same
    SVMData = clusteringTransform(data)
    # First I want all of the data, so that I can end up plotting it
    x = np.asarray(SVMData[:, 1]).reshape((SVMData.shape[0], 1))
    y = np.asarray(SVMData[:, 2]).reshape((SVMData.shape[0], 1))
    X = np.concatenate((x, y), axis=1)
    c = np.asarray(SVMData[:, 0])
    # Now I split the data into training and testing sets
    training_data, testing_data = train_test_split(
        SVMData, test_size=0.2, random_state=23)
    Xtrain = np.asarray(training_data[:, 1:3]).reshape(
        (training_data.shape[0], 2))
    ctrain = np.asarray(training_data[:, 0])
    Xtest = np.asarray(testing_data[:, 1:3]).reshape(
        (testing_data.shape[0], 2))
    ctest = np.asarray(testing_data[:, 0])
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Fitting SVM...")
    d0 = time.perf_counter()
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=0.5)
    clf.fit(Xtrain, ctrain)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Predicting using SVM...")
    d0 = time.perf_counter()
    cpred = clf.predict(Xtest)
    print("Accuracy:", metrics.accuracy_score(ctest, cpred))
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Plotting data and resulting decision boundary...")
    d0 = time.perf_counter()
    cdict = {0: 'green', 1: 'red'}
    fig, ax = plt.subplots()
    for g in np.unique(c):
        ix = np.where(c == g)
        ax.scatter(x[ix], y[ix], c=cdict[g], label=g, s=20, alpha=0.5)
    red_patch = mpatches.Patch(color='red', label='Team 1 Defeat')
    green_patch = mpatches.Patch(color='green', label='Team 1 Victory')
    plt.legend(handles=[green_patch, red_patch])
    plt.xlabel('mean(Avg_playerWR_t1, Avg_champWR_t1)')
    plt.ylabel('mean(Avg_playerWR_t2, Avg_champWR_t2)')
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='black')
    plt.show()
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")
    logger.info("All SVM operations complete")
