import time
from matplotlib.colors import LogNorm

from pandas import DataFrame
from sklearn.mixture import GaussianMixture
from util.DataTransformations import clusteringTransform
import matplotlib.pyplot as plt
import numpy as np


def runGMM(data, logger):
    """
    Gaussian mixture model using average player+champ winrates for each team.
    :param data:
    :param logger: see ChampionsQueueModel.py -> createLogger()
    """
    logger.info("Starting GMM operations...")
    d0 = time.perf_counter()
    clusterData = clusteringTransform(data, logger)
    numRows = clusterData.shape[0]
    clusterData = {
        'outcome': clusterData[:, 0],
        'team1wr': clusterData[:, 1],
        'team2wr': clusterData[:, 2]
    }
    df = DataFrame(clusterData, columns=['team1wr', 'team2wr'])
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Running GMM algorithm...")
    d0 = time.perf_counter()
    gmm = GaussianMixture(n_components=2)
    predictions = gmm.fit_predict(df)
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Evaluating GMM model...")
    d0 = time.perf_counter()
    numCorrect = 0
    correctPoints = []
    incorrectPoints = []
    # iterate through all data points, checking if prediction is correct
    for i in range(numRows):
        if predictions[i] == clusterData['outcome'][i]:
            numCorrect += 1
            correctPoints.append(
                [clusterData["team1wr"][i], clusterData["team2wr"][i]])
        else:
            incorrectPoints.append(
                [clusterData["team1wr"][i], clusterData["team2wr"][i]])

    accuracy = numCorrect / numRows

    # prediction values were inverse to data: swap as necessary
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        correctPoints, incorrectPoints = incorrectPoints, correctPoints

    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Plotting model results...")
    d0 = time.perf_counter()
    # creates mixture Gaussian log plot
    x = np.linspace(30, 70)
    y = np.linspace(30, 70)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)
    # plot results
    plt.scatter(*zip(*correctPoints), color='green',
                label='Correct', alpha=0.5)
    plt.scatter(*zip(*incorrectPoints), color='red',
                label='Incorrect', alpha=0.5)
    plt.contour(X, Y, Z, norm=LogNorm(
        vmin=1, vmax=1000), levels=np.logspace(0, 2, 10))
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    plt.show()

    logger.info(f"Gaussian Mixed Model accuracy: {accuracy:0.4f}")

    logger.info("GMM Model Successfully Completed")
