import time

from pandas import DataFrame
from sklearn.mixture import GaussianMixture
from util.DataTransformations import clusteringTransform
import matplotlib.pyplot as plt


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
    plt.scatter(*zip(*correctPoints), color='green',
                label='Correct', alpha=0.5)
    plt.scatter(*zip(*incorrectPoints), color='red',
                label='Incorrect', alpha=0.5)
    plt.show()
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info(f"Gaussian Mixed Model accuracy: {accuracy:0.4f}")

    logger.info("GMM Model Successfully Completed")
