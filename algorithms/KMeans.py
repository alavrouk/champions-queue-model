import time

from pandas import DataFrame
from sklearn.cluster import KMeans
from util.DataTransformations import clusteringTransform
import matplotlib.pyplot as plt


def runKMeans(data, logger):
    """
    This is the kmeans algorithm. This is not supposed to achieve high accuracy.
    Rather, it is supposed to serve as a baseline to see if our models are actually doing anything.
    Also it outputs a cool graph.
    :param data:
    :param logger: see ChampionsQueueModel.py -> createLogger()
    """
    logger.info("Starting KMeans operations...")
    d0 = time.perf_counter()
    kmData = clusteringTransform(data, logger)
    numRows = kmData.shape[0]
    kmData = {
        'outcome': kmData[:, 0],
        'team1wr': kmData[:, 1],
        'team2wr': kmData[:, 2]
    }
    df = DataFrame(kmData, columns=['team1wr', 'team2wr'])
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Running k-means algorithm and retrieving centroids...")
    d0 = time.perf_counter()
    kmeans = KMeans(n_clusters=2)
    predictions = kmeans.fit_predict(df)
    # centroids = kmeans.cluster_centers_
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Evaluating k-means model...")
    d0 = time.perf_counter()
    numCorrect = 0
    correctPoints = []
    incorrectPoints = []
    for i in range(numRows):
        if predictions[i] == kmData['outcome'][i]:
            numCorrect += 1
            correctPoints.append(
                [kmData["team1wr"][i], kmData["team2wr"][i]])
        else:
            incorrectPoints.append(
                [kmData["team1wr"][i], kmData["team2wr"][i]])

    accuracy = numCorrect / numRows

    # prediction values were inverse to data
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

    logger.info(f"Accuracy: {accuracy:0.4f}")

    logger.info("Concluded KMeans operations")
