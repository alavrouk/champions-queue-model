import time

from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from util.DataTransformations import clusteringTransform


def runKMeans(data, logger):
    """
    This is the kmeans algorithm. This is not supposed to achieve high accuracy.
    Rather, it is supposed to serve as a baseline to see if our models are actually doing anything.
    Also it outputs a cool graph.
    :param data:
    :param logger:
    :return:
    """

    logger.info("Starting KMeans operations...")
    d0 = time.perf_counter()
    kmData = clusteringTransform(data)
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
    print(predictions)
    # centroids = kmeans.cluster_centers_
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info("Evaluating k-means model...")
    d0 = time.perf_counter()
    # pred = np.zeros(shape=(kmDataOrig.shape[0], 1))
    # actual = (kmDataOrig[:, 0]).reshape((kmDataOrig.shape[0], 1))
    # for i in range(kmDataOrig.shape[0]):
    #     vec = kmDataOrig[i, 1:3]
    #     d1 = np.linalg.norm(vec - centroids[0])
    #     d2 = np.linalg.norm(vec - centroids[1])
    #     if d1 > d2:
    #         pred[i, 0] = 0
    #     else:
    #         pred[i, 0] = 1
    # d1 = time.perf_counter()
    # logger.info(f"Done in {d1 - d0:0.4f} seconds")

    # logger.info("Finding incorrectly predicted datapoints...")
    # d0 = time.perf_counter()
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
    # for i in range(kmDataOrig.shape[0]):
    #     if pred[i, 0] == actual[i, 0]:
    #         numCorrect += 1
    #         correctIndices.append(i)
    #     else:
    #         wrongIndices.append(i)
    # accuracy = numCorrect / kmDataOrig.shape[0]

    # prediction values were inverse to data
    if accuracy < 0.5:
        accuracy = 1 - accuracy
        correctPoints, incorrectPoints = incorrectPoints, correctPoints

    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")
    
    # incorrectLabels = kmDataOrig[wrongIndices, 1:3]
    # correctLabels = kmDataOrig[correctIndices, 1:3]
    # d1 = time.perf_counter()
    # logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Plotting model results...")
    d0 = time.perf_counter()
    plt.scatter(*zip(*correctPoints), color='green', label='Correct', alpha=0.5)
    plt.scatter(*zip(*incorrectPoints), color='red', label='Incorrect', alpha=0.5)
    plt.show()
    # xw = incorrectLabels[:, 0]
    # yw = incorrectLabels[:, 1]
    # xc = correctLabels[:, 0]
    # yc = correctLabels[:, 1]
    # plt.plot(xc, yc, 'o', color='green', alpha=0.5)
    # plt.plot(xw, yw, 'o', color='red', alpha=0.5)
    # plt.show()
    # d1 = time.perf_counter()
    logger.info(f"Done in {time.perf_counter() - d0:0.4f} seconds")

    logger.info(f"Accuracy: {accuracy:0.4f}")

    logger.info("Concluded KMeans operations")
