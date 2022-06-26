import time

from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

from util.DataTransformations import clusteringTransform

def runKMeans(data, logger):
    '''
    This is the kmeans algorithm. This is not supposed to achieve high accuracy.
    Rather, it is supposed to serve as a baseline to see if our models are actually doing anything.
    Also it outputs a cool graph.
    '''

    logger.info("Starting KMeans operations...")

    logger.info("Setting up dataframe for clustering...")
    d0 = time.perf_counter()
    kmDataOrig = clusteringTransform(data)
    kmData = {
            'team1': kmDataOrig[:, 1],
            'team2': kmDataOrig[:, 2],
            'outcome': kmDataOrig[:, 0]
    }
    df = DataFrame(kmData, columns=['team1', 'team2'])
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Running kmeans algorithm and retrieving centroids...")
    d0 = time.perf_counter()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Generating predictions from kmeans outcome...")
    d0 = time.perf_counter()
    pred = np.zeros(shape=(kmDataOrig.shape[0], 1))
    actual = (kmDataOrig[:, 0]).reshape((kmDataOrig.shape[0], 1))
    for i in range(kmDataOrig.shape[0]):
        vec = kmDataOrig[i, 1:3]
        d1 = np.linalg.norm(vec - centroids[0])
        d2 = np.linalg.norm(vec - centroids[1])
        if d1 > d2:
            pred[i, 0] = 0
        else:
            pred[i, 0] = 1
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Finding incorrectly predicted datapoints...")
    d0 = time.perf_counter()
    numCorrect = 0
    wrongIndices = []
    correctIndices = []
    for i in range(kmDataOrig.shape[0]):
        if pred[i,0] == actual[i,0]:
            numCorrect += 1
            correctIndices.append(i)
        else:
            wrongIndices.append(i)
    accuracy = numCorrect / kmDataOrig.shape[0]
    if accuracy < 0.5:
        numCorrect = 0
        wrongIndices = []
        correctIndices = []
        for i in range(kmDataOrig.shape[0]):
            if pred[i, 0] != actual[i, 0]:
                numCorrect += 1
                correctIndices.append(i)
            else:
                wrongIndices.append(i)
        accuracy = numCorrect / kmDataOrig.shape[0]
    incorrectLabels = kmDataOrig[wrongIndices, 1:3]
    correctLabels = kmDataOrig[correctIndices, 1:3]
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Plotting correctly and incorrectly predicted datapoints...")
    d0 = time.perf_counter()
    xw = incorrectLabels[:, 0]
    yw = incorrectLabels[:, 1]
    xc = correctLabels[:, 0]
    yc = correctLabels[:, 1]
    plt.plot(xc, yc, 'o', color='green', alpha=0.5)
    plt.plot(xw, yw, 'o', color='red', alpha=0.5)
    plt.show()
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Concluded KMeans operations")