import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
from sklearn import svm

"""
This file will contain transformations for each algorithm. All of these transformations will take in
the original "data" array, which for now is of format
patch result player1 ... player10 champion1 ... champion10

TODO: Honestly I don't really know how to make this modular. As in, a change in data format doesnt affect the functions
"""

import numpy as np
from numpy import genfromtxt


def kNNTransform(data):
    """
    Output: [ Outcome, Avg_playerWR_t1 + Avg_champWR_t1, Avg_playerWR_t2 + Avg_champWR_t2]
    In theory this could lend itself to k nearest neighbors, this would follow the theory that the higher
    winrates in both categories lead to more wins.

    The 3 features lends itself nicely to a plot, which is why I chose to try this first. ColorCode victory/defeat
    and then x = f1 and y = f2.

    Things to try:
        Average instead of adding
        Make each one into a separate feature
    """
    # First I will get rid of the patch
    kNNData = np.delete(data, 0, 1)
    # Now I will replace the champions with the specified champion winrate
    championWinrates = genfromtxt(
        'data/champion_winrate.csv', delimiter=',', dtype='U')
    for i in range(kNNData.shape[0]):
        for j in range(11, 21):
            index = np.where(championWinrates == kNNData[i, j])
            kNNData[i, j] = championWinrates[index[0][0], 1]
            kNNData[i, j] = kNNData[i, j].replace('%', '')
    # Now I will replace the players with the specified player winrate
    playerWinrates = genfromtxt(
        'data/player_winrate.csv', delimiter=',', dtype='U')
    for i in range(kNNData.shape[0]):
        for j in range(1, 11):
            index = np.where(playerWinrates == kNNData[i, j])
            kNNData[i, j] = playerWinrates[index[0][0], 1]
            kNNData[i, j] = kNNData[i, j].replace('%', '')

    # First column will be result, with defeat being 1 and victory 0
    finalKNNData = data[:, 1]
    for i in range(finalKNNData.size):
        if finalKNNData[i] == "Defeat":
            finalKNNData[i] = 0
        else:
            finalKNNData[i] = 1
    finalKNNData = np.asarray(finalKNNData)
    finalKNNData = finalKNNData.reshape((finalKNNData.size, 1))

    # Second column will is sum of average of first team players and champions winrate
    team1 = []
    for i in range(kNNData.shape[0]):
        sum1 = 0
        for j in range(1, 6):
            sum1 = sum1 + float(kNNData[i, j])
        sum1 = sum1 / 5
        sum2 = 0
        for j in range(11, 16):
            sum2 = sum2 + float(kNNData[i, j])
        sum2 = sum2 / 5
        team1.append((sum1 + sum2) / 2)
    team1 = np.asarray(team1)
    team1 = team1.reshape((team1.size, 1))
    finalKNNData = np.concatenate((finalKNNData, team1), 1)

    # Third column will be sum of average of second team players and champions winrate
    team2 = []
    for i in range(kNNData.shape[0]):
        sum1 = 0
        for j in range(6, 11):
            sum1 = sum1 + float(kNNData[i, j])
        sum1 = sum1 / 5
        sum2 = 0
        for j in range(16, 21):
            sum2 = sum2 + float(kNNData[i, j])
        sum2 = sum2 / 5
        team2.append((sum1 + sum2) / 2)
    team2 = np.asarray(team2)
    team2 = team2.reshape((team2.size, 1))
    finalKNNData = np.concatenate((finalKNNData, team2), 1)

    # Finally convert to an array of doubles and return
    finalKNNData = finalKNNData.astype(np.double)

    return finalKNNData


def neuralNetTransform(data, logger):
    """
    Formats data as follows: 
    [Outcome, Player 1 Team 1 Champ + Overall WR, ... Player 5 Team 2 Champ + Overall WR]
    """
    logger.info("Building Neural Network Data set...")
    d0 = time.perf_counter()

    # First I will get rid of the patch
    neuralNetData = np.delete(data, 0, 1)

    # Now I will replace the champions with the specified champion winrate
    championWinrates = genfromtxt(
        'data/champion_winrate.csv', delimiter=',', dtype='U')
    for i in range(neuralNetData.shape[0]):
        for j in range(11, 21):
            index = np.where(championWinrates == neuralNetData[i, j])
            neuralNetData[i, j] = championWinrates[index[0][0], 1]
            neuralNetData[i, j] = neuralNetData[i, j].replace('%', '')

    # Now I will replace the players with the specified player winrate
    playerWinrates = genfromtxt(
        'data/player_winrate.csv', delimiter=',', dtype='U')
    for i in range(neuralNetData.shape[0]):
        for j in range(1, 11):
            index = np.where(playerWinrates == neuralNetData[i, j])
            neuralNetData[i, j] = playerWinrates[index[0][0], 1]
            neuralNetData[i, j] = neuralNetData[i, j].replace('%', '')

    # First column will be result, with defeat being 1 and victory 0
    for i in range(neuralNetData.shape[0]):
        if neuralNetData[i, 0] == "Defeat":
            neuralNetData[i, 0] = 0
        else:
            neuralNetData[i, 0] = 1

    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")
    neuralNetData = neuralNetData.astype(np.double)

    return neuralNetData
