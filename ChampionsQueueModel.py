from keras.backend import binary_crossentropy
from keras.layers import Dense, Dropout
from keras.optimizer_v2.gradient_descent import SGD
from keras.regularizers import l2

import dataGenerator
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from tensorflow import *
from keras import *


def changeChampionNamesToNumbers(data):
    champ_numbers =np.genfromtxt('LOL_CHAMPIONS.csv', delimiter=',', dtype='U')
    for i in range(data.shape[0]):
        if data[i, 1] == 'Victory':
            data[i, 1] = 1
        if data[i, 1] == 'Defeat':
            data[i, 1] = 0
        for j in range(2, data.shape[1]):
            loc = champ_numbers[np.where(champ_numbers == data[i, j])[0], 1]
            if loc.shape == (0, ):
                print(data[i,j])
            data[i, j] = loc[0]
    return data

def createModel():
    model = Sequential()
    model.add(Dense(25, kernel_initializer='normal', input_dim=10, activation='linear'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='linear'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    # numClicks = 40
    # if len(sys.argv) > 1:
    #     numClicks = np.int_(sys.argv[1])
    # data = dataGenerator.generateData("https://championsqueue.gg/matches", numClicks)
    # np.savetxt("champions_queue_data.csv", data, delimiter=",", fmt="%s")
    data = np.genfromtxt('champions_queue_data.csv', delimiter=',', dtype='U')
    data = changeChampionNamesToNumbers(data)
    # Here I could do a patch thing but I really cba rn

    model = createModel()

    train_X = np.int_(data[100:, 2:])
    validation_X = np.int_(data[:100, 2:])
    train_Y = np.int_(data[100:, 1])
    train_Y = train_Y.reshape((train_Y.shape[0], 1))
    validation_Y = np.int_(data[:100, 1])
    validation_Y = validation_Y.reshape((validation_Y.shape[0], 1))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=50, batch_size=15)


    success = 0
    for i in range(validation_X.shape[0]):
        value = model.predict(validation_X[i, :].reshape((1, 10)))[0]
        if value > 0.5 and validation_Y[i,0] == 1:
            print(validation_Y[i, 0] - value[0])
            success += 1
        if value <= 0.5 and validation_Y[i,0] == 0:
            print(validation_Y[i, 0] - value[0])
            success += 1
    print(success / validation_X.shape[0])
