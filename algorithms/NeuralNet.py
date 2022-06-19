import time

from keras import Sequential
from keras.layers import Dense
import tensorflow as tf

from util.DataTransformations import neuralNetTransform


def runNeuralNetwork(data, logger):

    neuralNetworkData = neuralNetTransform(data, logger)

    Xtrain = neuralNetworkData[0:400, 1:]

    Ytrain = neuralNetworkData[0:400, 0].reshape((400))

    Xtest = neuralNetworkData[400:512, 1:]

    Ytest = neuralNetworkData[400:512, 0].reshape((111))

    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='leaky_relu'))
    model.add(Dense(10, activation='leaky_relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, epochs=200, batch_size=20)
    test_loss, test_acc = model.evaluate(Xtest, Ytest)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)
    Ypreds = model.predict(Xtest)
    for i in range(Xtest.shape[0]):
        print("Predicted: ", Ypreds[i])
        print("Actual: ", Ytest[i])
