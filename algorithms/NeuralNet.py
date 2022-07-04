import time

from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from util.DataTransformations import neuralNetTransform


def runNeuralNetwork(data, logger):
    """
    Builds a neural network and evaluates it on the data provided in util.neuralNetTransform
    WIP, as we collect more data (more games played in the split), then the network will most
    likely expand and start performing better. But for now, its pretty simple and doesnt really
    perform that well

    :param data: The data generated from dataGenerator.py
    :param logger: The logger created in ChampionsQueueModel.py
    :return: Nothing
    """
    logger.info("Starting Neural Network Operations")

    logger.info("Splitting data into training and validation set")
    d0 = time.perf_counter()
    neuralNetworkData = neuralNetTransform(data, logger)
    toSplit = int(neuralNetworkData.shape[0] * 0.7)
    Xtrain = neuralNetworkData[0:toSplit, 1:]
    ytrain = neuralNetworkData[0:toSplit, 0].reshape((toSplit))
    Xtest = neuralNetworkData[toSplit:neuralNetworkData.shape[0], 1:]
    ytest = neuralNetworkData[toSplit:neuralNetworkData.shape[0], 0].reshape((neuralNetworkData.shape[0] - toSplit))
    split2 = int(Xtest.shape[0] * 0.5)
    Xtest = Xtest[split2:]
    ytest = ytest[split2:]
    Xval = Xtest[:split2]
    yval = ytest[:split2]
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Building and compiling model")
    d0 = time.perf_counter()
    model = Sequential()
    model.add(Dense(50, input_dim=20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    Kevin = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        # Exponential decay rate for first moment estimates
        beta_1=0.9,
        # Exponential decay rate for second moment estimates
        beta_2=0.999,
        # Should be really small, helps avoid divide by 0
        epsilon=1e-07,
        # AMSGrad is an extension to the Adam version of gradient descent that attempts to improve the convergence
        # properties of the algorithm, avoiding large abrupt changes in the learning rate for each input variable.
        amsgrad=False,
        name='Adam',
    )
    model.compile(optimizer=Kevin,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")


    logger.info("Fitting and Evaluating Performance")
    d0 = time.perf_counter()
    history = model.fit(Xtrain, ytrain, epochs=150, batch_size=7, validation_data=(Xval, yval))
    test_loss, test_acc = model.evaluate(Xtest, ytest)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Visualizing Model performance")
    print(history.history.keys())
    d0 = time.perf_counter()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    # This is to debug, see predictions on individual datapoints
    ypreds = model.predict(Xtest)
    for i in range(Xtest.shape[0]):
        print("Predicted: ", ypreds[i])
        print("Actual: ", ytest[i])
        print("--------------------------------------------------")
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)
    for i in range(ypreds.shape[0]):
        if ypreds[i] < 0.5:
            ypreds[i] = 0
        else:
            ypreds[i] = 1
    print(classification_report(ytest, ypreds))


    logger.info("All Neural Network operations complete")
