import logging
from algorithms.GMM import runGMM
from algorithms.KMeans import runKMeans
from algorithms.RandomForest import runRandomForest
from algorithms.SVM import runSVM
from algorithms.NeuralNet import runNeuralNetwork
import dataGenerator
import numpy as np
import util.DataTransformations
from util.CustomFormatter import CustomFormatter


def createLogger():
    """
    Creates the custom logger (colorful) that is used in the program. Just basically fancier print statements

    :return: The logger, which is in turn used in whatever function needs it
    """
    # Create custom logger logging all five levels
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Define format for logs
    fmt = '%(asctime)s | %(levelname)8s | %(message)s'

    # Create stdout handler for logging to the console (logs all five levels)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt))

    # Create file handler for logging to a file (logs all five levels)
    # today = datetime.date.today()
    # file_handler = logging.FileHandler(
    #     'my_app_{}.log'.format(today.strftime('%Y_%m_%d')))
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(logging.Formatter(fmt))

    # Add both handlers to the logger
    logger.addHandler(stdout_handler)
    # logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    """
    The main function. The way we see this project being formatted is just to be able to plug and play into the n
    main function. So whichever model you want to try, just write it into the main function. The models are each their 
    own separate class.

    """
    # Pass this guy into everything (or make global variable I have no idea how python works)
    logger = createLogger()

    yesno = input("Would you like to regenerate your data? (y/n) \n")
    if yesno == 'y':
        numClicks = input("How many sets of 20 data points would you like? \n")
        print("You have selected", numClicks, "sets of 20 data points")
        logger.info("Starting data generation...")
        dataGenerator.generateData(int(numClicks), logger)
        logger.info("Data generation complete!")
    data = np.genfromtxt('data/champions_queue_data.csv',
                         delimiter=',', dtype='U')

    algorithm = input("Which algorithm would you like to run? \n"
                      "1 ---- SVM \n"
                      "2 ---- RandomForest \n"
                      "3 ---- NeuralNet \n"
                      "4 ---- KMeans \n"
                      "5 ---- Gaussian Mixture Models \n")
    if algorithm == '1':
        runSVM(data, logger)
    elif algorithm == '2':
        runRandomForest(data, logger)
    elif algorithm == '3':
        runNeuralNetwork(data, logger)
    elif algorithm == '4':
        runKMeans(data, logger)
    elif algorithm == '5':
        runGMM(data, logger)
    else:
        print('bruh')
