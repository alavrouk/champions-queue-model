import logging
import dataGenerator
import numpy as np
import util.DataTransformations
from util.CustomFormatter import CustomFormatter
from algorithms.SVM import runSVM
from algorithms import RandomForest
from algorithms.KMeans import runKMeans
from algorithms.NeuralNet import runNeuralNetwork


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
    The main function. The way we see this project being formatted is just to be able to plug and play into the 
    main function. So whichever model you want to try, just write it into the main function. The models are each their 
    own separate class.

    """
    # Pass this guy into everything (or make global variable I have no idea how python works)
    logger = createLogger()
    numClicks = 60
    # if len(sys.argv) > 1:
    #     numClicks = np.int_(sys.argv[1])

    # TODO: Need to add some sort of things with args or something to generate data or not, like an if statement

    logger.info("Starting data generation...")
    #dataGenerator.generateData(numClicks, logger)
    logger.info("Data generation complete!")
    data = np.genfromtxt('data/champions_queue_data.csv', delimiter=',', dtype='U')
    runKMeans(data, logger)
    #RandomForest.runRandomForest(data, logger)
    #runSVM(data, logger)
    #runNeuralNetwork(data, logger)
