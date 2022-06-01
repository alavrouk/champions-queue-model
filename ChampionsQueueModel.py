import logging
import dataGenerator
import numpy as np
from CustomFormatter import CustomFormatter
import datetime


def createLogger():
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
    today = datetime.date.today()
    file_handler = logging.FileHandler(
        'my_app_{}.log'.format(today.strftime('%Y_%m_%d')))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))

    # Add both handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    # Pass this guy into everything (or make global variable I have no idea how python works)
    logger = createLogger()
    numClicks = 3
    # if len(sys.argv) > 1:
    #     numClicks = np.int_(sys.argv[1])
    logger.info("Starting data generation...")
    data = dataGenerator.generateData(
        "https://championsqueue.gg/matches", numClicks, logger)
    logger.info("Data generation complete!")

    np.savetxt("champions_queue_data.csv", data, delimiter=",", fmt="%s")
    data = np.genfromtxt('champions_queue_data.csv', delimiter=',', dtype='U')
