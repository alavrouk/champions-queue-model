import h2o  # do we need the entire package? # Probably not lmao
from h2o.estimators import H2ORandomForestEstimator


def runRandomForest(data):
    """
    :param data:
    :return forest:
    """
    h2o.init()
    print("Starting Random Forest...")
    # Create a random forest classifier
    forest = H2ORandomForestEstimator(ntrees=100)

    # create factors
    # labels = ["patch", "outcome",
    #           "Team1Champ1", "Team1Champ2", "Team1Champ3", "Team1Champ4", "Team1Champ5",
    #           "Team2Champ1", "Team2Champ2", "Team2Champ3", "Team2Champ4", "Team2Champ5"]
    # dataCopy = np.vstack([labels, data])
    dataCopy = h2o.upload_file("../data/champions_queue_data.csv")
    dataCopy["outcome"] = dataCopy["outcome"].asfactor()
    predictors = ["Team1Champ1", "Team1Champ2", "Team1Champ3", "Team1Champ4", "Team1Champ5",
                  "Team2Champ1", "Team2Champ2", "Team2Champ3", "Team2Champ4", "Team2Champ5"]
    response = "outcome"

    # Split data into training and test sets
    train, valid = dataCopy.split_frame(ratios=[.8], seed=69420)

    # Train the model on the training data
    forest.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    # Evaluate on the test data
    perf = forest.model_performance()
    print(perf)
    # Return the random forest
    return forest
