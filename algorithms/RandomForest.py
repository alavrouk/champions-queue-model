import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

from util.DataTransformations import clusteringTransform


def runRandomForest(data, logger):
    """
    :param data:
    :return forest:
    """

    rfData = clusteringTransform(data)
    print(rfData)

    rfData = pd.DataFrame({'team1': rfData[:, 1], 'team2': rfData[:, 2], 'result': rfData[:, 0]})
    print(rfData)

    X = rfData[['team1', 'team2']]
    y = rfData['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)

    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    plt.show()