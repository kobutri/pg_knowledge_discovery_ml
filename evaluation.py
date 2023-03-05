import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

def evaluate(input):
    """Return dataframe with random data.

    Parameters
    ----------
    input : pd.DataFrame
        Data frame with labels and results. Data columns are as follows:

        ======== ==============================================================
        author   actual author of document (as `str`)
        <method> predicted author of document with <method> (as `str`)
        ...
        ======== ==============================================================
    """
    pass

#utility functions

def plotOverview(input):
    # calc metrics
    input.iloc[:, 1:].apply(lambda x: (accuracy_score(x), precision_score(x), recall_score(x), f1_score(x)))
    # pivot table
    pass

def plotMethodOverview(input, title):
    pass

def plotConfusionMatrix(input, title):
    pass

def plotPerAuthor(input, title):
    pass

def plotROC(input, title):
    pass

def plotTable(input, title):
    pass

if __name__ == "__main__":
    pass
