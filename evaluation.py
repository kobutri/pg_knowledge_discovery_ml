import pandas as pd

def evaluate(input):
    """Return dataframe with random data.

    Parameters
    ----------
    no_rows : pd.DataFrame
        Data frame with labels and results. Data columns are as follows:

        ======== ==============================================================
        author   actual author of document (as `str`)
        <method> predicted author of document with <method> (as `str`)
        ...
        ======== ==============================================================
    """