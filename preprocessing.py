import pandas as pd

def preprocess(path):
    """Return preprocessed dataframe

    Parameters
    ----------
    path : str
        path of data

    Returns
    -------
    pd.DataFrame
        Preprocessed Dataframe. Data columns are as follows:

        ======  ==============================================================
        id      id of document, e.g. "id18154" (as `str`)
        token   a token in the dataset, e.g. "the" (as `str`)
        author  author of the document that this token belongs to (as `str`)
        ======  ==============================================================

    """