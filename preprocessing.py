import pandas as pd
from nltk.tokenize import NLTKWordTokenizer
import json

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
        token   a token in the dataset, e.g. "the" (as `list(str)`)
        author  author of the document that this token belongs to (as `str`)
        ======  ==============================================================
    """


def full_without_highpass(path):
    file =  open(path, "r")
    data = json.load(file)
    df = pd.DataFrame(data, columns=['id', 'token', 'author'])
    df.token = df.token.apply(lambda s: s.lower())
    df['token'] = df.token.apply(lambda s: [s[start:end] for start, end in NLTKWordTokenizer().span_tokenize(s)])

    return df

if __name__ == "__main__":
    print(full_without_highpass("new_train.json").head())