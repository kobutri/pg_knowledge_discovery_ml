import pandas as pd
from nltk.tokenize import NLTKWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import json
import string
from math import floor

# naming convention: preprocessing_<W|C>[L][S][P][H<cutoff>]
#   W: with words
#   C: with chunks
#   L: with lemmatization
#   S: with stopwords
#   P: with punctuation
#   H: with highpass

def preprocessing_W(path):
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
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _lemmatize(df.token)
    df.token = _remove_punctuation(df.token)
    df.token = _remove_stopwords(df.token)
    return df


def preprocessing_L(path):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _lemmatize(df.token)
    df.token = _remove_punctuation(df.token)
    df.token = _remove_stopwords(df.token)
    return df

def preprocessing_SP(path):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    return df

def preprocessing_SPC(path):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _chunks(df.token)
    return df

def preprocessing_C(path):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _remove_punctuation(df.token)
    df.token = _remove_stopwords(df.token)
    df.token = _chunks(df.token)
    return df

def preprocessing_SPH(path, cutoff):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _high_pass(df.token, cutoff)
    return df

def preprocessing_SPHC(path, cutoff):
    df = _load(path)
    df.token = _lower(df.token)
    df.token = _tokenize(df.token)
    df.token = _chunks(df.token)
    df.token = _high_pass(df.token, cutoff)
    return df


# utilitiy functions to facilitate preprocessing

def _load(path):
    file =  open(path, "r")
    data = json.load(file)
    return pd.DataFrame(data, columns=['id', 'token', 'author'])

def _lower(docs: pd.Series):
    return docs.apply(lambda s: s.lower())

def _tokenize(docs: pd.Series):
    return docs.apply(lambda s: [s[start:end] for start, end in NLTKWordTokenizer().span_tokenize(s)])

def _lemmatize(docs: pd.Series):
    def pos_tagger(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return None
        
    lemm = WordNetLemmatizer()
    return docs.apply(lambda doc: nltk.pos_tag(doc)) \
        .apply(lambda doc: list(map(lambda x: (x[0], pos_tagger(x[1])), doc))) \
        .apply(lambda doc: list(map(lambda x: x[0] if x[1] == None else lemm.lemmatize(x[0], x[1]), doc)))

def _remove_stopwords(docs: pd.Series):
    stop_words = set(stopwords.words('english'))
    return docs.apply(lambda doc: [w for w in doc if not w in stop_words])

def _remove_punctuation(docs: pd.Series):
    return docs.apply(lambda doc: [w for w in doc if not w in string.punctuation])

def _high_pass(docs: pd.Series, cutoff: float):
    tokens = docs.explode()
    dist = nltk.FreqDist(tokens)
    n = len(tokens)
    return docs.apply(lambda doc: [w for w in doc if dist[w] > n * cutoff])

def _chunks(doc: pd.Series):
    joined = doc.apply(lambda doc: "".join(doc))
    return joined.apply(lambda doc: ["".join([doc[3*i], doc[3*i+1], doc[3*i+2]]) for i in range(floor(len(doc)/3))])

if __name__ == "__main__":
    for path in ["new_train.json", "new_test.json"]:
        preprocessing_W(path).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_W.json', orient='records')
        preprocessing_L(path).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WL.json', orient='records')
        preprocessing_SP(path).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSP.json', orient='records')
        preprocessing_SPC(path).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSP.json', orient='records')
        preprocessing_C(path).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_C.json', orient='records')
        preprocessing_SPH(path, 5e-4).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSPH5e-4.json', orient='records')
        preprocessing_SPHC(path, 5e-4).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSPH5e-4.json', orient='records')
        preprocessing_SPH(path, 1e-4).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSPH1e-4.json', orient='records')
        preprocessing_SPHC(path, 1e-4).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSPH1e-4.json', orient='records')
        preprocessing_SPH(path, 5e-5).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSPH5e-5.json', orient='records')
        preprocessing_SPHC(path, 5e-5).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSPH5e-5.json', orient='records')
        preprocessing_SPH(path, 1e-5).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSPH1e-5.json', orient='records')
        preprocessing_SPHC(path, 1e-5).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSPH1e-5.json', orient='records')
        preprocessing_SPH(path, 1e-6).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_WSPH1e-6.json', orient='records')
        preprocessing_SPHC(path, 1e-6).to_json(f'preprocessing_output/preprocessed_{"train" if "train" in path else "test"}_CSPH1e-6.json', orient='records')
        