import pandas as pd
from nltk.tokenize import NLTKWordTokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import math
from random import randint
from gensim.matutils import hellinger
from sklearn.metrics import accuracy_score

df = pd.read_json("../preprocessing_output/preprocessed_train_WL.json")
train_data = df.groupby("author").agg(
    token=pd.NamedAgg(column="token", aggfunc=lambda x: x.explode().dropna()),
).reset_index()
vocab = Dictionary(train_data["token"])

num_topics = 50
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = vocab[0]  # This is only to "load" the dictionary.
id2word = vocab.id2token

train_corpus = [vocab.doc2bow(doc) for doc in train_data["token"]]
lda = LdaModel(
    corpus=train_corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=1
)

test_data = pd.read_json("../preprocessing_output/preprocessed_test_WL.json")

test_corpus = [vocab.doc2bow(doc) for doc in test_data["token"]]
print(lda.print_topics())
predictions = []
for i, doc in enumerate(test_corpus):
    if i % 20 == 0:
        print("iteration", i, "of", len(test_corpus))
    min = math.inf
    min_index = -1
    for j, doc2 in enumerate(train_corpus):
        dist = hellinger(lda[doc],
                    lda[doc2])
        if dist < min:
            min = dist
            min_index = j
    predictions.append(train_data["author"].iloc[min_index])

predicdf = pd.DataFrame(test_data["author"])
predicdf["predictions"] = predictions
predicdf
predicdf.groupby(["author", "predictions"]).size().unstack(fill_value=0)

print("accuracy WL", accuracy_score(predicdf["author"], predicdf["predictions"]))
predicdf["predictions"].to_json("./results_output/result_lda_WL.json", orient="values")