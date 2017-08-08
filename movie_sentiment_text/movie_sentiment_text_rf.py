import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from Tokenizer import Tokenizer


def showDataLength(sentences):
    # 1. # 첫 번째 컬럼 'num_critic_for_reviews'에 대한 그래프 (w, h - inch)
    fig = plt.figure(figsize=(5, 5))
    # 2x2(행x열)의 그래프
    fig.add_subplot(1, 1, 1)
    lengths = []
    for i in range(len(sentences)):
        lengths.append(len(sentences[i]))
    # 그래프의 유형 (kind를 주지않으면 전체 데이터 출력, kde는 부드러운 선, bar는 막대 그래프
    plt.plot(lengths)
    plt.legend(['sequece lengths'])
    plt.title("Frequency of Sequence Lenghts")
    plt.show()

def makeWord2Vec(sentences, vectorSize, fileName):
    # Set values for various parameters
    vectorSize = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, size=vectorSize, min_count=min_word_count, window=context,
                     sample=downsampling, seed=1, sg=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(fileName)

    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")

    return model

def makeFeatureVec(words, model, vectorSize):
    featureVec = np.zeros((vectorSize,),dtype="float32")
    nwords = 0.

    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, vectorSize):
    counter = 0.

    reviewFeatureVecs = np.zeros((len(reviews),vectorSize),dtype="float32")

    for review in reviews:

       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))

       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, vectorSize)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews, stopwords):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( Tokenizer.review_to_words( review, stopwords))
    return clean_reviews


if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', 'labeledTrainData 2.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)

    stopwords = Tokenizer.readStopWord()
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for review in train["review"]:
        sentences.append( Tokenizer.review_to_words(review, stopwords) )
    #showDataLength(sentences)

    vectorSize = 300
    w2v_file = "resources/model/300features_40minwords_10context_skipgram.w2v"
    #model = makeWord2Vec(sentences, vectorSize, w2v_file)
    model = Word2Vec.load(w2v_file)


    # ****** Create average vectors for the training and test sets
    print ("Creating average feature vecs for training reviews")
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train, stopwords), model, vectorSize)

    print ("Creating average feature vecs for test reviews")
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test, stopwords), model, vectorSize)

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Test & extract results
    print("predicting test data")
    result = forest.predict(testDataVecs)

    result_file = "resources/result/Wrote Word2Vec_AverageVectors.csv"
    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(result_file, index=False, quoting=3)
    print (result_file)