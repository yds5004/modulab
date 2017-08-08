import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from Tokenizer import Tokenizer

import tensorflow as tf

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

def train(xData, yData, modelName):
    answer = []
    for y in yData:
        if y == 0: answer.append([1, 0])
        else: answer.append([0, 1])

    sequenceSize = len(xData[0])
    learning_rate = 0.001
    beta = 0.001
    training_epochs = 1000
    batch_size = len(xData)
    display_step = 100
    keep_drop_out = 0.8
    hidden_nodes_1 = 1024
    hidden_nodes_2 = int(hidden_nodes_1 * 0.5)
    hidden_nodes_3 = int(hidden_nodes_1 * np.power(0.5, 2))
    hidden_nodes_4 = int(hidden_nodes_1 * np.power(0.5, 3))
    hidden_nodes_5 = int(hidden_nodes_1 * np.power(0.5, 4))

    # tf Graph Input
    X = tf.placeholder(tf.float32, [None, sequenceSize])  # mnist data image of shape 28*28=784
    Y = tf.placeholder(tf.float32, [None, 2])  # 0-9 digits recognition => 10 classes
    keep_prob = tf.placeholder("float")

    # Set model weights
    W1 = tf.Variable(tf.truncated_normal([sequenceSize, hidden_nodes_1], stddev=math.sqrt(2.0 / sequenceSize)))
    W2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0 / hidden_nodes_1)))
    W3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], stddev=math.sqrt(2.0 / hidden_nodes_2)))
    W4 = tf.Variable(tf.truncated_normal([hidden_nodes_3, hidden_nodes_4], stddev=math.sqrt(2.0 / hidden_nodes_3)))
    W5 = tf.Variable(tf.truncated_normal([hidden_nodes_4, hidden_nodes_5], stddev=math.sqrt(2.0 / hidden_nodes_4)))
    W6 = tf.Variable(tf.truncated_normal([hidden_nodes_5, 2], stddev=math.sqrt(2.0 / hidden_nodes_5)))
    b1 = tf.Variable(tf.random_uniform([hidden_nodes_1], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([hidden_nodes_2], -1.0, 1.0))
    b3 = tf.Variable(tf.random_uniform([hidden_nodes_3], -1.0, 1.0))
    b4 = tf.Variable(tf.random_uniform([hidden_nodes_4], -1.0, 1.0))
    b5 = tf.Variable(tf.random_uniform([hidden_nodes_5], -1.0, 1.0))
    b6 = tf.Variable(tf.random_uniform([2], -1.0, 1.0))

    # Construct model
    prediction1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, W1) + b1), keep_prob)
    prediction2 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction1, W2) + b2), keep_prob)
    prediction3 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction2, W3) + b3), keep_prob)
    prediction4 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction3, W4) + b4), keep_prob)
    prediction5 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction4, W5) + b5), keep_prob)
    prediction = tf.nn.softmax(tf.matmul(prediction5, W6) + b6)

    # Minimize error using cross entropy
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=1))
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
    loss = tf.reduce_mean(loss + beta * regularizers)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            _, c = sess.run([optimizer, loss], feed_dict={X: xData, Y: answer, keep_prob: keep_drop_out})

            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))

        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, modelName)

        print("Optimization Finished!")

def predict(xData, modelName):

    sequenceSize = len(xData[0])
    learning_rate = 0.001
    beta = 0.001
    training_epochs = 1000
    batch_size = len(xData)
    display_step = 100
    keep_drop_out = 0.8
    hidden_nodes_1 = 1024
    hidden_nodes_2 = int(hidden_nodes_1 * 0.5)
    hidden_nodes_3 = int(hidden_nodes_1 * np.power(0.5, 2))
    hidden_nodes_4 = int(hidden_nodes_1 * np.power(0.5, 3))
    hidden_nodes_5 = int(hidden_nodes_1 * np.power(0.5, 4))

    # tf Graph Input
    X = tf.placeholder(tf.float32, [None, sequenceSize])  # mnist data image of shape 28*28=784
    Y = tf.placeholder(tf.float32, [None, 2])  # 0-9 digits recognition => 10 classes
    keep_prob = tf.placeholder("float")

    # Set model weights
    W1 = tf.Variable(tf.truncated_normal([sequenceSize, hidden_nodes_1], stddev=math.sqrt(2.0 / sequenceSize)))
    W2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0 / hidden_nodes_1)))
    W3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], stddev=math.sqrt(2.0 / hidden_nodes_2)))
    W4 = tf.Variable(tf.truncated_normal([hidden_nodes_3, hidden_nodes_4], stddev=math.sqrt(2.0 / hidden_nodes_3)))
    W5 = tf.Variable(tf.truncated_normal([hidden_nodes_4, hidden_nodes_5], stddev=math.sqrt(2.0 / hidden_nodes_4)))
    W6 = tf.Variable(tf.truncated_normal([hidden_nodes_5, 2], stddev=math.sqrt(2.0 / hidden_nodes_5)))
    b1 = tf.Variable(tf.random_uniform([hidden_nodes_1], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([hidden_nodes_2], -1.0, 1.0))
    b3 = tf.Variable(tf.random_uniform([hidden_nodes_3], -1.0, 1.0))
    b4 = tf.Variable(tf.random_uniform([hidden_nodes_4], -1.0, 1.0))
    b5 = tf.Variable(tf.random_uniform([hidden_nodes_5], -1.0, 1.0))
    b6 = tf.Variable(tf.random_uniform([2], -1.0, 1.0))

    # Construct model
    prediction1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, W1) + b1), keep_prob)
    prediction2 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction1, W2) + b2), keep_prob)
    prediction3 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction2, W3) + b3), keep_prob)
    prediction4 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction3, W4) + b4), keep_prob)
    prediction5 = tf.nn.dropout(tf.nn.relu(tf.matmul(prediction4, W5) + b5), keep_prob)
    prediction = tf.nn.softmax(tf.matmul(prediction5, W6) + b6)

    # Minimize error using cross entropy
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=1))
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5) + tf.nn.l2_loss(W6)
    loss = tf.reduce_mean(loss + beta * regularizers)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(modelName)
        saver = tf.train.Saver(tf.global_variables())
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        result, result1 = sess.run([prediction, tf.argmax(prediction, 1)], feed_dict={X: xData, keep_prob: 1.0})
        print(result)

if __name__ == '__main__':

    # Read data from files
    train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', 'labeledTrainData 2.tsv'), header=0, delimiter="\t", quoting=3)
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', 'testData.tsv'), header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'resources', "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3)

    stopwords = Tokenizer.readStopWord()
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for review in train_data["review"]:
        sentences.append( Tokenizer.review_to_words(review, stopwords) )
    #showDataLength(sentences)

    vectorSize = 300
    w2v_file = "resources/model/300features_40minwords_10context_skipgram.w2v"
    #model = makeWord2Vec(sentences, vectorSize, w2v_file)
    model = Word2Vec.load(w2v_file)


    # ****** Create average vectors for the training and test sets
    print ("Creating average feature vecs for training reviews")
    ###trainDataVecs = getAvgFeatureVecs(getCleanReviews(train_data, stopwords), model, vectorSize)

    print ("Creating average feature vecs for test reviews")
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test_data, stopwords), model, vectorSize)

    # ****** Fit a random forest to the training set, then make predictions
    model_name = './resources/sentiment.ckpt'
    ###train(trainDataVecs, train_data["sentiment"], model_name)
    predict(testDataVecs, model_name)