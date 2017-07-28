import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.utils import shuffle
import math
import sys

"""
kaggle에서 다운받은 영화 데이터를 사용했다.
(https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset)

imdb_score가 7보다 작으면 부정, 7보다 크거나 같으면 긍정 평가로 간주하며,
아래 feature를 이용하여 logistic regression을 수행한다.

(전체적으로 Pandas를 사용했지만, 맞게 쓴 것인지는 의문)
"""

POSITIVE_THRETHOLD = 7.0
LABEL_FIELD_NAME = 'sentiment'

def replaceNaN(str):
    value = str
    if math.isnan(str): value = float(0)
    return float(value)

def normalize(data, dataTitle):
    """ set to 0 ~ 1 """
    dataFrame = DataFrame()
    for i in range(len(dataTitle)):
        minValue = sys.maxsize
        maxValue = 0
        tempData = data[dataTitle[i]]
        values = []
        for j in range(len(tempData)):
            if tempData[j] < minValue: minValue = tempData[j]
            if tempData[j] > maxValue: maxValue = tempData[j]

        diff = maxValue - minValue
        for j in range(len(tempData)):
            values.append( (tempData[j] - minValue) / diff )

        dataFrame[dataTitle[i]] = values
    return dataFrame

def readResources(inFileName, dataTitle):
    data = {}
    raw_data = pd.read_csv(inFileName)

    for i in range(len(dataTitle)):
        raw_data[dataTitle[i]] = raw_data[dataTitle[i]].apply(replaceNaN)

    sentimentField = []
    for i in range(raw_data['imdb_score'].shape[0]):
        if (float(raw_data['imdb_score'][i]) >= POSITIVE_THRETHOLD):
            sentimentField.append([0, 1])
        else:
            sentimentField.append([1, 0])

    raw_data[LABEL_FIELD_NAME] = sentimentField

    # normalize data
    normalizedItems = normalize(raw_data, dataTitle)
    normalizedItems[LABEL_FIELD_NAME] = raw_data[LABEL_FIELD_NAME]

    # shuffle data
    #shffledItems = shuffle(normalizedItems)
    shffledItems = normalizedItems
    trainDataFrame = DataFrame()
    testDataFrame = DataFrame()
    dataCount = raw_data.shape[0]
    for i in range(len(dataTitle)):
        trainDataFrame[dataTitle[i]] = shffledItems[dataTitle[i]][:int(dataCount * 0.8)]
        testDataFrame[dataTitle[i]] = shffledItems[dataTitle[i]][int(dataCount * 0.8):]
    trainDataFrame[LABEL_FIELD_NAME] = shffledItems[LABEL_FIELD_NAME][:int(dataCount * 0.8)]
    testDataFrame[LABEL_FIELD_NAME] = shffledItems[LABEL_FIELD_NAME][int(dataCount * 0.8):]

    return trainDataFrame, testDataFrame

def getXYData(data, dataTitle):
    xData = []
    yData = []
    columnSize = len(dataTitle)
    matrix = data.values
    for i in range(len(matrix)):
        xData.append(matrix[i][:columnSize])
        yData.append(matrix[i][columnSize])

    return xData, yData

def showDataDistribution(trainData):
    return 0

# 사용할 title
data_Title=['num_critic_for_reviews', 'duration','director_facebook_likes','actor_3_facebook_likes',
            'actor_2_facebook_likes', 'actor_1_facebook_likes','gross','num_voted_users',
            'cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews','budget','aspect_ratio',
            'movie_facebook_likes', 'imdb_score']

train_data_frame, test_data_frame = readResources('resources/movie_metadata.csv', data_Title)
showDataDistribution(train_data_frame)
train_x_data, train_y_data = getXYData(train_data_frame, data_Title)
test_x_data, test_y_data = getXYData(test_data_frame, data_Title)

# Parameters
learning_rate = 0.01
beta = 0.01
training_epochs = 2500
batch_size = len(train_x_data)
display_step = 100


# tf Graph Input
X = tf.placeholder(tf.float32, [None, len(data_Title)]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.random_uniform([len(data_Title), 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([2], -1.0, 1.0))

# Construct model
prediction = tf.nn.softmax(tf.matmul(X, W) + b)

# Minimize error using cross entropy
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, Y))
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(prediction), reduction_indices=1))
regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(loss + beta * regularizer)

# Gradient Descent
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
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
        _, c = sess.run([optimizer, loss], feed_dict={X: train_x_data, Y: train_y_data})

        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(c))

    print ("Optimization Finished!")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    correct = 0
    wrong = 0
    # prediction
    result = sess.run([tf.argmax(prediction, 1)], feed_dict={X: test_x_data})
    answer = np.argmax(test_y_data, 1)
    for i in range(len(answer)):
        if (answer[i] == result[0][i]):
            correct += 1
        else:
            wrong += 1
    print ("accuracy: ", float(correct)/float(correct + wrong))