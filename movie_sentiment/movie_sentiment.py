import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math
import sys

"""
kaggle에서 다운받은 영화 데이터를 사용했다.
(https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset)

imdb_score가 7보다 작으면 부정, 7보다 크거나 같으면 긍정 평가로 간주하며,
아래 feature를 이용하여 logistic regression을 수행한다.

num_critic_for_reviews
duration
director_facebook_likes
actor_3_facebook_likes
actor_2_facebook_likes
actor_1_facebook_likes
gross
num_voted_users
cast_total_facebook_likes
facenumber_in_poster
num_user_for_reviews
budget
aspect_ratio
movie_facebook_likes
"""

def getValue(str):
    value = str
    if math.isnan(str): value = 0.0
    return value

def normalize(data):
    """ set to 0 ~ 1 """
    colCount = len(data[0]) - 1 # label do not normalize
    for i in range(colCount):
        minValue = sys.maxsize
        maxValue = 0
        for j in range(len(data)):
            if data[j][i] < minValue: minValue = data[j][i]
            if data[j][i] > maxValue: maxValue = data[j][i]

        diff = maxValue - minValue
        for j in range(len(data)):
            data[j][i] = (data[j][i] - minValue) / diff

    return data

def readResources(inFileName):
    data = []
    raw_data = pd.read_csv(inFileName)
    dataCount = raw_data.shape[0]
    for i in range(dataCount):
        item = []
        item.append(getValue(raw_data['num_critic_for_reviews'][i]))
        item.append(getValue(raw_data['duration'][i]))
        item.append(getValue(raw_data['director_facebook_likes'][i]))
        item.append(getValue(raw_data['actor_3_facebook_likes'][i]))
        item.append(getValue(raw_data['actor_2_facebook_likes'][i]))
        item.append(getValue(raw_data['actor_1_facebook_likes'][i]))
        item.append(getValue(raw_data['gross'][i]))
        #item.append(getValue(raw_data['num_voted_users'][i]))
        item.append(getValue(raw_data['cast_total_facebook_likes'][i]))
        #item.append(getValue(raw_data['facenumber_in_poster'][i]))
        #item.append(getValue(raw_data['num_user_for_reviews'][i]))
        item.append(getValue(raw_data['budget'][i]))
        #item.append(getValue(raw_data['aspect_ratio'][i]))
        item.append(getValue(raw_data['movie_facebook_likes'][i]))

        imdb_score = raw_data['imdb_score'][i]
        sentiment = 0
        if (float(imdb_score) >= 7.0):
            sentiment = 1
        item.append(sentiment)
        data.append(item)

    # normalize
    normalizedData = normalize(data)
    random.shuffle(normalizedData)
    train_data = normalizedData[:int(dataCount * 0.8)]
    test_data = normalizedData[int(dataCount * 0.8):]

    return train_data, test_data

def getXYData(data, dataColumnSize):
    xData = []
    yData = []
    for i in range(len(data)):
        xData.append(data[i][0:dataColumnSize])
        if (data[i][dataColumnSize] ==0):
            yData.append([1, 0])
        else:
            yData.append([0, 1])

    #return np.array(xData, dtype='float'), np.array(yData, dtype='float')
    return xData, yData


train_data, test_data = readResources('resources/movie_metadata.csv')
data_column_size = len(train_data[0]) - 1 # -1 is label
train_x_data, train_y_data = getXYData(train_data, data_column_size)
test_x_data, test_y_data = getXYData(test_data, data_column_size)

# Parameters
learning_rate = 0.01
training_epochs = 2000
batch_size = len(train_x_data)
display_step = 100

# tf Graph Input
X = tf.placeholder(tf.float32, [None, data_column_size]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
#W = tf.Variable(tf.zeros([13, 2]))
#b = tf.Variable(tf.zeros([2]))
W = tf.Variable(tf.random_uniform([data_column_size, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([2], -1.0, 1.0))

# Construct model
prediction = tf.nn.softmax(tf.matmul(X, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(prediction), reduction_indices=1))

# Gradient Descent
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    w, b, x, y = sess.run([W, b, X, Y], feed_dict={X: train_x_data, Y: train_y_data})
    print ("w: ", w)
    print("b: ", b)
    print("x: ", x)
    print("y: ", y)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        _, c = sess.run([optimizer, cost], feed_dict={X: train_x_data, Y: train_y_data})

        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

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
    print ("result: ", result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6], result[0][7], result[0][8], result[0][9])
    print("y: ", answer[0], answer[1], answer[2], answer[3], answer[4], answer[5], answer[6], answer[7], answer[8], answer[9])
