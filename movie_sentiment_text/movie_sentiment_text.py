import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from Tokenizer import Tokenizer




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


    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print ("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling, seed=1, sg=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "resources/model/300features_40minwords_10context_skipgram.w2v"
    model.save(model_name)

    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")


    """
    # ****** Create average vectors for the training and test sets
    #
    print ("Creating average feature vecs for training reviews")

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print ("Creating average feature vecs for test reviews")

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print ("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print ("Wrote Word2Vec_AverageVectors.csv")
    """