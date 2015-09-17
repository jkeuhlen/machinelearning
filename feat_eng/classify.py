# Completed by Joseph Keuhlen
from csv import DictReader, DictWriter

import numpy as np
from numpy import array
import re
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, HashingVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

def words_and_char_grams(examples):
    words = re.findall("\w+", examples)
    bigrams = re.findall(r"\b\w+\s\w+", examples)
    trigrams = re.findall(r"\b\w+\s\w+\s\w+", examples)
    quadgrams = re.findall(r"\b\w+\s\w+\s\w+\s\w+", examples)
    result = list()
    for q in quadgrams:
        result.append(q)
    for t in trigrams:
        result.append(t)
    for b in bigrams:
        result.append(b)
    for w in words:
        result.append(w)
        for i in range(len(w)):
            for j in range (len(w)):
                result.append(w[i:i+j])
    return result


class Featurizer:
    def __init__(self):
        # Build a list of stop words that I don't want to use as features. These are often '.' but maybe other ones down the road
        my_stop_words = ['.', '(', ')', ' ', ' .', '..', ').', ' )', ' , ', ' ,']
        stop_words = ENGLISH_STOP_WORDS.union(my_stop_words)
        self.vectorizer = CountVectorizer(analyzer=words_and_char_grams, ngram_range=(1,10), stop_words=stop_words)
        #self.vectorizer = HashingVectorizer(analyzer='char', ngram_range=(1,50), stop_words=stop_words)

    def train_feature(self, examples):
        print self.vectorizer.get_params()
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        print len(feature_names)
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos:%s" % "|".join(feature_names[top10]))
            print("Neg:%s" % "|".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

#    feat = Featurizer()
#
#    labels = []
#    for line in train:
#        if not line[kTARGET_FIELD] in labels:
#            labels.append(line[kTARGET_FIELD])
#
#    print("Label set: %s" % str(labels))
#    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
#    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
#
#    y_train = array(list(labels.index(x[kTARGET_FIELD])
#                         for x in train))
#    print(len(train), len(y_train))
#    print(set(y_train))
#
#    # Train classifier
#    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#    lr.fit(x_train, y_train)
#
#    feat.show_top10(lr, labels)
#
#    predictions = lr.predict(x_test)
#    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
#    o.writeheader()
#    for ii, pp in zip([x['id'] for x in test], predictions):
#        d = {'id': ii, 'spoiler': labels[pp]}
#        o.writerow(d)

    # Create a development set from the training set
    idx = int(round(len(train)*0.7))
    idx2 = int(round(len(train)*0.3))
    dev_train = train[:idx]
    dev_test = train[-idx2:]

    # Redo everything we just did but on the dev set to calc accuracy
    dev_feat = Featurizer()

    dev_labels = []
    for line in dev_train:
        if not line[kTARGET_FIELD] in dev_labels:
            dev_labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(dev_labels))
    dev_x_train = dev_feat.train_feature(x[kTEXT_FIELD] for x in dev_train)
    dev_x_test = dev_feat.test_feature(x[kTEXT_FIELD] for x in dev_test)
    dev_y_train = array(list(dev_labels.index(x[kTARGET_FIELD])
                         for x in dev_train))
    print(len(dev_train), len(dev_y_train))
    print(set(dev_y_train))

    # Train classifier
    dev_lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    dev_lr.fit(dev_x_train, dev_y_train)

    dev_feat.show_top10(dev_lr, dev_labels)

    dev_predictions = dev_lr.predict(dev_x_test)
    # Now use dev_predictions and dev_test[i][kTARGET_FIELD] to compute accuracy
    count = len(dev_predictions)
    accuracy = 0
    for i in range(0, count):
        if (str(bool(dev_predictions[i])) == dev_test[i][kTARGET_FIELD]):
            accuracy += 1
    print "Accuracy: ", float(accuracy)/count*100.0

