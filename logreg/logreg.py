import random
from numpy import zeros, sign, dot, where, median
from math import exp, log
from collections import defaultdict

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1


class LogReg:
    def __init__(self, num_features, mu, step=lambda x: 0.05):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.beta = zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = defaultdict(int)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """

        eta = self.step(iteration)
        # Unregularlized        
        #for i in range(len(self.beta)-1):
        #    val_pi =  pi(self.beta[i], train_example.x[i])
        #    self.beta[i] = self.beta[i] + eta*(train_example.y-val_pi)*train_example.x[i]

        # Regularlized adds on the mu part. If you want unreg, mu=0 so the last part goes away
        val_pi =  pi(self.beta, train_example.x)
        for i in range(len(self.beta)):
            if (train_example.x[i] != 0):
                self.beta[i] = (self.beta[i] + eta*(train_example.y-val_pi)*train_example.x[i])*(1-2*eta*self.mu)**(self.last_update[i] + 1)
                self.last_update.update({i: 0})
            else:
                val = self.last_update[i] + 1
                self.last_update.update({i: val})
        return self.beta

def pi(beta, x):
    val = (exp(dot(beta,x)))/(1+exp(dot(beta,x)))
    return val


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def step_update(iteration):
    # TODO (extra credit): Update this function to provide an
    # effective iteration dependent step size
    step_size = 1.0
    #Add this so it still passes the unit tests
    #if (iteration == 0 or iteration == 1):
    #    step_size = 1.0    
    #else:
    #    # Start out with something big that gets smaller as a function of iteration
    #    step_size = 1.0-iteration/1000.0
    #    if (step_size <= 0.0):
    #        step_size = 0.05
    return step_size

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.mu, lambda x: args.step)

    # Iterations
    update_number = 0
    for pp in xrange(args.passes):
        for ii in train:
            update_number += 1
            lr.sg_update(ii, update_number)

            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
    # This code outputs the best indicator word for the baseball class, the best for the hockey class, and the worst (that could be etiher class)
    idx1 = lr.beta.argmax(axis=None)
    idx2 = lr.beta.argmin(axis=None)
    idx3 = where(lr.beta==median(lr.beta))[0][0]
    print idx1, idx2, idx3
    print train[idx1].y, train[idx2].y
    print vocab[idx1], vocab[idx2], vocab[idx3]