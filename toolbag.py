import numpy as np
from numpy.random import permutation

class WordBag:
    def __init__(self):
        self.b = 0
        self.word_weights = 0
        self.learning_rate = 0.001
        self.ridge_lambda = 0.001
        self.seen_words = {}
        self.documents_bag = {}

    def read_file(self, file_name, review):

        # list of individual words that appeared
        words = {}

        # read the file contents
        file = open(file_name, encoding="utf8")
        text = file.read()
        file.close()

        # split the text up and save the data
        text = text.split()
        for word in text:
            if word not in words:
                words[word] = word
            if word not in self.seen_words:
                self.seen_words[word] = len(self.seen_words)

        # put the list of unique words in the document bag
        self.documents_bag[len(self.documents_bag)] = [words, review]

        return words

    def vectorize_data(self):
        # matrix to return
        data = np.zeros((len(self.documents_bag), len(self.seen_words)))
        # truth of positive or negative review
        truth = np.zeros(len(self.documents_bag))

        # change seen_words to numpy array
        seen_words = np.array(list(self.seen_words.keys()))
        for word in self.seen_words:
            seen_words[self.seen_words[word]] = word

        # look to check which words are in which documents
        for w, word in enumerate(seen_words):
            for d, document in enumerate(self.documents_bag.values()):
                truth[d] = document[1]
                if word in document[0]:
                    data[d, w] = 1

        # save vectors to the object
        self.data = data
        self.truth = truth

        return data, truth

    # read the movie reviews and stor their info
    def read_movie_reviews(self, start = 0, stop=1000, step = 1, read_positive = True, read_negative = True, vectorize = True):
        
        for i in range(start, stop, step):

            if read_positive:
                pos_file_name = "reviews/pos/{}.txt".format(i)
                self.read_file(pos_file_name, 1)

            if read_negative:
                neg_file_name = "reviews/neg/{}.txt".format(i)
                self.read_file(neg_file_name, 0)
            
        if vectorize:
            self.vectorize_data()
            self.set_weights()

    # set the weights of each word to be 0
    def set_weights(self):
        self.word_weights = np.zeros((len(self.seen_words)))

    # change the learning rate
    def set_learning_speed(self, rate):
        self.learing_rate = rate

    # update the weights and bias
    def update(self, count = 1):
        f = lambda u: 1 / (1 + np.exp(-u))
        for c in range(count):
            weights = np.zeros_like(self.word_weights)
            b = 0
            for i in range(len(self.documents_bag)):
                y = self.truth[i]
                words = self.data[i]
                j = np.dot(words, self.word_weights)
                increment = self.learning_rate * (y - f(self.b + j))
                weights[words == 1] += increment
                b += increment

            self.word_weights += weights
            self.b += b

    def update_ridge_regularization(self, count = 1):
        f = lambda u: 1 / (1 + np.exp(-u))
        for c in range(count):
            weights = np.zeros_like(self.word_weights)
            b = 0
            for i in range(len(self.documents_bag)):
                y = self.truth[i]
                words = self.data[i]
                j = np.dot(words, self.word_weights)
                increment = self.learning_rate * (y - f(self.b + j))
                weights[words == 1] += increment
                b += increment

            self.word_weights += (weights - (self.ridge_lambda * np.sum(self.word_weights)))
            self.b += b

    def score_file(self, file_name):
        score = 0

        # list of individual words that appeared

        # read the file contents
        file = open(file_name, encoding="utf8")
        text = file.read()
        file.close()

        # split the text up and save the data
        text = text.split()

        for word in text:
            if word in self.seen_words:
                score += self.word_weights[self.seen_words[word]]

        return score


def random_subgroup(group, subgroup_size=1):
    shuffled = np.random.permutation(group)
    subgroup = shuffled[:subgroup_size]
    return subgroup

