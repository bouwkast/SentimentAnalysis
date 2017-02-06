"""
This file will scrape some metrics from our data so that we can build our
Tensors more intelligently - looking for word counts, review length, useless words, etc.
Filename: metrics.py
Author: Steven
Date Last Modified: 2/3/2017
Email: bouwkast@mail.gvsu.edu
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
from html.parser import HTMLParser
import pickle


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return ' '.join(self.fed)


parser = MyHTMLParser()


# going to replace the ignore_sequence with a SPACE
# we will separate each word out into a frequency histogram

def get_frequent_words(filename):
    word_freq = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)  # skip first line
        next(reader)

        try:
            for row in reader:
                review = row[0]
                parser.feed(review)
                review = parser.get_data()
                review = review.lower().split()
                for word in review:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1

            freq_file = open('word_frequency', 'wb')
            pickle.dump(word_freq, freq_file)

        except UnicodeDecodeError:  # some characters we can't decode in the given dataset
            next(reader)


def create_review_len_hist(filename):
    review_word_count = []
    with open('training_movie_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            reader = csv.reader(csvfile)
            for row in reader:
                review_word_count.append(len(row[0].split()))
        except UnicodeDecodeError:  # some characters we can't decode in the given dataset
            next(reader)
    bins = np.arange(0, 750, 50)

    plt.xlim(0, 800)
    plt.hist(review_word_count, bins=bins, alpha=.5)
    plt.title('Average review length in words')
    plt.xlabel('Number of words')
    plt.ylabel('count')
    #  we are making a histogram
    plt.show()

def create_frequency_hist(filename):
    """
    Takes a pickled dictionary and create a histogram of word frequencies
    :param filename: is the pickled dictionary
    :return: an image of the histogram
    """
    file_open = open('word_frequency', 'rb')
    word_dict = pickle.load(file_open)
    x = np.arange(len(word_dict))
    y = word_dict.values()
    bins = np.arange(0, 750, 50)
    plt.xlim(0, 800)

    # plt.bar(len(word_dict), word_dict.values(), align='center')
    plt.bar(x, y, width=100)
    plt.xticks(x , word_dict.keys(), rotation='vertical')
    plt.title('Unique words used in reviews and their counts')
    plt.xlabel('Word')
    plt.ylabel('count')
    #  we are making a histogram
    plt.show()


def create_integer_encoding(filename):
    """
    Return a dictionary with the key as the word and the value is its popularity

    Example - the word 'the' might be the most frequently used word in the data
    however, the word 'the' doesn't help us to classify the data - so it's integer
    encoding would be 1 and we could then exclude it from training.
    :param filename: dataset csv file
    :return: a dictionary of encodings
    """
    review_word_count = []
    with open('training_movie_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            reader = csv.reader(csvfile)
            for row in reader:
                review_word_count.append(len(row[0].split()))
        except UnicodeDecodeError:  # some characters we can't decode in the given dataset
            next(reader)


# get_frequent_words('training_movie_data.csv')
create_frequency_hist('word_frequency')