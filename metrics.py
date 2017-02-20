"""
This file will scrape some metrics from our data so that we can build our
Tensors more intelligently - looking for word counts, review length, useless words, etc.
Filename: metrics.py
Author: Steven Bouwkamp
Date Last Modified: 2/6/2017
Email: bouwkast@mail.gvsu.edu
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
from html.parser import HTMLParser
import pickle
import pandas as pd
import re

from bs4 import BeautifulSoup



#  This uses the python HTLMParser to find and remove all HTML elements from our data
from nltk import PorterStemmer, wordpunct_tokenize
from nltk.corpus import stopwords

# Parses through the data to remove any HTML code that would cause errors with checking the data.
class MyHTMLParser(HTMLParser):
    def error(self, message):
        pass

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def clear_data(self):
        self.fed = []

    def get_data(self):
        return ' '.join(self.fed)

# Removes the suffixes of words so they'll be read along with their root word.
# For example, "run," "running," and "runner" are all treated as the same word.
porter_stem = PorterStemmer()

# Sets the list of words to ignore in terms of sentiment analysis. Starts with a pre-determined set of words,
# and then we personally add more; "I", "it," and various bits of punctuation.
stop_words = set(stopwords.words('english')).union({'.', 'I', 'i', ',', '\'', 'it', '*', '?', '/', '-', '&', '<', '>', '\"', ':'})

# Returns the text as a series of tokens.
def tokenizer(text):
    new_text = [porter_stem.stem(word) for word in text.lower().split()]
    return ' '.join(new_text)


# Tokenizes everything before we stem it, and returns paranoid
def preprocessor(text):
    new_text = re.sub("[^a-zA-Z]", " ", text.lower())
    return new_text


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

# Takes a pickled dictionary and create a histogram of word frequencies
def create_frequency_hist(filename):
    """
    :param filename: is the pickled dictionary
    :return: an image of the histogram
    """
    file_open = open('word_frequency', 'rb')
    word_dict = pickle.load(file_open)
    x = np.arange(len(word_dict))
    y = word_dict.values()
    bins = np.arange(0, 750, 50)
    plt.xlim(0, 800)

    plt.bar(x, y, width=100)
    plt.xticks(x, word_dict.keys(), rotation='vertical')
    plt.title('Unique words used in reviews and their counts')
    plt.xlabel('Word')
    plt.ylabel('count')

    plt.show()

# Write a csv file of the cleaned file by removing HTML and unknown characters.
def remove_unknown(filename):
    """
    :param filename: is the name of the file to clean
    :return: None - writes to csv file the clean data
    """
    file = filename.split('.')
    file[0] += '_cleaned'

    file = '.'.join(file)  # taking our input data and splicing 'cleaned' onto it
    #  TODO - need to use df.drop() somehow to remove unknown characters (I think)
    df = pd.read_csv(filename, encoding='utf-8', keep_default_na=True)

    for i in range(0, len(df)):
        parser = MyHTMLParser()
        parser.feed(df['review'][i])
        row = parser.get_data()
        new_row = ''
        for char in row:
            if ord(char) < 128:
                new_row += char
        df.set_value(i, 'review', new_row)
        parser.clear_data()

    df.to_csv(path_or_buf=file, index=False, encoding='utf-8', na_rep=' ')


def create_integer_encoding(filename):
    """
    Return a dictionary with the key as the word and the value is its popularity

    Example - the word 'the' might be the most frequently used word in the data
    however, the word 'the' doesn't help us to classify the data - so it's integer
    encoding would be 1 and we could then exclude it from training.

    With this we could simply go and tell our training model to exclude the 5 most common words
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


def readme_examples(filename):
    """
    Help to show what we are doing in for the GitHub README
    :param filename: name of the file that we want to clean
    :return: the final clean data
    """

    # Read data set into a pandas dataframe.
    df = pd.read_csv(filename, encoding='utf-8', keep_default_na=True)

    test_review = df['review'][15]
    print('Raw: ' + test_review)
    parser = MyHTMLParser()
    parser.feed(test_review)
    test_review = parser.get_data()
    print('HTML removed: ' + test_review)

    no_unknown = ''
    # Check to make sure ASCII value is within range.
    for character in test_review:
        if ord(character) < 128:
            no_unknown += character

    print('Foreign characters removed: ' + no_unknown)

    no_stop_words = preprocessor(no_unknown)

    print('All stopwords gone: ' + no_stop_words)

    # Last step is to tokenize, and return.
    tokenized = tokenizer(no_stop_words)
    print(tokenized)
