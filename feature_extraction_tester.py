"""
The purpose of this file is to be able to quickly test different ways of extracting features
then we can implement these into our main pipeline.
"""
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import nltk
import numpy as np
import pandas as pd

#  Try to not run it on the whole dataframe - only the first 10 are necessary
# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data_cleaned.csv')
print(df.head(5))
# print(df.index)
np.random.seed(0)  # seed for reproducibility COMMENT OUT TO REMOVE RANDOMIZATION
# TODO - actually not sure what I'm doing wrong because the randomization doesn't seem to work
df = df.reindex(np.random.permutation(df.index))  # COMMENT OUT TO REMOVE RANDOMIZATION
# print(df.index)
print(df.head(5))
to_remove = {'and', 'or', 'i', 'be', 'a'}
# let's specify what other type of words we want removed
test_review = df['review'][5]  # this will be different each time because we are shuffling the indices
print('INITIAL REVIEW: \n' + test_review + '\n')
porter_stem = PorterStemmer()
# nltk.download('punkt')
# this one we are removing pretty much all punctuation TODO is this too much?
stop_words = set(stopwords.words('english')).union({'.', 'I', 'i', ',', '\'', 'it', '*'})
# stop_words = set(stopwords.words('english'))  # This is the default set of stopwords to remove
print(stop_words)  # to visually see what we are going to remove

def remove_stop_words(text):
    #  let's tokenize everything before we stem it
    normal_tokens = word_tokenize(text)
    print('Normal Tokenizer: ' + str(normal_tokens) + '\n')
    word_punct = wordpunct_tokenize(text)
    print('Word Punct Tokenizer: ' + str(word_punct) + '\n')
    new_normal = [word for word in normal_tokens if word not in stop_words]
    print('New Normal Text: ' + str(new_normal) + '\n')
    new_punct = [word for word in word_punct if word not in stop_words]
    print('New Punct Text: ' + str(new_punct) + '\n')

    # After running - it seems like new_punct is the best
    return ' '.join(new_punct)

def tokenizer(text):
    test = [porter_stem.stem(word) for word in text.split()]
    print('PORTER STEMMER: ' + str(test))
    return [porter_stem.stem(word) for word in text.split()]

cleaned = remove_stop_words(test_review.lower())
tokenizer(cleaned)

