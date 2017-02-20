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
import re
#  Try to not run it on the whole dataframe - only the first 10 are necessary
# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data_cleaned.csv')
print(df.head(5))
np.random.seed(0)  # seed for reproducibility

df = df.reindex(np.random.permutation(df.index))

print(df.head(5))
to_remove = {'and', 'or', 'i', 'be', 'a'}
# Let's specify what other type of words we want removed
test_review = df['review'][31]  # This will be different each time because we are shuffling the indices
print('INITIAL REVIEW: \n' + test_review + '\n')
porter_stem = PorterStemmer()
nltk.download('punkt')
# Removes almost all punctuation
stop_words = set(stopwords.words('english')).union({'.', 'I', 'i', ',', '\'', 'it', '*'})

print(stop_words)  # to visually see what we are going to remove

def remove_stop_words(text):
    #  Tokenizes everything before we stem it

    # Prints the initial tokenized text.
    normal_tokens = word_tokenize(text)
    print('Normal Tokenizer: ' + str(normal_tokens) + '\n')
    # Print the tokenized text, but with all the punctuation separated from words.
    word_punct = wordpunct_tokenize(text)
    print('Word Punct Tokenizer: ' + str(word_punct) + '\n')
    # Prints the tokenized text, but removes all stop-words like "is" and "and."
    new_normal = [word for word in normal_tokens if word not in stop_words]
    print('New Normal Text: ' + str(new_normal) + '\n')
    # Prints the tokenized text with stop-words removed,, and with all punctuation separated from the words.
    new_punct = [word for word in word_punct if word not in stop_words]
    print('New Punct Text: ' + str(new_punct) + '\n')

    output = ' '.join(new_punct)
    letters_only = re.sub("[^a-zA-Z]",  " ", output)
    print(letters_only)
    # After running - it seems like new_punct is the best
    return letters_only

# Prints the final string after it's been run through the above methods to remove punctuation and stop-words.
def test_tokenizer(text):
    test = [porter_stem.stem(word) for word in text.split()]
    print('PORTER STEMMER: ' + str(test))
    return [porter_stem.stem(word) for word in text.split()]


cleaned = remove_stop_words(test_review.lower())
test_tokenizer(cleaned)


