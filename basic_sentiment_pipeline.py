"""
    Author(s): Steven Bouwkamp, Andy Vuong, and Matt Schuch
    Dataset used: 45K IMDB movie reviews with either a 0 as having a negative sentiment or a 1 having
                    a positive sentiment.
    Average Accuracy Obtained: 91.5%
    
    Trains a model to be able to predict whether a given movie review is positive
    or negative with around a 91.5% accuracy.

    It utilizes scikit-learn's TfidfVectorizer to transform reviews into a Bag of Words encoding
    and then applies an L2 regularization formula to all of the values. Through extensive
    parameter searching we've been able to tune it to solve decently solve this problem.

    After this step, a Pipeline is built with TfidfVectorizer with a Stochastic Gradient Descent (SGD)
    algorithm as the classification algorithm. The benefit from using SGD as our classifier was being able
    to test out different loss functions that would handle outliers better than standard log loss.

    To ensure that we aren't overfitting our model we have regularization done by the TfidifVectorizer
    and also cross-validate on 10 folds to get a rough estimate for how well the model will predict
    unseen data.

    At the end, we are able to train on all of the data that we have available based off of relatively
    decent results from the cross-validation scores.
"""


# TODO - clean up the other helper files (need to turn in most likely)
# TODO - remove print statements of the cross validation scores
# TODO - maybe make a logger? might help for overall completeness

# TODO - WRITE UP
# TODO - finish the README


# TODO - ON FEB 22nd turn repo to public (unless people haven't turned in)


import pandas as pd
import pickle
import numpy as np
import re

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import SGDClassifier as SGD

#  porter stemming seemed to reduce our accuracy by around .1-.5%
#  think it has to do with how it handles contractions
porter_stem = PorterStemmer()

#  these are the default stopwords along with some added punctuation
#  no longer use this due to the addition of bigrams
#  by removing stop words we could potentially change the actual meaning of the review
stop_words = set(stopwords.words('english')).union(
    {'.', 'I', 'i', ',', '\'', 'it', '*', '?', '/', '-', '&', '<', '>', '\"'})


# The commented out lines for tokenizer and preprocessor are the more 'advanced'
# preprocessing steps - but they lower accuracy, so the basic ones are in there instead.


def tokenizer(text):
    """
    Return a list of words

    When stemming the words we found that it would consistently reduce our accuracy that
    we were able to achieve.
    :param text: is the String to split on spaces
    :return: a list containing all of the individual words that were in the String
    """
    return text.split()
    # return [porter_stem.stem(word) for word in text.split()]


def preprocessor(text):
    """
    Return a String that only contains the characters 'a' to 'z' and 'A' to 'Z'

    Originally, we would use wordpunct to tokenize the input review String -
    it would then check to see if each element in the list was in our set of stopwords.
    If it was in the set - it would be removed.

    This actually reduced our accuracy that we were able to get - through cross-validation
    we averaged around 2% lower than without it.

    We think that the reason why this was happening is because we used bigrams as an
    additional feature - removing words actually changed the meaning of these.
    :param text: is the String containing the movie review to process
    :return: the processed String
    """
    new_text = re.sub("[^a-zA-Z]", " ", text.lower())
    # word_punct = wordpunct_tokenize(text.lower())
    # new_punct = [word for word in word_punct if word not in stop_words]
    # output = ' '.join(new_punct)
    return new_text
    # return output


# this is to protect for running multiple jobs for gridsearchcv when on Windows
if __name__ == '__main__':
    # Read in the dataset and store in a pandas dataframe
    df = pd.read_csv('./training_movie_data_cleaned.csv')
    #  we could print out the seed_int so we could use it on later runs
    seed_int = np.random.randint(low=0, high=100000)  # randomly seed our rng
    np.random.seed(seed_int)  # seed for reproducibility
    df = df.reindex(np.random.permutation(df.index))  # 'shuffle' the dataset

    #  we have binary data (0 = negative, 1 = positive)
    #  split it with 10 folds for us to use in cross-validation
    skf = StratifiedKFold(n_splits=10)  # 10 is a very common/standard split ratio

    X = df['review'].values.astype('U')  # reviews
    y = df['sentiment'].values  # sentiment

    skf.get_n_splits(X, y)  # make n_splits for cross validation

    #  After a lot of GridSearching these seem to be the best
    #  further explanation in write-up and on GitHub README (public after due data)
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            analyzer='word',
                            stop_words=None,
                            lowercase=False,
                            preprocessor=preprocessor,
                            tokenizer=tokenizer,
                            max_df=.05,
                            min_df=2,
                            max_features=80000,
                            sublinear_tf=True,
                            ngram_range=(1, 2))

    #  This is an example of our parameter grid that we would use for searching
    #  It was found that sometimes having a really large parameter grid took too long
    #   to be useful (example 700+ fits would take more than an 3 hours running on 8 cores)
    param_grid_2 = [{'vect__max_df': [.75, .5, .1, .075]}]

    #  This is our pipeline to train the model with
    #  We use a TfidfVectorizer and a Stochastic Gradient Descent Classifier
    #  More information can be found in the write-up and on the GitHub README
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', SGD(loss='modified_huber', alpha=0.00015, n_iter=np.ceil(10 ** 6 / len(df['review'])),
                                     random_state=5, l1_ratio=0.05, penalty='l2', shuffle=False,
                                     learning_rate='optimal'))])

    #  Beginning of example for how we used GridSearchCV to find parameters
    #  This is only here to showcase how we found parameters - it would use parameter_grid2 from above
    # gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid_2,
    #                            scoring='accuracy',
    #                            cv=5,
    #                            verbose=1,
    #                            n_jobs=6)  # how many cores to run on - this gets all of them
    #
    #
    #
    # gs_lr_tfidf.fit(X_train, y_train)
    #
    # print('BEST PARAM: ')
    # print(gs_lr_tfidf.best_params_)
    #
    # print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    #
    # clf = gs_lr_tfidf.best_estimator_
    # print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    #
    # pickle.dump(clf, open('saved_model.sav', 'wb'))
    # # End of GridSearchCV example

    # Train n_fold different models on each kfold and see how they perform
    scores = cross_val_score(lr_tfidf, X, y, cv=skf, verbose=2, n_jobs=2)
    # this is our average score of the models that we trained on with cross-validation
    print(scores.mean())

    #  After we cross validate to ensure that the model can predict unseen data - train on everything
    lr_tfidf.fit(X, y)

    # Save the classifier for use later.
    pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
