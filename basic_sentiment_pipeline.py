"""
    Train a logistic regression model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""

import pandas as pd
import pickle
import numpy as np
import re

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.linear_model import SGDClassifier as SGD

# Hint: These are not actually used in the current 
# pipeline, but would be used in an alternative 
# tokenizer such as PorterStemming.
# import nltk
# nltk.download('stopwords')

# stop = stopwords.words('english')

"""
    This is a very basic tokenization strategy.  
    
    Hint: Perhaps implement others such as PorterStemming
    Hint: Is this even used?  Where would you place it?

    Yeah PorterStemming decreases accuracy by .2% for some stupid reason.
"""

porter_stem = PorterStemmer()

stop_words = set(stopwords.words('english')).union({'.', 'I', 'i', ',', '\'', 'it', '*', '?', '/', '-', '&', '<', '>', '\"'})


def tokenizer(text):
    return text.split()
    # return [porter_stem.stem(word) for word in text.split()]

# stop words reduces accuracy
def preprocessor(text):
    new_text = re.sub("[^a-zA-Z]", " ", text.lower())
    # word_punct = wordpunct_tokenize(new_text)
    # new_punct = [word for word in word_punct if word not in stop_words]
    # output = ' '.join(new_punct)
    return new_text
    # return output

# this is to protect for running multiple jobs for gridsearchcv
if __name__ == '__main__':
    # Read in the dataset and store in a pandas dataframe
    df = pd.read_csv('./training_movie_data_cleaned.csv')
    np.random.seed(0)  # seed for reproducibility
    df = df.reindex(np.random.permutation(df.index))

    # Split your data into training and test sets.
    # Allows you to train the model, and then perform
    # validation to get a sense of performance.
    #
    # Hint: This might be an area to change the size
    # of your training and test sets for improved
    # predictive performance.
    training_size = 40000

    X_train = df.loc[:training_size, 'review'].values.astype('U')
    y_train = df.loc[:training_size, 'sentiment'].values
    X_test = df.loc[training_size:, 'review'].values.astype('U')
    y_test = df.loc[training_size:, 'sentiment'].values

    # Perform feature extraction on the text.
    # Hint: Perhaps there are different preprocessors to
    # test?

    tfidf = TfidfVectorizer(strip_accents='unicode',
                            analyzer='word',
                            stop_words=None,
                            lowercase=False,
                            preprocessor=preprocessor,
                            tokenizer=tokenizer,
                            max_df=.9,

                            max_features=100000,
                            sublinear_tf=True,
                            ngram_range=(1, 2))

    # exit(0)

    # Hint: There are methods to perform parameter sweeps to find the
    # best combination of parameters.  Look towards GridSearchCV in
    # sklearn or other model selection strategies.

    # Create a pipeline to vectorize the data and then perform regression.
    # Hint: Are there other options to add to this process?
    # Look to documentation on Regression or similar methods for hints.
    # Possibly investigate alternative classifiers for text/sentiment.

    # TODO to run the grid search cv - uncomment the following
    #
    # param_grid = [{'vect__ngram_range': [(1, 1)],
    #                'vect__stop_words': ['english', None],
    #                'vect__tokenizer': [tokenizer, None],
    #                'clf__penalty': ['l1', 'l2'],
    #                'clf__C': [1.0, 10.0, 100.0]},
    #               {'vect__ngram_range': [(1, 1)],
    #                'vect__stop_words': ['english', None],
    #                'vect__tokenizer': [tokenizer, None],
    #                'vect__use_idf':[False, True],
    #                'vect__norm':[None],
    #                'clf__penalty': ['l1', 'l2'],
    #                'clf__C': [1.0, 10.0, 100.0]},
    #               ]
    # TODO - only choose one of the parameter grids at a time
    param_grid_2 = [{'clf__penalty': ['l2', 'l1', 'elasticnet'],
                     'clf__l1_ratio': [0.05, 0.15, 0.35, 0.5],
                     'clf__fit_intercept': [True, False],
                     'clf__shuffle': [True, False],
                     'clf__learning_rate': ['optimal']
                     }]

    #  http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
    #  Above link for info on the value C in the classifier (clf)

    # LIST OF BEST PARAMS FOR SGC:
    #   loss = 'modified_huber' SCORE = 0.906
    #   penalty = 'l2' SCORE = 0.906; this is the default value
    #   alpha = 0.00015 SCORE = 0.906; default is 0.0001 gets a 0.906 too
    #   n_iter = 10 SCORE = 0.907; default is 5 - this is the number of epochs
    #   SGD max score is 0.907
    # TODO - don't comment this out - unless you want to change the classifier
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', SGD(loss='modified_huber', alpha=0.00015, n_iter=20, random_state=5, l1_ratio=0.05, penalty='l2', shuffle=False, learning_rate='optimal'))])

    # TODO - was using this one as a test classifier (ie changing the classifier)
    # lr_tfidf = Pipeline([('vect', tfidf),
    #                      ('clf', LassoCV(max_iter=20))])

    # # TODO - uncomment this to run the grid search with cross validation
    # gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid_2,
    #                            scoring='accuracy',
    #                            cv=5,
    #                            verbose=1,
    #                            n_jobs=6)  # how many cores to run on - this gets all of them
    #
    #
    # # TODO - ucnomment this to run the grid search
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
    # # TODO - end of uncommenting

    #  #  TODO - comment out the following to run gridsearchcv
    # # Train the pipline using the training set.
    lr_tfidf.fit(X_train, y_train)
    #
    # # Print the Test Accuracy
    print('Test Accuracy: %.3f' % lr_tfidf.score(X_test, y_test))
    #
    # # Save the classifier for use later.
    pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
    #  #  TODO - end of comment out the following for non gridsearch



