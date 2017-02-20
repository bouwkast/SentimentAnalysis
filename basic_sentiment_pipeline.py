"""
    Train a stochastic gradient descent model for movie review classification.

"""

# TODO - finish commenting this file
# TODO - clean up the other helper files (need to turn in most likely)
# TODO - maybe run some final GridSearchCVs - don't have to if don't want to
# TODO - remove print statements of the cross validation scores
# TODO - maybe make a logger? might help for overall completeness
# TODO - don't remove gridsearchcv - keep it there as a reference for how we calculated parameters
# TODO - don't remove porterstemmer or wordpunct - good ideas just they got trumped by bigrams
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

#  porter stemming decreases accuracy by .2%
porter_stem = PorterStemmer()

#  these are the default stopwords along with some added punctuation - no longer used.
stop_words = set(stopwords.words('english')).union(
    {'.', 'I', 'i', ',', '\'', 'it', '*', '?', '/', '-', '&', '<', '>', '\"'})

# The commented out lines for tokenizer and preprocessor are the more 'advanced'
# preprocessing steps - but they lower accuracy, so the basic ones are in there instead.

def tokenizer(text):
    return text.split()
    # return [porter_stem.stem(word) for word in text.split()]

# stop words reduces accuracy
def preprocessor(text):
    new_text = re.sub("[^a-zA-Z]", " ", text.lower())
    # word_punct = wordpunct_tokenize(text.lower())
    # new_punct = [word for word in word_punct if word not in stop_words]
    # output = ' '.join(new_punct)
    return new_text
    # return output


# this is to protect for running multiple jobs for gridsearchcv
if __name__ == '__main__':
    # Read in the dataset and store in a pandas dataframe
    df = pd.read_csv('./training_movie_data_cleaned.csv')
    magic_number = np.random.randint(low=0, high=100000)  # randomly seed our rng
    np.random.seed(magic_number)  # seed for reproducibility
    df = df.reindex(np.random.permutation(df.index))  # 'shuffle' the dataset

    skf = StratifiedKFold(n_splits=10)  # 10 is a very common/standard split ratio

    X = df['review'].values.astype('U')  # reviews
    y = df['sentiment'].values  # sentiment

    skf.get_n_splits(X, y)  # make n_splits for cross validation

    #  After a lot of GridSearching these seem to be the best
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

    # TODO - only choose one of the parameter grids at a time
    param_grid_2 = [{'vect__max_df': [.75, .5, .1, .075]}]

    # TODO - don't comment this out - unless you want to change the classifier
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', SGD(loss='modified_huber', alpha=0.00015, n_iter=np.ceil(10 ** 6 / len(df['review'])),
                                     random_state=5, l1_ratio=0.05, penalty='l2', shuffle=False,
                                     learning_rate='optimal'))])

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

    # Train n_fold different models on each kfold and see how they perform
    # TODO WARNING! change n_jobs to how many cores to run on
    scores = cross_val_score(lr_tfidf, X, y, cv=skf, verbose=2, n_jobs=2)
    print(scores)
    #  After we cross validate to ensure that the model can predict unseen data - train on everything
    lr_tfidf.fit(X, y)

    # print('Test Accuracy: %.3f' % lr_tfidf.score(X_test, y_test))

    # total = 0
    # for train_index, test_index in skf.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     lr_tfidf.fit(X_train, y_train)
    #     print('Test Accuracy: %.3f' % lr_tfidf.score(X_test, y_test))
    #     total += lr_tfidf.score(X_test, y_test)
    # print(total / 10)
    # print('end')

    #
    # # Print the Test Accuracy


    # Save the classifier for use later.
    pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
    #  TODO - end of comment out the following for non gridsearch
