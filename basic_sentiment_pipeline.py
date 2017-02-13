"""
    Train a logistic regresion model for document classification.

    Search this file for the keyword "Hint" for possible areas of
    improvement.  There are of course others.
"""

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

# Hint: These are not actually used in the current 
# pipeline, but would be used in an alternative 
# tokenizer such as PorterStemming.
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop = stopwords.words('english')

"""
    This is a very basic tokenization strategy.  
    
    Hint: Perhaps implement others such as PorterStemming
    Hint: Is this even used?  Where would you place it?
"""
def tokenizer(text):
    return text.split()

# Read in the dataset and store in a pandas dataframe
df = pd.read_csv('./training_movie_data_cleaned.csv')

# Split your data into training and test sets.
# Allows you to train the model, and then perform
# validation to get a sense of performance.
# 
# Hint: This might be an area to change the size
# of your training and test sets for improved 
# predictive performance.
training_size = 37500
X_train = df.loc[:training_size, 'review'].values
y_train = df.loc[:training_size, 'sentiment'].values
X_test = df.loc[training_size:, 'review'].values
y_test = df.loc[training_size:, 'sentiment'].values

# Perform feature extraction on the text.
# Hint: Perhaps there are different preprocessors to
# test?

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

# Hint: There are methods to perform parameter sweeps to find the
# best combination of parameters.  Look towards GridSearchCV in 
# sklearn or other model selection strategies.

# Create a pipeline to vectorize the data and then perform regression.
# Hint: Are there other options to add to this process?
# Look to documentation on Regression or similar methods for hints.
# Possibly investigate alternative classifiers for text/sentiment.

#  http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
#  Above link for info on the value C in the classifier (clf)
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(C=1.00,fit_intercept=False,penalty='l2',random_state=0))])


param_grid = []

# Train the pipline using the training set.
lr_tfidf.fit(X_train, y_train)

# Print the Test Accuracy
print('Test Accuracy: %.3f' % lr_tfidf.score(X_test, y_test))

# Save the classifier for use later.
pickle.dump(lr_tfidf, open("saved_model.sav", 'wb'))
