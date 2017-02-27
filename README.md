## Sentiment Analysis on Movie Reviews
####Authors: Steven Bouwkamp, Andy Vuong, and Matt Schuch

The goal of this project is to train a model with given movie reviews with known values of either being
a positive or negative review.

The data provided hasn't been cleaned (HTML tags, foreign characters, fluff words) are present and must be
removed to achieve a more accurate model.

## Usage
Note: this repo already has the cleaned data in it, but also contains the raw data, which
is no longer used.

```
python basic_sentiment_pipeline.py
```

It will take around 10 minutes to run or so - varies per machine.

When it ends there should be an average accuracy printed from the cross-validation
and a pickled object will be saved that contains the trained model - file name of the
trained model will be 'saved_model.sav'

There is currently no longer any prediction code - however at some point it may be re-
implemented if desired. 

# Explanation of the Project
## Preprocessing of the Data
The model that you train can only be as good as the data you feed it and the algorithm
that you fit it with. Preprocessing data is an integral step throughout all of machine
learning - better preprocessors can give massive boosts in prediction accuracy.

Our data was a collection of IMDB movie reviews loaded with HTML tags, unknown/foreign
characters, excessive punctuation, meaningless words ('the', 'a', etc.), and similar words
like 'runner' and 'running' that were counted as unique.

Each of these had to be dealt with, so we put our raw data through several cleanings.

Let's start out with a sample movie review _potential spoilers_ :)


```
"No. Just NO. That's all that needs to be said.<br /><br />Summary: A random guy 
is in a cornfield. For some reason, I'm not sure, but it's his duty to run around 
inside. The next great thriller?<br /><br />A five year old could make a better 
 movie just filming an anthill, or even just grass growing. 
 Seriously.....<br /><br />You can't say it has bad acting, because there is 
 NO acting. You can't say it has bad writing, because it has NO writing. 
 You can't say it has bad cinematography, because there is NO cinematography. 
 You can't say it's a bad movie, BECAUSE THERE IS NO MOVIE! If you don't believe me, 
 go watch it. Just don't say I never warned you....."
```

#####Remove HTML tags and unknown characters

```
No. Just NO. That's all that needs to be said. Summary: A random guy is in a cornfield.
For some reason, I'm not sure, but it's his duty to run around inside. 
The next great thriller? A five year old could make a better movie just filming an 
anthill, or even just grass growing. Seriously..... You can't say it has bad acting,
because there is NO acting. You can't say it has bad writing, because it has NO writing.
You can't say it has bad cinematography, because there is NO cinematography. 
You can't say it's a bad movie, BECAUSE THERE IS NO MOVIE! 
If you don't believe me, go watch it. Just don't say I never warned you.....
```

The next thing to do is to remove words like 'it', 'no', 'has', 'is' - etc.
The reason being is that these words don't give us much meaningful insight to whether the
review is either positive or negative.
Removing them helps our model to have to only considered features that are beneficial to
improving accuracy.

These are known as stop words and we used the Natural Language Toolkit to accomplish this.

####Remove all stop words and punctuation

```angular2html
needs said summary random guy cornfield reason sure duty run around inside next
great thriller five year old could make better movie filming anthill even grass
growing seriously say bad acting acting say bad writing writing say bad
cinematography cinematography say bad movie movie believe go watch say never warned
```

This is a drastic difference now - almost all of the fluff words and punctuation are 
removed. The amount that is removed can always be tweaked - some don't like removing
stop words because of the lost meaning from the overall picture.

The last step is to combine similar words through a process called stemming.
This process takes words like 'running' and 'runner' and sees them as a single
word 'run'.

To accomplish this we utilized the Natural Language Toolkit's PorterStemmer package.

Note: this gives us a list of every single word in the review as an element for
readability this was converted to a string.
####Stemmed Review
```
need said summari random guy cornfield reason sure duti run around insid next
great thriller five year old could make better movi film anthil even grass
grow serious say bad act act say bad write write say bad
cinematographi cinematographi say bad movi movi believ go watch say never warn
```

One thing you'll notice from this is that some of the words are quite different.
For example: 'summary' became 'summari' and 'cinematography' became 'cinematographi'

Pluralities and suffixes that don't really change the underlying meaning of the word
are removed and everything is generalized. This allows us to now go through this 
cleaned review and create a count of each word.


### Tweaking the Preprocessing step

As with many things in life there exists a good balance between everything. For this 
project you can run into the issue of preprocessing out too much information - losing 
some of the underlying meaning that sits in the data. 

All of this preprocessing got us a much higher accuracy while considering each word
independently; however, the notion of n-grams comes into play here.
Take for instance the following:
```angular2html
This movie was not funny, nor good
```

If we were to pass this into our trained model it would predict it as being a positive
review, even though it is a negative review. This is because our original model treated
each word as a singular feature and applies a certain weight to each feature as it trains.

In this case 'not' and 'nor' could most likely have a small weight that points to 
being negative, but they (especially 'not') are very common words and would be in
a majority of the reviews - both positive and negative.
The words 'funny' and 'good' were probably in more positive reviews than negative, thus
a stronger weight was applied to them causing the entire review to shift toward being
classified as being positive.

This is actually quite common and the solution is extremely simple.
We can treat each word as a single feature, but also add on to it all of the 
possible bigrams that exist in the review as well. 

The above example would become the following when taking all n-grams of size 1 to 2:

Note: each 'feature' is separated by a comma
```angular2html
This, this movie, movie, movie was, was, was not, not, not funny, funny, funny nor,
nor, nor good, good
```

This increases how many features that we have to train on and also highlights more 
important features like the phrase 'not funny' and 'nor good' - now the weight that
is given to funny for being a positive term is overshadowed by the negative weight
that is given to the phrase 'not funny'

By including bigrams we are able to more accurately predict whether the review
is positive or negative - however we had to put the stop words back in and could no
longer stem the data.


## Conclusion of Preprocessing
As you've probably noticed, the section on preprocessing was massive.
This is typical with many implementations of machine learning - you have to get the
best representation of your data before you even begin training the model.

We started out with removing words like 'it' and 'the' and got a pretty decent
increase in our prediction accuracy - but when adding in bigrams we found it beneficial
to keep all of the words in the original review to retain more meaning.

Now we'll venture into the different types of algorithms that we tried and our
results with each.

## Regularization with TfidfVectorizer

One key way to mitigate overfitting of data when training models with machine learning
is to both convert your data into pure numbers and to regularize it (basically have a
small range).

Scikit-Learn's TfidfVectorizer does both of these for us. 
Tf-idf stands for **term frequency - inverse document frequency** and is used for 
reflecting how important a word is. It keeps track of how many times a word appears
in the review and offset that by how common that word is throughout all reviews.

To do this, the TfidfVectorizer creates a count of every single unique word in a
review then how common the word is throughout all other reviews and take
the inverse of that - both terms would then be multiplied together
 (term frequency X inverse of the document frequency).
 
This would give us a large range of varying values - uncommon words would have a score
that sits near 0 while more common scores would have a score much greater.
If we were to put these values into our model for training we would most likely
overfit our model - since the values aren't generalized it learns a lot of the
*background noise* - it might work really well for the training data, but probably
won't have good prediction capabilities on unseen data.

L2 normalization is one of the many methods that can solve this and the TfidfVectorizer
does this automatically. An easy way to understand this is applying Occam's razor to the 
problem - if there is two explanations, the simpler one is usually better. 

How does this help us? 

and 

How does L2 normalization work?

Both of these can be answered quite simply. We want our model to be general enough
that it can consistently predict unseen data accurately. One of the causes of overfitting
your model is the model itself becomes too complex. The weights that are applied to each
 feature (in this case words and bigrams like 'not' and 'not funny') determines the 
model's complexity.

What regularization does is pull everything in - those large weights that may cause 
the model to overfit are reigned in. Since everything becomes more generalized the model 
should have better predictive capabilities.

### Tuning the Parameters of the TfidfVectorizer

Here are all of the parameters that we changed for our TfidfVectorizer and what 
impact they had on our model.

**strip_accents='unicode'**

Many of the reviews had foreign characters in them - while we removed most, if not all,
during our initial cleaning of the data, this was here just in case.

**analyzer='word'**

This tells our model what the features are that it should be training on - for us
we want it to look at words.

**stop_words=None**

Stop words are words like 'the', 'it', 'a' etc. At first we wanted this - it made sense at
the time- however, as we made the shift of including both single words and bigrams as
a feature, removing stop words actually changed the sentiment/meaning of the review.

**lowercase=False**

We transformed all reviews to lower case in our preprocessor - no need to do it twice.

**preprocessor=preprocessor**

Calls our preprocessor function that removes all punctuation and digits from the review.

**tokenizer=tokenizer**

Calls our tokenizer function that splits the preprocessed review into each distinct word.

**max_df=.05**

Means we would not consider/look at features or words that were seen in more than 5%
of the documents - this allows us to look at less common features that most likely
carry much more weight toward determining whether a review is positive or negative.

**min_df=2**

Means we would not consider/look at features that were only in a single review - this 
helps us eliminate people who talk about specific actors/actresses and directors.

**max_features=80000**

Set an upper limit for how many features we want to look at - 80K was determined to be
a pretty good parameter through extensive searching.

**sublinear_tf=True**

Speed up the process just a little

**ngram_range=(1, 2)**

This was found to be one of the most important parameters - this means that we are looking
at single words and bigrams. An example of a bigram 'not funny', this allows the meaning
of the negation of funny to be kept - without taking bigrams into account it would ONLY
look at 'not' and 'funny' independently. Now it looks at all three.

## Classification Algorithms Tested w/ Results

Throughout this project we tested many different classification algorithms for training
a linear model to predict whether a given movie review was either positive or negative.

This is a binary classification problem (either positive or negative) and we found that 
many of the linear algorithms proved to be quite accurate - with some more accurate than others. 

#### Logistic Regression
We started with some basic code that implemented a logistic regression model to try and fit
the data.

[Here's a good explanation of the algorithm](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

Logistic regression seemed relatively well - and with some parameter tuning we were able to
get around an average of 89% accuracy.


#### Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) is one of the more commonly used classifiers throughout
machine learning. 

[Here's a good explanation of it](http://sebastianruder.com/optimizing-gradient-descent/index.html#stochasticgradientdescent)

One of the nicer aspects of SGD is the ability to easily change its loss function. One of
the problems that was arising within plain Logistic Regression was the effect of outliers
on our results. SGD allows for a huber loss as its loss function which basically
lowers the impact that outliers had on the result.

Most of the settings that were used for SGD were the default values.