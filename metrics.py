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

# might be a little bit off - but it is close enough to get meaningful data from it
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