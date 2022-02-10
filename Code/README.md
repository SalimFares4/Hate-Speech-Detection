# Data Pre-Processing

This file contains normalization, cleaning functions we used to pre-process our data
you do not need to run this file we have already excuted it and appended the results 
to the data file in a seperated column called "clean". You can simply load the data
and display the "tweet" field to see the original text and "clean" field to see
the same text after cleaning. We just wanted to share the pre-processing steps we used in this project.
However if you want to use the same methods you should have these libraries already installed:
* pandas
* nltk
* string
* tashaphyne.

# Statistical Methods using the Lexicon
This file contains a simple classification and statistical analysis using the lexicon.
To run this file you should have these libraries already installed:
* pandas
* nltk
* matplotlib.

# Classification Models

This file contains the implementation to three different classification algorithms:
Naiev Bayes, Logistic Regression, SVM. Using 5 ways to represent the text as features:
Count Vectors, TF-IDF Word-Levl, N-Gram-Leve, Char-Level Vectors, NLP based features.
And finally combine all previous features and re-apply the classification algorithms.
To run this file you should have these libraries already installed:
* pandas
* sklearn
* re
* string

# Word Embedding
This file contains the implementation to Word Embedding model. We used a pre-trained model
and we trained our own model.
To run this file you should have these libraries already installed:
* numpy
* gensim
* pandas
* sklear
* keras 
* You should also download the pre-trained model AraVec V2.0 tweets_cbow_300 link for downloading [here](https://archive.org/download/aravec2.0/tweet_cbow_300.zip)
* [Link for the whole AraVec Project](https://github.com/bakrianoo/aravec)
