import pre_processing
import pickle
import pandas
import re
import string


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
lexicon = pandas.read_csv('New_Lexicon.csv')
hurt_words = set(lexicon['clean'])

NB_CountVector_feature = pickle.load(open("NB_CountVector_feature", 'rb'))
NB_tfidf_word_feature = pickle.load(open("NB_tfidf_word_feature", 'rb'))
NB_tfidf_ngram_feature = pickle.load(open("NB_tfidf_ngram_feature", 'rb'))
NB_tfidf_char_feature = pickle.load(open("NB_tfidf_char_feature", 'rb'))

LR_CountVector_feature = pickle.load(open("LR_CountVector_feature", 'rb'))
LR_tfidf_word_feature = pickle.load(open("LR_tfidf_word_feature", 'rb'))
LR_tfidf_ngram_feature = pickle.load(open("LR_tfidf_ngram_feature", 'rb'))
LR_tfidf_char_feature = pickle.load(open("LR_tfidf_char_feature", 'rb'))

SVM_CountVector_feature = pickle.load(open("SVM_CountVector_feature", 'rb'))
SVM_tfidf_word_feature = pickle.load(open("SVM_tfidf_word_feature", 'rb'))
SVM_tfidf_ngram_feature = pickle.load(open("SVM_tfidf_ngram_feature", 'rb'))
SVM_tfidf_char_feature = pickle.load(open("SVM_tfidf_char_feature", 'rb'))



count_vect = pickle.load(open("count_vect", 'rb'))
tfidf_vect = pickle.load(open("tfidf_vect", 'rb'))
tfidf_vect_ngram = pickle.load(open("tfidf_vect_ngram", 'rb'))
tfidf_vect_ngram_chars = pickle.load(open("tfidf_vect_ngram_chars", 'rb'))


def prepare_live(live_test):
    cleaned = pre_processing.clean(live_test)
    # stemmed = pre_processing.stem(cleaned)
    live_df = pandas.DataFrame({'clean': [cleaned]})
    # transform the training and test data using count vectorizer object
#     live_df['CV_features'] =  [count_vect.transform(live_df['clean'].values.astype('U'))]
    CV_features =  count_vect.transform(live_df['clean'].values.astype('U'))
    # print(CV_features)
    # word level tf-idf
#     live_df['tfidf_word_features'] =  [tfidf_vect.transform(live_df['clean'].values.astype('U'))]
    tfidf_word_features =  tfidf_vect.transform(live_df['clean'].values.astype('U'))
    # print(tfidf_word_features)
    # ngram level tf-idf 
#     live_df['tfidf_ngram_features'] =  [tfidf_vect_ngram.transform(live_df['clean'].values.astype('U'))]
    tfidf_ngram_features =  tfidf_vect_ngram.transform(live_df['clean'].values.astype('U'))
    # print(tfidf_ngram_features)
    # characters level tf-idf
#     live_df['tfidf_char_features'] =  [tfidf_vect_ngram_chars.transform(live_df['clean'].values.astype('U'))]
    tfidf_char_features =  tfidf_vect_ngram_chars.transform(live_df['clean'].values.astype('U'))


    live_df['char_count'] = len(live_test)
    live_df['word_count'] = len(live_test.split())
    live_df['word_density'] = live_df['char_count'] / (live_df['word_count'])
    live_df['punctuation_count'] = len("".join(_ for _ in live_test if _ in punctuations_list))
    live_df['hashtag'] = len(re.findall(r"#(\w+)", live_test))
    live_df['hate_words_count'] = live_df['clean'].apply(lambda x: len([wrd for wrd in hurt_words if wrd in str(x)]))
    live_df['hate_word'] = live_df['clean'].apply(lambda x: 1 if (len([wrd for wrd in hurt_words if wrd in str(x)])) > 0 else 0)

    live_df['NB_CountVector_feature'] = NB_CountVector_feature.predict(CV_features)
    live_df['NB_tfidf_word_feature'] = NB_tfidf_word_feature.predict(tfidf_word_features)
    live_df['NB_tfidf_ngram_feature'] = NB_tfidf_ngram_feature.predict(tfidf_ngram_features)
    live_df['NB_tfidf_char_feature'] = NB_tfidf_char_feature.predict(tfidf_char_features)

    live_df['LR_CountVector_feature'] = LR_CountVector_feature.predict(CV_features)
    live_df['LR_tfidf_word_feature'] = LR_tfidf_word_feature.predict(tfidf_word_features)
    live_df['LR_tfidf_ngram_feature'] = LR_tfidf_ngram_feature.predict(tfidf_ngram_features)
    live_df['LR_tfidf_char_feature'] = LR_tfidf_char_feature.predict(tfidf_char_features)

    live_df['SVM_CountVector_feature'] = SVM_CountVector_feature.predict(CV_features)
    live_df['SVM_tfidf_word_feature'] = SVM_tfidf_word_feature.predict(tfidf_word_features)
    live_df['SVM_tfidf_ngram_feature'] = SVM_tfidf_ngram_feature.predict(tfidf_ngram_features)
    live_df['SVM_tfidf_char_feature'] = SVM_tfidf_char_feature.predict(tfidf_char_features)
    
    combined_features = ['char_count', 'word_count', 'word_density', 'punctuation_count', 'hate_words_count', 'hate_word', 'hashtag', 
            'NB_CountVector_feature', 'NB_tfidf_word_feature', 'NB_tfidf_ngram_feature','NB_tfidf_char_feature',
            'LR_CountVector_feature','LR_tfidf_word_feature','LR_tfidf_ngram_feature','LR_tfidf_char_feature',
            'SVM_CountVector_feature','SVM_tfidf_word_feature','SVM_tfidf_ngram_feature','SVM_tfidf_char_feature']

    return live_df[combined_features]

    # print(tfidf_char_features)