import pandas as pd
import re
from nltk import word_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
import string
import tashaphyne.arabic_const as arabconst

def normalizeArabic(text):
    # Remove Tashkeel
    text = arabconst.HARAKAT_PAT.sub('', text)
    # Remove Repeated Characters
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text
    
stop_words = [normalizeArabic(x) for x in stopwords.words('arabic')]
st = ISRIStemmer()

def remove_single_chars(text):
    words = text.split(" ")
    text = " ".join([word for word in words if len(word) > 1])
    return text
    
def clean(text):
    text = normalizeArabic(text)
    # Remove Punctuations
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    text = text.translate(str.maketrans('', '', punctuations_list))
    # Remove Hashtag Signs
    text = re.sub(r"#", " ", text)
    # Remove URLs, Mentions, Trailing Non-Whitespaces Characters
    text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
    # Remove Numbers
    text = re.sub(r"\d+", " ", text)
    # Remove English Characters
    text = re.sub(r"[A-Z|a-z]+", " ", text)
    # Remove Single Characters
    text = remove_single_chars(text)
    # Remove Stop Words
    text = " ".join([word for word in word_tokenize(text) if not word in stop_words])
    return str(text)
    
def stem(text):
    # Stemming
    text = " ".join([st.stem(word)for word in word_tokenize(text)])
    return text