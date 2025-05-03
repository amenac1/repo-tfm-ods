import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_abstract(text: str) -> str:
    """Clean undesired metadata from paper abstracts.

    This function removes copyright notices and trailing metadata
    (e.g. affiliations, summaries, and other extra information) from
    an abstract string.

    Args:
        text: The original abstract string.

    Returns:
        The cleaned abstract string.
    """

    cleaned_text = ''
    
    if isinstance(text, str):
        # Remove copyright data
        text = re.sub(r'^©\s*\d{4}[^.]*\.\s*', '', text)
        text = re.sub(r'\s*(?:\(C\)|©)\s*\d{4}.*$', '', text)

        # Remove any aditional data
        pattern = re.compile(
            r'^(.*?\.)\s*(?:\((?:general|summary|authors with filk|recognitions linked to)[^)]*\)(?:\s*;.*)?)\s*$',
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(text)
        if match:
            text = match.group(1)
        else:
            segments = [seg.strip() for seg in text.split(';')]
            candidates = [seg for seg in segments if len(seg.split()) >= 50]
            if candidates:
                candidate = max(candidates, key=lambda s: len(s.split()))
                text = re.sub(r'\s*\([^)]*\)\s*$', '', candidate)
        cleaned_text = text.strip()

    return cleaned_text

def soft_preprocessing(text:str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


DEFAULT_STOPWORDS = set(stopwords.words('english'))
df_custom = pd.read_excel('../data/previous_analisis/custom_stopwords.xlsx')
CUSTOM_STOPWORDS = set(df_custom['word'].tolist())
STOP_WORDS = DEFAULT_STOPWORDS.union(CUSTOM_STOPWORDS)

def preprocess_text(text: str, keep_stopwords=True) -> list[str]:
    """
    Preprocess a string by converting it to lowercase, removing non-letter characters,
    tokenizing, and lemmatizing the words. Optionally, removes stopwords.

    Args:
        text: The original string.
        keep_stopwords: If False, removes stopwords from the text.

    Returns:
        A list of preprocessed words.
    """
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    
    if not keep_stopwords:
        words_to_lemmatize = [word for word in words if word not in STOP_WORDS]
    else:
        words_to_lemmatize = words

    preprocessed_words = [lemmatizer.lemmatize(word) for word in words_to_lemmatize]
    
    return preprocessed_words