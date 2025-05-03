import sys
import os
sys.path.append(os.path.abspath("../"))
from src.preprocessing.preprocessing_functions import preprocess_text
import numpy as np
import pandas as pd


def df_2_embeddings(df: pd.DataFrame, text_column:str, model, keep_stopwords=False, mean=True, padding_max_length:int=None):
    embeddings = []
    embedding_dim = model.vector_size

    for text in df[text_column]:
        # Tokenize and preprocess text
        preprocessed_words = preprocess_text(text, keep_stopwords=keep_stopwords)
        
        # Get word vectors
        word_vectors = [model[word] for word in preprocessed_words if word in model]
        
        if not word_vectors:  # If no words are found in Word2Vec, return a zero vector/matrix
            if mean:
                vector = np.zeros(embedding_dim)
            else:
                vector = np.zeros((padding_max_length, embedding_dim))  # Full padding if empty text
        else:
            if mean:
                vector = np.mean(word_vectors, axis=0)  # Compute mean embedding
            else:
                vector = np.array(word_vectors)

                # Apply padding to ensure all embeddings have the same shape
                if vector.shape[0] < padding_max_length:
                    pad_size = padding_max_length - vector.shape[0]
                    pad = np.zeros((pad_size, embedding_dim))
                    vector = np.vstack((vector, pad))
                else:
                    vector = vector[:padding_max_length]  # Truncate if too long

        embeddings.append(vector)

    # Convert to NumPy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array

