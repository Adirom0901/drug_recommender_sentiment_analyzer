# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 22:34:14 2025

@author: aditya namdeo
"""

import numpy as np
import pickle
import nltk
import tensorflow
nltk.download('wordnet')
import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
import requests
import io
import gdown
from nltk.data import find

@st.cache_data  # Cache the function to prevent rechecking
def ensure_wordnet():
    try:
        find("corpora/wordnet.zip")  # Check if wordnet is already downloaded
    except LookupError:
        nltk.download("wordnet") 
        
dataset_id = "1gt5awJUv00Avu6HKgJI_ERpn8i4rqTgi"
url_dataset = f"https://drive.google.com/uc?id={dataset_id}"
response = requests.get(url_dataset)
response.raise_for_status()  # Raise an error if the download fails

# Load pickle file from bytes
whole_dataset_pkl = pickle.load(io.BytesIO(response.content))  # Ensure it has a 'condition' column
whole_dataset = pd.DataFrame(whole_dataset_pkl)
condition_list = whole_dataset.condition.unique().tolist()
def recommend_drugs_for_condition(condition, df=whole_dataset, top_n=5):
    # Get only relevant drugs for the condition
    condition_df =df[df['condition']==condition]

    if condition_df.empty:
        return "Condition not found in dataset"

    # Compute drug-level effectiveness score (group by drug)
    drug_scores = condition_df.groupby('drugName')['Effectiveness_Score'].mean().reset_index()

    # Sort by effectiveness score
    recommended = drug_scores.sort_values(by="Effectiveness_Score", ascending=False).head(top_n)
    recommended = recommended['drugName'].tolist()

    return recommended
model_id = "1e_iRKYcU5SmF-MmzV67ByqkhHNhd_tCj"
url_model =  f"https://drive.google.com/uc?id={model_id}"
response_model = requests.get(url_model)
response_model.raise_for_status()  # Raise an error if the download fails
loaded_model = pickle.load(io.BytesIO(response_model.content))

word_id = "12wjuUzZupaQC9Ndbqazu3rIbqS82kcpn"
url_word= f"https://drive.google.com/uc?id={word_id}"
output_file = "glove.pkl"
@st.cache_data  # Cache the downloaded file
def download_glove():
    url_word = f"https://drive.google.com/uc?id={word_id}"
    gdown.download(url_word, output_file, quiet=False)
    return output_file  # Return file path

@st.cache_data  # Cache the loaded embeddings
def load_glove():
    with open(download_glove(), "rb") as f:
        glove_data = pickle.load(f)
    return glove_data

# Load GloVe embeddings
words = load_glove()


def sentiment_prediction(review):
    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = tokenizer.tokenize(review.lower())  # Tokenize and lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
    word_vectors = [words[t] for t in lemmatized_tokens if t in words]  # Map to vectors

            # If no valid tokens, use a zero vector
    if not word_vectors:
      word_vectors = [np.zeros(50, dtype=np.float32)]
    sequences =[]
    sequences.append(np.array(word_vectors, dtype=np.float32))  # Convert to array

        # Pad sequences
    num_samples = len(sequences)
    X_padded = np.zeros((num_samples, 90, 50), dtype=np.float32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), 90)  # Clip to max length
        X_padded[i, :length, :] = seq[:length]  # Copy truncated/padded sequence

    prediction = loaded_model.predict(X_padded)
    if prediction >0.5:
      return 'positive'
    else:
      return 'negative'
  
    
  
    
def main():
    
    st.title('Drug recommender')
    
    condition = st.selectbox('select the condition to be treated',condition_list)
    
    recommendations =[]
    
    if st.button('recommend'):
        recommendations = recommend_drugs_for_condition(condition)
        for i in recommendations:
            st.write(i)
    
    #giving a title
    st.title('multilingual review sentiment analyzer')
    # 
    review = st.text_input('Enter the review in any language')
    
    review = GoogleTranslator(source='auto', target='en').translate(review)
    st.write(review)
    
    
    
    #code for prediciton
    sentiment=''
    
    if st.button('review sentiment'):
        sentiment = sentiment_prediction(review)
        
    st.success(sentiment)
    
    
    
if __name__=='__main__':
    main()
    

