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



whole_dataset_pkl = pickle.load(open("C:/Users/aditya namdeo/Downloads/drug_recomm_Sent_web_code/drug_Datase_all.pkl","rb"))  # Ensure it has a 'condition' column
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

loaded_model = pickle.load(open("C:/Users/aditya namdeo/Downloads/drug_recomm_Sent_web_code/final_drug_sentiment_model.sav","rb"))

words =pickle.load(open("C:/Users/aditya namdeo/Downloads/drug_recomm_Sent_web_code/glove_50d_embedding.pkl","rb"))

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
    

