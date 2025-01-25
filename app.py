import streamlit as st 
import pickle as pkl 
from helper import normalize_text
import tensorflow as tf 
from tensorflow.keras.models import load_model

model = load_model(r'model_with_embidding.keras')

title = st.title('Welcome to Your Sentiment Analysis Application')

text = st.text_input('Enter your text here:')
text = normalize_text(text)


button = st.button('Predict')

dict = {1 : 'Positive', 0 : 'Negative'}

if button:
    prediction = model.predict(text)[0]
    sentiment = dict[prediction]
    st.write(sentiment)
    