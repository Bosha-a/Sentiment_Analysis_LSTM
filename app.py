import streamlit as st 
from helper import normalize_text
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import nltk 
import pickle 
nltk.download('stopwords')
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = stopwords.words('english')

with open('tokenizer.pickle' , 'rb') as file: 
    tokenizer = pickle.load(file)

model = load_model(r'model_with_embidding.keras')


def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post', truncating='post')
    return padded

title = st.title('Welcome to Your Sentiment Analysis Application')

text = st.text_input('Enter your text here:')
text = preprocess_text(text)

button = st.button('Predict')

dict = {1 : 'Positive', 0 : 'Negative'}

if button:
    prediction = model.predict(text)[0]
    predicted_class = int(prediction.round())
    sentiment = dict[predicted_class]
    st.write(sentiment)