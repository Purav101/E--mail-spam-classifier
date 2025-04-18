import streamlit as st
import pickle
import re,string
import nltk 
from nltk.corpus import stopwords
nltk.download("stopwords")
cv=pickle.load(open('cv.pkl','rb'))
model=pickle.load(open('ss.pkl','rb'))
st.title("SMS SPAM CLASSIFIER BY NEXUS ML (PURAV SONI)")
input_sms =st.text_area("ENTER THE SMS HERE")

#preprocess
stop_word=stopwords.words("english")
def preprocess(text):
    text= text.lower()
    text=''.join([char for char in text if char  not in string.punctuation])
    text=' '.join([word for word in text.split() if word not in stop_word])
    return text
if st.button('Predict'):
#preprocess
    ppsms=preprocess(input_sms)
    vc=cv.transform([ppsms])
    result= model.predict(vc)[0]
    if result == 1:
      st.header("SPAM")
    else:
       st.header("NOT SPAM")