import streamlit as st
import pickle as pkl
import string
import nltk
nltk.download("punkt_tab")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  x = []
  for i in text:
    if i.isalnum():
      x.append(i)
  text = x[:]
  x.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      x.append(i)
  text = x[:]
  x.clear()

  for i in text:
    x.append(ps.stem(i))
  text = x[:]
  x.clear()

  return " ".join(text)

model = pkl.load(open("model.pkl", "rb"))
tfidf = pkl.load(open("vectorizer.pkl", "rb"))

st.title("Email/SMS Spam Classifier")
input_msg = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_msg = transform_text(input_msg)
    vector_input = tfidf.transform([transformed_msg])
    result = model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not spam")