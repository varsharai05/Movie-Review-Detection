import streamlit as st

from tensorflow.keras.models import load_model
import pypickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(layout='centered')

st.title("Movie Review Sentimental Analysis ")
word_dic = pypickle.load("word_dictionary.pkl")

def convert_txt_into_word_id(text):
  text = text.lower().split()
  print(text)
  word_ids = [1]
  for t in text:
    if t == '?':
      word_ids.append(2)
    else:
      word_id = word_dic.get(t , 2)
      if word_id != 2:
        word_ids.append(word_id + 3)
      else:
        word_ids.append(word_id)
      
  
  return word_ids



def load_rnn_model():
  model = load_model("movie_review_detection.keras")
  return model


review = st.text_area("Movie Review" , placeholder="Write Your Review")

isClicked = st.button("Predict" , type='primary')

if isClicked:
  word_ids = convert_txt_into_word_id(review)
  word_ids = pad_sequences([word_ids] , padding='pre' , truncating='post' , maxlen=200)
  rnn = load_rnn_model()
  pred = rnn.predict(word_ids)
  with st.spinner("Processing..."):
    if pred >= 0.5:
      st.image("./positive.jpg" , width=200)
      st.success("Positive Review")
    else:
      st.image("./negative.jpg" , width=200)
      st.success("Negative Review")
    
   