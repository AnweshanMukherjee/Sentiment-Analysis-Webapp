import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow import keras
import nltk
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

model1 = tf.keras.models.load_model('model1.h5')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
nltk.download('wordnet')
lem = WordNetLemmatizer()
with open('tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)
review_len = 100

def predict_sentiment(test_text):
	# Removing contractions
	test_text_without_contractions = contractions.fix(test_text)

	# Removing commas, fullstops, @ and apostrophe
	test_text_filtered = ""
	txt = test_text_without_contractions
	for i in range(len(test_text_without_contractions)):
		if txt[i] != ',' and txt[i] != '.' and txt[i] != '@':
			test_text_filtered += txt[i]
		else:
			if txt[i] == ',' and i!=0 and txt[i-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				test_text_filtered += ""
			elif txt[i] == '.' and i!=0 and txt[i-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				test_text_filtered += "."
			elif txt[i] == '.' and i!=len(txt)-1 and txt[i-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
				test_text_filtered += "."
			else:
				test_text_filtered += " "

	temp = ""
	i = 0
	txt = test_text_filtered
	while i < len(txt):
		if txt[i] == '\'':
			if i != len(txt) - 1 and txt[i+1] == 's':
				i += 1
		elif txt[i] != '\"':
			temp += txt[i]
		i += 1
	test_text_filtered = temp

	# Tokenization
	test_text_tokenized = word_tokenize(test_text_filtered)

	# Filtering
	filtered_sentence = []
	for w in test_text_tokenized:
		if w not in stop_words:
			filtered_sentence.append(w)

	# Stemming
	stemmed_words = []
	for w in filtered_sentence:
		stemmed_words.append(ps.stem(w))

	# Lemmatization
	lemmatized_words = []
	for w in stemmed_words:
		lemmatized_words.append(lem.lemmatize(w))

	# Converting text to sequence
	temp1 = tokenizer.texts_to_sequences(lemmatized_words)
	temp2 = []
	for i in range(len(temp1)):
		if len(temp1[i]) != 0:
			temp2.append(temp1[i][0])
	temp2 = [temp2]
	sequence = pad_sequences(temp2, maxlen=review_len)

  	# Getting prediction
	prediction = model1.predict(sequence)
	st.write("## Probabilities and Prediction: ")
	st.write("## P(Negative Sentiment) = ", prediction[0][0])
	st.write("## P(Neutral Sentiment) = ", prediction[0][1])
	st.write("## P(Positive Sentiment) = ", prediction[0][2])
	prediction = prediction.argmax(axis=-1)

	if prediction == 0:
		st.write("## Predicted Sentiment: Negative")
	elif prediction == 1:
		st.write("## Predicted Sentiment: Neutral")
	else:
		st.write("## Predicted Sentiment: Positive")


st.write("# Sentiment Analysis Project")

test_text = st.text_input("Enter sentence to predict the sentiment: ")
if test_text is "":
	st.text("Waiting for an input!")
else:
	predict_sentiment(test_text)

components.html("""
	<hr color="lightgreen">
	<b>By</b>
	<ul>
		<li>Anweshan Mukherjee (519)</li>
		<li>Rajarshi Saha (539)</li>
		<li>Ashwin Gupta (598)</li>
	</ul>

	<b>Project Guide</b>: Prof. Debabrata Datta
	"""
	)