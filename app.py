import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
# Load the pre-trained model and vectorizer
model = pickle.load(open('./models/model_updated.pkl', 'rb'))
vectorizer = pickle.load(open('./models/vectorizer_updated.pkl', 'rb'))  # Save vectorizer too if you haven't already

# Initialize PorterStemmer
port_stem = PorterStemmer()

# Function for preprocessing (stemming)
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

# Function to preprocess user input
def preprocess_input(sentence):
    # Step 1: Apply preprocessing (stemming) on the input sentence
    processed_sentence = stemming(sentence)
    
    # Step 2: Vectorize the processed sentence using the same vectorizer
    vectorized_input = vectorizer.transform([processed_sentence])
    
    # Step 3: Predict the sentiment (0 for negative, 1 for positive)
    prediction = model.predict(vectorized_input)
    
    return "Positive" if prediction[0] == 1 else "Negative"

# Set up the Streamlit interface
st.title("Sentiment Analysis App")
st.write("Enter a sentence below and get the sentiment prediction (Positive/Negative).")

# User input
user_input = st.text_area("Enter your text:")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    if user_input:
        result = preprocess_input(user_input)
        
        # Display sentiment with custom styles
        if result == "Positive":
            st.success(f"The sentiment of the given text is: {result}")
        else:
            st.error(f"The sentiment of the given text is: {result}")
    else:
        st.error("Please enter some text to analyze.")
