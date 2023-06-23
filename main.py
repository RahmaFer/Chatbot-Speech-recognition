import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import speech_recognition as sr

# Load the text file and preprocess the data
with open('C:/Users/ferja/Desktop/test.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
    # Tokenize the text into sentences
    sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if
             word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence


def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence


# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()
    # Reading Microphone as source
    with sr.Microphone() as source:
        st.info("Speak now...")
        # listen for speech and store in audio_text variable
        audio_text = r.listen(source)
        st.info("Transcribing...")

        try:
            # using Google Speech Recognition
            text = r.recognize_google(audio_text)
            return text
        except:
            return "Sorry, I did not get that."

# Create a Streamlit app
def main():
    st.title("Chatbot with speech recognition app")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    st.write("You can choose between speaking or writing your request: ")
    # Get the user's question
    question = st.text_input("You:")

    writing = st.button("Submit")

    st.write("Click on the microphone to start speaking:")
    speaking = st.button("Start Recording")

    # Create a button to submit the question
    if writing:
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)


    # add a button to trigger speech recognition
    elif speaking:
        text = transcribe_speech()
        response = chatbot(text)
        st.write("Chatbot: " + response)



def main1():
    st.write("Click on the microphone to start speaking:")

    # add a button to trigger speech recognition
    if st.button("Start Recording"):
        text = transcribe_speech()
        st.write("Transcription: ", text)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

