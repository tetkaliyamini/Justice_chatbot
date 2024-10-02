import random
import json
import torch
import streamlit as st
import easyocr
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit UI
st.set_page_config(page_title="AI Chatbot with OCR", layout="centered")

# Set white background and center title
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    .title {
        text-align: center;
    }
    .chat {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        text-align: left;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-top: 5px;
        max-width: 60%;
    }
    .bot-bubble {
        text-align: right;
        background-color: #d9fdd3;
        padding: 10px;
        border-radius: 10px;
        margin-top: 5px;
        max-width: 60%;
    }
    .stTextInput input {
        padding: 10px;
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title at center
st.markdown('<h1 class="title">Interactive AI Chatbot</h1>', unsafe_allow_html=True)
st.write("Ask any question or upload a document for analysis.")

# Chatbot response logic
def chatbot_response(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

# Store user input and bot responses
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# User input at the bottom (constant input area)
def submit():
    if st.session_state.input_text:
        user_message = st.session_state.input_text
        bot_response = chatbot_response(user_message)
        st.session_state.conversation.append(("user", user_message))
        st.session_state.conversation.append(("bot", bot_response))
        st.session_state.input_text = ""

# Chat display in conversation flow
for role, message in st.session_state.conversation:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{message}</div>', unsafe_allow_html=True)

# Input area for new messages (no className argument)
st.text_input("Type your message:", key="input_text", on_change=submit, placeholder="Type a message...")

# OCR document upload and processing
uploaded_file = st.file_uploader("Upload a document for OCR", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
    
    # If PDF, convert to image
    if uploaded_file.name.endswith(".pdf"):
        import fitz  # PyMuPDF
        pdf_document = fitz.open(uploaded_file)
        page = pdf_document.load_page(0)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        image = Image.open(uploaded_file)
    
    # Perform OCR
    results = reader.readtext(np.array(image))
    
    # Display OCR results
    ocr_text = " ".join([res[1] for res in results])
    st.write(f"**OCR Text**: {ocr_text}")
    
    # Use the OCR result as input to the chatbot
    response_from_ocr = chatbot_response(ocr_text)
    st.write(f"**Sam** (OCR-based response): {response_from_ocr}")
