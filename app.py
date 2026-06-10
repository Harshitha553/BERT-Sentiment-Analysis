import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

import torch
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():

    model = BertForSequenceClassification.from_pretrained(
        "bert_sentiment_model"
    )

    tokenizer = BertTokenizer.from_pretrained(
        "bert_sentiment_model"
    )

    return model, tokenizer

model, tokenizer = load_model()

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    return prediction

st.title("🤖 BERT Sentiment Analysis")

user_input = st.text_area(
    "Enter Review"
)

if st.button("Analyze Sentiment"):

    prediction = predict_sentiment(
        user_input
    )

    if prediction == 1:
        st.success("😊 Positive Sentiment")
    else:
        st.error("😞 Negative Sentiment")