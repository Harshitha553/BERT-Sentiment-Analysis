import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖"
)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

st.title("🤖 BERT Sentiment Analysis")

text = st.text_area(
    "Enter Review",
    height=150
)

if st.button("Analyze"):

    if text.strip():

        result = classifier(text)

        label = result[0]["label"]
        score = result[0]["score"]

        if label == "POSITIVE":
            st.success(
                f"😊 Positive\nConfidence: {score:.2%}"
            )
        else:
            st.error(
                f"😞 Negative\nConfidence: {score:.2%}"
            )
