import streamlit as st
from summarizer import summarizeBigBirdPegasus, summarizeBartLarge

st.title("Text Summarization")

with st.sidebar:
    st.title("Text Summarization")
    st.write("This app is created using the open-source BigBird Pegasus model from Google")

    selected_model = st.selectbox("Select model",
                                  [None, 'Big-bird-Pegasus-Large', 'Bart-Large-CNN'],
                                  key='selected_model',
                                  help='Select appropriate model to compute')
    max_words = st.slider("Maximum Words in Summary:", 50, 200, 100)
    min_words = st.slider("Minimum Words in Summary:", 10, 50, 20)

text_input = st.text_area("Enter Text to Summarize:", height=200, disabled=False)


if st.button("Summarize", key='button'):
    if not text_input:
        st.error("Please enter some text to summarize.")
    else:
        if selected_model == 'Big-bird-Pegasus-Large':
            summary = summarizeBigBirdPegasus(text_input, max_words, min_words)
            st.subheader("Summary:")
            st.write(summary)
        elif selected_model == 'Bart-Large-CNN':
            summary = summarizeBartLarge(text_input, max_words, min_words)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please select any model!", icon='⚠️')

