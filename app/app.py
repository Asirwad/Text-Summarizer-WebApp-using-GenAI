import streamlit as st
from summarizer import summarizeBartLarge, summarizeFalconsai_T5Small

st.title("Text Summarization using :rainbow[GenAI] :pencil:")


with st.sidebar:
    st.title("_Text_ _Summarization_")
    st.write("This app is created using the open-source Falconsai model and Bart by facebook")

    selected_model = st.selectbox("Select model",
                                  [None, 'Falconsai-t5Small', 'Bart-Large-CNN'],
                                  key='selected_model',
                                  help='Select appropriate model to compute')
    max_words = st.slider("Maximum Words in Summary:", 20, 200, 100)
    min_words = st.slider("Minimum Words in Summary:", 10, 50, 20)

text_input = st.text_area(":green[Enter Text to Summarize:]", height=350, disabled=False)

if st.button("Summarize", key='button', type='primary'):
    if not text_input:
        st.warning("Please enter some text to summarize.")
    else:
        if selected_model == 'Falconsai-t5Small':
            summary = summarizeFalconsai_T5Small(text_input, max_words, min_words)
            st.subheader(":green[Summary:]")
            st.write(summary)
        elif selected_model == 'Bart-Large-CNN':
            summary = summarizeBartLarge(text_input, max_words, min_words)
            st.subheader(":green[Summary:]")
            st.write(summary)
        else:
            st.warning("Please select any model!", icon='⚠️')

