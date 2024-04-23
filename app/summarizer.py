import os.path

import streamlit
import torch
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@streamlit.cache_resource
def summarizeBartLarge(text, max_words, min_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model_dir = "./models/bart_large_cnn"
    model_name = "facebook/bart-large-cnn"
    if os.path.exists(model_dir):
        model = BartForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = BartTokenizer.from_pretrained(model_dir)
    else:
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        model.save_pretrained("./models/bart_large_cnn")
        tokenizer.save_pretrained("./models/bart_large_cnn")

    model.to(device)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_words, min_length=min_words, do_sample=False)
    return summary[0]["summary_text"]


@streamlit.cache_resource
def summarizeFalconsai_T5Small(text, max_words, min_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    model_dir = "./models/text_summarization"
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
        model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    model.to(device)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = summarizer(text, max_length=max_words, min_length=min_words, do_sample=False)
    return summary[0]["summary_text"]
