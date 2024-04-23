import os.path

import streamlit
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import torch
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import re


@streamlit.cache_resource
def summarizeBigBirdPegasus(text, max_words, min_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv", attention_type="original_full")
    model.to(device)

    # Convert word length to token length
    max_tokens = tokenizer.decode(tokenizer.encode(text, max_length=1024, truncation=True)).count(' ') + max_words
    min_tokens = tokenizer.decode(tokenizer.encode(text, max_length=1024, truncation=True)).count(' ') + min_words

    command_text = f"Summarize the following string with minimum length {min_tokens} tokens and maximum length {max_tokens} tokens\n"
    print(command_text)
    text = command_text + text

    inputs = tokenizer(text, return_tensors='pt').to(device)
    prediction_ids = model.generate(**inputs, min_length=min_tokens, max_length=max_tokens, num_beams=4, early_stopping=True)
    prediction_text = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)[0]
    prediction_text = prediction_text.replace('<n>', '\n')
    prediction_text = re.sub(r'\[\s*\[\s*section\s*\]\s*\]', '', prediction_text)
    prediction_text = re.sub(r'@\w+', '', prediction_text)

    return prediction_text


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

