import streamlit as st 
from model_class import Pipeline 
import spacy_streamlit
import spacy
import os
from PIL import Image

nlp = spacy.load('en_core_web_trf')
model = Pipeline()

def main():
    st.title("Knowledge Extraction AI Demo")

    raw_text = st.text_area("Your Text", "Enter Text Here")
    docx = nlp(raw_text)
    spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe('ner').labels)

    triplets = model.process_request(raw_text)
    st.write(triplets)

if __name__ == '__main__':
    main()
