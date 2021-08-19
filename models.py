import spacy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# NER Product Recognizer
nlp = spacy.load('./spacy_prod')

# Transformer QA Model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")