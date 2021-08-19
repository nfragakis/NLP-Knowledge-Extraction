class ARGUS(mlflow.pyfunc.PythonModel):
  """
  ARGUS pipeline wrapped in ml flow function for use via
    mlflow platform
  Steps
    1) Product Recognition
    2) Tokenize Text
    3) Information extraction on step 1 output
  Load
    model = mlflow.pyfunc.load_model('models:/ARGUS IE/1')
  """
  
  def __init__(self, MODEL_PATH = '/dbfs/FileStore/ARGUS'):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from nltk.tokenize import word_tokenize, sent_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    
    import mlflow
    
    import spacy
    assert spacy.__version__=='2.1.0'
    
    self.nlp = mlflow.spacy.load_model('models:/ARGUS NER/1')
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH + '/tokenizer')
    self.qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH + '/qa_model')
    
    
  def predict(self, context, model_input):
    return self.process_request(model_input)
    
    
  def find_issue(self, txt, ent, question, verbose=True):
    query = question + ent + '?'

    inputs = self.tokenizer(query, txt, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = self.qa_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    confidence_raw = answer_start_scores[0][answer_start] + answer_end_scores[0][answer_end]
    confidence_score = 1 / (1 + np.exp(-confidence_raw.detach().numpy()))

    if verbose:
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    return str(answer), confidence_score
  
  def flatten_list(self, docs):
    return [x for sublist in docs for x in sublist]
  
  
  def id_ents(self, txt):
    return [ent.text for ent in self.nlp(txt).ents if (ent.label_ == 'PRODUCT')]


  def clean_request(self, txt):
      txt = re.sub('[^A-Za-z0-9]+', ' ', txt)
      txt = re.sub(r'\b[0-9]+\b\s*', '', txt)
      return txt
  
  
  def sentence_splitter(self, txt):
      sents = sent_tokenize(txt)
      sents = self.flatten_list([sent.split('\n') for sent in sents])
      return sents
  
  
  def split_request(self, txt):
    """
    split sentences into chunks < 512
    characters to feed to qa relation
    extraction model.
    """
    sent_staging = []
    paragraph = []
    for sent in self.sentence_splitter(txt):
        sent = self.clean_request(sent)
        staged_len = sum([len(chunk) for chunk in sent_staging])
        if len(sent) + staged_len < 512:
            sent_staging.append(sent)
        else:
            paragraph.append('. '.join(sent_staging))
            if len(sent) < 512:
                sent_staging = [sent]

    paragraph.append('. '.join(sent_staging))
    return paragraph
  
  
  def resolve_relations(self, txt, THRESHOLD=0.9):
    """
    loop through all relation/question pairs
    in the query dict to extract key relation
    values from request.
    Return all pairs w/ score greater than
    THRESHOLD value.
    """
    query_dict = {
        'CONTEXT': "What is the context of the ",
        'ISSUE': "What is the issue with the ",
        'LOCATION': "What is the location of the "
    }

    triplets = []
    ents = set(self.id_ents(txt))

    # need to move upstream as this gets 'PRODUCT' triples
    #  for every passage in large texts... to much noise
    # if len(ents) == 0: ents = ['PRODUCT']

    for ent in ents:
        for relation, question in query_dict.items():
            ans, score = self.find_issue(txt, ent, question, verbose=False)
            if score > THRESHOLD:
                triplets.append((ent, relation, ans, score))
    return triplets
  
  
  def process_triplets(self, triplets):
    """
    loop through triplets append non-empty results
    choose highest confidence product, response
    pair for all responses
    """
    final_triplets = []
    answers = defaultdict(list)

    for trip in triplets:
        if trip[2] != '': answers[trip[2]].append(trip)

    # remove duplicate answers
    for ans in answers.keys():
        final_triplets.append(sorted(
            answers[ans], key=lambda x: x[2],
            reverse=True)[0:3][0]
                              )
    return final_triplets
  
  
  def process_request(self, txt):
    try:
      chunks = self.split_request(txt)
      if len(chunks) == 1:
          triplets = (self.resolve_relations(txt))
      else:
          triplets = self.flatten_list([self.resolve_relations(chunk) for chunk in chunks if chunk != ''])
      triplets = self.process_triplets(triplets)
      
      return triplets
    except Exception as e:
      print(e)
      pass