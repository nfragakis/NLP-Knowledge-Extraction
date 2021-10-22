from collections import defaultdict
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import ast

stop_words = ['a','about','above','after','again','against','ago','ain','all','am','an','and','any','are','aren',"aren't",'as',
              'at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d',
              'did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'double','down','during','each','few','for',
              'from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here',
              'hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself',
              'just','know','know','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','need','needs','needn',"needn't",
              'no','nor','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s',
              'same','shan',"shan't",'she',"she's",'should',"should've",'so','some','such','sure','t','than','that',"that'll",'the',
              'their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under',
              'until','up','ve','very','was','wasn','we','were','weren','what','when','where','which','while','who','whom',
              'why','will','with','won','y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']

stop_words_custom = list(set(stop_words + add_stop_words))

flatten_list = lambda docs: [x for sublist in docs for x in sublist]


def isolate_issueContext(triplets):
    response = []
    response_dict = defaultdict(list)

    triplets = ast.literal_eval(triplets)

    # loop through triplets collecting entity, relation, and extraction
    for trip in triplets:
        if trip[1] in ['CONTEXT', 'ISSUE']:
            response.append(trip[:3])

    # collapse extractions on entity
    for entity, _, extract in response:
        response_dict[entity].append(extract)

    # empty responses and add joined entity collapsed data
    response = []
    for key, val in response_dict.items():
        response.append(': '.join(val))
    return response


def clean_sentence(string, stem=False):
    global stop_words_custom
    string = re.sub('[^A-Za-z0-9]+', ' ', string)
    sent = string.lower()
    words = word_tokenize(sent)
    words = [w for w in words if not w in stop_words_custom]
    words = [i for i in words if len(i) < 15]
    sent = ' '.join(word for word in words)
    return sent


def clean_issues(sentlist):
    resultsentlist = []

    for senttoken in sentlist:
        sent = clean_sentence(senttoken)
        if 'email' not in sent:
            resultsentlist.append(sent)
    result = list(set(resultsentlist))
    return result


def c_tf_idf(documents, m, ngram_range=(1, 2)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words_custom, min_df=0.001).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def clean_dups(topic):
    words = word_tokenize(topic)
    if (len(words) == 2) & \
            (len(set(words)) == 1):
        return list(set(words)).pop()
    else:
        return topic


word_count = lambda x: len(word_tokenize(x))


def clear_redundent_unigrams(topics):
    unigrams = [x for x in topics if word_count(x) == 1]
    bigrams = [x for x in topics if word_count(x) == 2]

    bigrams.reverse()
    for bi in bigrams:
        words = word_tokenize(bi)
        if words[1] + ' ' + words[0] in bigrams:
            topics.remove(bi)
            bigrams.remove(bi)

    for uni in unigrams:
        uni_rank = topics.index(uni)
        for bi in bigrams:
            bi_rank = topics.index(bi)

            if (uni in bi) & (uni in topics):
                topics.remove(uni)

    return topics
