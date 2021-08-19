## Load the packages
import json
import pandas as pd
import logging
import multiprocessing
import os
import re
import numpy as np

import sys
import traceback
import zipfile
import subprocess
import time
import tarfile
import boto3
from io import StringIO
from collections import defaultdict

import pickle as pkl
import json
import ast

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
#import neuralcoref
assert spacy.__version__ == '2.1.0'
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import umap
import hdbscan

# import util functions from code files
from .predict import process_request
from .data import (isolate_issueContext, clean_issues, flatten_list,
                   c_tf_idf, extract_top_n_words_per_topic, extract_topic_sizes,
                   clear_redundent_unigrams, clean_dups
                   )

# These are the paths to where SageMaker mounts interesting things in your container.
print("hello")
prefix = '/opt/ml/'

input_path = prefix + 'input/data'
print("input_path : {}".format(input_path))
print("Files in Input path")
print(subprocess.call(["ls", input_path]))

output_path = os.path.join(prefix, 'output')
print("output_path : {}".format(output_path))

model_path = os.path.join(prefix, 'model')
print("model_path : {}".format(model_path))

param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
print("param_path : {}".format(param_path))

data_path = os.path.join(input_path, "ptccb10")
print("data_path : {}".format(data_path))
print("Files in Data path")
print(subprocess.call(["ls", data_path]))


def train_entry_point():
    print('Starting the training')
    try:
        ## Read in any hyperparameters that the user passed with the training job
        # with open(param_path, 'r') as tc:
        #    trainingParams = json.load(tc)

        # print("Hyperparameters passed:")
        # print(trainingParams)

        s = time.time()
        print("Reading data")
        input_files = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path)]
        raw_data = [pd.read_csv(file_name) for file_name in input_files]
        new_rawData_all = pd.concat(raw_data, ignore_index=True)
        print(new_rawData_all.shape)
        e = time.time()
        print("Reading data took : {}".format(e - s))

        # In[5]:
        # print("Filtering the data on Countries & Category...")
        new_rawData = new_rawData_all  # .loc[((new_rawData_all['Country']=="USA")|
        # (new_rawData_all['Country']=="United Kingdom")|
        # (new_rawData_all['Country']=="Canada")|
        # (new_rawData_all['Country']=="South Africa")|
        # (new_rawData_all['Country']=="New Zealand")|
        # (new_rawData_all['Country']=="Singapore")|
        # (new_rawData_all['Country']=="Malaysia")|
        # (new_rawData_all['Country']=="Ireland")|
        # (new_rawData_all['Country']=="Australia"))&
        # (new_rawData_all['Category']=="10 - Troubleshooting*")]
        # new_rawData.shape

        # In[6]:
        rawdf = new_rawData.loc[new_rawData['Product Line'].isnull() == False][
            ['Case ID', 'Case Number', 'Country', 'DateTime_Opened', 'Product Line', 'Category',
             'Customer Request']].reset_index(drop=True)
        print("Data Shape:")
        print(rawdf.shape)
        print("Unique Product Line:")
        print(rawdf['Product Line'].unique())
        print("Unique Case Category:")
        print(rawdf['Category'].unique())
        # rawdf.head()

        # In[7]:
        ## Extracting Month wise date from 'DateTime_Opened' for later aggregation of monthly issues
        print("Datetime datatype conversion...")
        print(rawdf['DateTime_Opened'].head())
        rawdf['Date raw'] = pd.to_datetime(rawdf['DateTime_Opened'], format='%Y-%m-%d %H:%M:%S')  # '%d/%m/%Y %I:%M %p'
        rawdf['Date_Month'] = pd.to_datetime(rawdf['Date raw'].dt.strftime('1/%m/%Y'), format='%d/%m/%Y')

        # load models
        nlp = spacy.load('./spacy_prod')
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        # extract product/issue triplets from customer request
        print('Initiating Information Extraction Modules...')
        rawdf['triplets'] = rawdf['Customer Request'].apply(process_request)

        # extract key info from triplets
        rawdf['context_issue'] = rawdf['triplets'].apply(isolate_issueContext)
        rawdf['context_issue_cleaned'] = rawdf['context_issue'].apply(clean_issues)

        issues_stacked = rawdf.context_issue_cleaned.apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('Issues').dropna()
        issues_stacked = rawdf.drop('context_issue_cleaned', axis=1).join(issues_stacked, how='inner').reset_index(drop=True)  ## To avoid missing Issues use inner join


        print('Initiating Clustering...')
        # convert info extract to tfidf feature matrix
        tfidf_vectorizer = TfidfVectorizer(max_df=0.1, min_df=0.0015, ngram_range=(1, 2))  #
        feature_matrix = tfidf_vectorizer.fit_transform(flatten_list(rawdf['context_issue_cleaned'].values))
        print(feature_matrix.shape)

        umap_model = umap.UMAP(n_neighbors=30,
                           min_dist=0.0,
                           n_components=10,
                           metric='hellinger')
        # transform feature matrix to UMap embedding
        umap_embeddings = umap_model.fit_transform(feature_matrix)

        # cluster embeddings using HDBScan
        cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  min_samples=1,
                                  cluster_selection_method='eom').fit(umap_embeddings)

        # create doc cluster dataframe
        docs_df = pd.DataFrame(flatten_list(rawdf['context_issue_cleaned'].values), columns=["Doc"])
        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

        print('Cluster naming using TFIDF & Bigram Approach')
        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(flatten_list(rawdf['context_issue_cleaned'].values)))

        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        topic_sizes = extract_topic_sizes(docs_df)
        print(f'Initial Topic Count: {topic_sizes.count()}')

        # loop through topics combining similar clusters
        for i in range(10):
            # Calculate cosine similarity
            similarities = cosine_similarity(tf_idf.T)
            np.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
            topic_to_merge = topic_sizes.iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            old_topics = docs_df.sort_values("Topic").Topic.unique()
            map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
            docs_df.Topic = docs_df.Topic.map(map_topics)
            docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ', '.join})

            # Calculate new topic words
            m = len(flatten_list(rawdf['context_issue_cleaned'].values))
            tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
            top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

        print(f'Optimized Topic Count: {topic_sizes.count()}')
        topic_sizes = extract_topic_sizes(docs_df)

        print('Naming Cluster Topics...')
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=6)

        topic_dict = {}
        for cluster, topics in top_n_words.items():
            # clear duplicates
            topics = [clean_dups(top) for top, score in topics]

            # remove duplicates, list(set) loses ordering
            final_topics = []
            [final_topics.append(x) for x in topics if x not in final_topics]

            # clear unigrams that repeat in bigrams
            final_topics = clear_redundent_unigrams(final_topics)

            # return top 3 words as string
            topic_dict[cluster] = ', '.join(final_topics[0:3])

        topics = pd.DataFrame(topic_dict, index=[0]).T.reset_index()
        cluster_df = pd.merge(docs_df, topics, left_on='Topic', right_on='index').drop(['Doc_ID', 'index'], axis=1)
        cluster_df.columns = ['Issue', 'Cluster', 'Topic']
        print('Total Clusters: ', cluster_df['Topic'].value_counts().count())

        print('Merging Noise documents with most similar clusters')
        noise_inds = cluster_df[cluster_df['Cluster'] == -1].index

        noise = cluster_df.loc[noise_inds].drop(['Cluster', 'Topic'], axis=1)
        classified = cluster_df.drop(noise_inds)
        # compute feature matrix for previous "noise"
        feature_matrix = tfidf_vectorizer.transform(noise['Issue'].values)

        # compute topic feature matrix for previous topics (ignoring "noise" i.e. -1)
        topic_matrix = tfidf_vectorizer.transform(docs_per_topic.Doc.values[1:])

        similarity = cosine_similarity(feature_matrix, topic_matrix)
        top = np.argmax(similarity, axis=1)

        # get top cluster and scores for all docs
        top_scores = [(x, similarity[i][x]) for i, x in enumerate(top)]

        # if tf_idf score over 5% add to new cluster else keep as noise
        noise['Cluster'] = [x[0] if x[1] > 0.05 else -1 for x in top_scores]
        noise = pd.merge(noise, topics, left_on='Cluster', right_on='index')
        noise = noise.drop('index', axis=1)
        noise.columns = ['Issue', 'Cluster', 'Topic']

        noise.loc[noise['Cluster'] == -1, 'Topic'] = 'Noise'

        # update cluster df with noise updates
        cluster_df = classified.append(noise)

        cases_df = pd.merge(issues_stacked[['Case Number', 'Date', 'Severity', 'Customer Request', 'Issues']],
                            cluster_df[['Issue', 'Cluster', 'Topic']],
                            left_on='Issues',
                            right_on='Issue',
                            how='inner').drop_duplicates().reset_index()

        bi_cluster = cases_df.groupby('Case Number')['Cluster'].apply(list)
        tfidf_cluster = cases_df.groupby('Case Number')['Topic'].apply(set)
        cases_df = pd.merge(rawdf, bi_cluster, on='Case Number', how='left').merge(tfidf_cluster, on='Case Number')

        cases_df.to_csv(os.path.join(model_path, "product_category_clustered.csv"), header=True, index=False)
        print('Script done')

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        """
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        """
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == "__main__":
    train_entry_point()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
