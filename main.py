# MODULES ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import csv
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from tqdm import tqdm
tqdm.pandas()


# GLOBAL VARIABLES --------------------------------------------------------------------------------------------------------------------

corpus_dataframe = pd.read_csv(     
    "costumer_complain_data/consumer_complaints.csv",   # define the datafile
    sep=",",                 # choose seperator
    quotechar='"',           # character start and end of quoted field
    engine="python",         # python parser
    on_bad_lines="warn",     # warn at bad line interpretation
)

complaint_narrative = "consumer_complaint_narrative"    # define columne in the dataframe that is important
relevant_columns = ["product", "issue","sub_issue", "consumer_complaint_narrative"]     # define all relevant columns in the dataframe
try:
    stop_words = set(stopwords.words('english'))    # download if stopwords not found
except LookupError:
    nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")  # load the language model
vector_best_words = 25 # define output for most words in vector
topic_quantity = 10 # define topic quantity


# FUNCTIONS ------------------------------------------------------------------------------------------------------------------

def clean_text(text):
# text preparation and removing of stopwords 

    text = text.lower() # all letter in lower
    text = text.replace('xxxx', '').replace('xx', '') # remove xxx from text
    text = re.sub(r'[^a-z\s]', '', text).strip()  # replace everything that is not letter from a - z or blankspace with nothing

    words = text.split()
    cleaned_words = []  # empty list

    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)
    text = ' '.join(cleaned_words)  # join the single words into text with black space between

    return text


def lemmatize_text(text):
# lemmatization

    lemmatized_tokens = []
    text = nlp(text)    # process text in language model 
    for token in text:
        if not token.is_stop:
            lemma = token.lemma_
            lemmatized_tokens.append(lemma)

    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text


def display_topics(model, feature_names, n_top_words=10):
# print the topics
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

display_topics(lda, bow.get_feature_names_out())


# MAIN -------------------------------------------------------------------------------------------------------------------------

print("\033[32m\n--------------------------------------------------------------------------------------------------------")
print("##         This is the Project for the IU Modul DLBDSEDA02_D with the topic of Data Analysis.         ##")
print("--------------------------------------------------------------------------------------------------------\n\033[0m")
print(f"\nThe used Dataframe have {corpus_dataframe.shape[0]} rows and {corpus_dataframe.shape[1]} columns.\n")


# remove rows with empty value in complaint_narrative, duplicates & irrelevant columns -----------------------------------------

print("\033[32m\n> Phase 1: Prepare the Dataframe <")
print("----------------------------------\n\033[0m")

if os.path.exists("lemma_text.csv"):
    print("Prepared Dataframe found! - Data Preparation \033[31mSKIPPED\033[0m")

else:

    print("\t- Remove the unused Columns\n\t- Remove all rows with empty Fields in narrative\n\t- Remove duplicate Rows\n")


    corpus_dataframe = corpus_dataframe[relevant_columns]   # remove all unused columns
    corpus_dataframe = corpus_dataframe.dropna(subset=[complaint_narrative])    # remove all rows with empty field in complaint_narrative
    corpus_dataframe = corpus_dataframe.drop_duplicates()   # remove duplicate rows

    print(f"The Dataframe have {corpus_dataframe.shape[0]} rows and {corpus_dataframe.shape[1]} columns after the preparation.\n")


# text cleaning ----------------------------------------------------------------------------------------------------------------

print("\033[32m\n> Phase 2: Text Cleaning <")
print("--------------------------\n\033[0m")

if os.path.exists("lemma_text.csv"):
    print("Existing cleaned Text found! - Text Cleaning \033[31mSKIPPED\033[0m")

else:
    print("\t- All letters in lower characters\n\t- Remove all X's in Text\n\t- Remove everything except a - z and white spaces\n\t- Remove Stop Words\n")
    print("Added new column with cleaned Text.\n")

    corpus_dataframe["clean_text"] = corpus_dataframe[complaint_narrative].progress_apply(clean_text) 


# tokenizing, lemmatizing -------------------------------------------------------------------------------------------------------

print("\033[32m\n\n> Phase 3: Tokenizing & Lemmatize <")
print("-----------------------------------\n\033[0m")

# try to skip the lammatize process with existing lemma file
try:
    corpus_dataframe = pd.read_csv("lemma_text.csv")
    print("Existing Lemma File found! - Lemmatization \033[31mSKIPPED\033[0m")

except:
    print("No existing Lemma File found!\n")
    print("\t- Reduction of inflected words to their common root\n")

    corpus_dataframe["lemma_text"] = corpus_dataframe["clean_text"].progress_apply(lemmatize_text)

    corpus_dataframe = corpus_dataframe[corpus_dataframe["lemma_text"].str.len() > 0]   # remove rows with empty fields
    corpus_dataframe = corpus_dataframe[corpus_dataframe["lemma_text"].str.strip() != ""] # remove rows with blankspaces
    corpus_dataframe = corpus_dataframe.reset_index(drop=True)  # reindexing after removing

    corpus_dataframe.to_csv(
    "lemma_text.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    lineterminator="\n"
)
    print("\nNew Lemma File \033[32mCREATED\033[0m")

# vectorization -------------------------------------------------------------------------------------------------------------

print("\033[32m\n\n> Phase 4: Vectorization <")
print("--------------------------\n\033[0m")

lemma_texts = corpus_dataframe["lemma_text"]

bow = CountVectorizer(  # define bow parameters
    max_df=0.9, # ignore extermly frequent words that have frequency by x%
    min_df=5,   # ignore rare words, has to exist in x files 
    ngram_range=(1,1)   # define n-gram (1,2 = unigram & bigram)
)
bow_vector = bow.fit_transform(lemma_texts) # create the bow vector & box dictionary

tfidf = TfidfVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1,1)
)
tfidf_vector = tfidf.fit_transform(lemma_texts)     # create the tf-idf vector


# compare the bow & tf-idf vectors -----------------------------------------------------------------------------------------------

print(f"Bow Vector (Most {vector_best_words} Words):\n")

word_counts = bow_vector.toarray().sum(axis=0)   # convert to np array and summarize it for every columne
feature_names = bow.get_feature_names_out() # get vocabs from the bow dictionary

top_indices = np.argsort(word_counts)[::-1]   # sort word_counts descending

print("Number\t  Feature Name\t  Word Count ")
print("--------------------------------------")
number = 1
for dict_id in top_indices[:vector_best_words]:  # top "vector_best_words" words
    print("{:<10}{:<16}{:<10}".format(number,feature_names[dict_id],word_counts[dict_id]))
    number += 1


print(f"\n\nTF-IDF Vector (Most {vector_best_words} Words):\n")

tfidf_scores = tfidf_vector.toarray().sum(axis=0)
feature_names = tfidf.get_feature_names_out()

top_indices = np.argsort(tfidf_scores)[::-1]

print("Number\t  Feature Name\t  TF-IDF value")
print("---------------------------------------------")
number = 1
for dict_id in top_indices[:vector_best_words]:
    print("{:<10}{:<16}{:<10}".format(number,feature_names[dict_id], tfidf_scores[dict_id]))
    number += 1







# topic analysis ------------------------------------------------------------------------------------------------------

print("\033[32m\n\n> Phase 5: Topic Analysis <")
print("---------------------------\n\033[0m")

# LDA = Latent Dirichlet Allocation
# NMF = Non-Negative Matrix Factorization


lda = LatentDirichletAllocation(    #define the lda parameters
    topic_quantity,
    random_state=42
)



lda.fit(bow_vector)


# 2) define nfm modell 
nmf = NMF(
    topic_quantity,  
    random_state=42
)
nmf.fit(tfidf_vector)

# 3) Topics show
display_topics(nmf, tfidf.get_feature_names_out())
