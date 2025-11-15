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
from tqdm import tqdm   #
tqdm.pandas()


# VARIABLES --------------------------------------------------------------------------------------------------------------------

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
    stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")  # load the language modell for "english"


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
    text = nlp(text)
    for token in text:
        if not token.is_stop:
            lemma = token.lemma_
            lemmatized_tokens.append(lemma)

    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text


#def build_final_text(row):
#    extras = ' '.join(str(x).lower() for x in [row["product"], row["issue"], row["sub_issue"]] if pd.notna(x))
 #   return extras + " " + row["lemma_text"]


# MAIN -------------------------------------------------------------------------------------------------------------------------

print("\033[32m\n--------------------------------------------------------------------------------------------------------")
print("##         This is the Project for the IU Modul DLBDSEDA02_D with the topic of Data Analysis.         ##")
print("--------------------------------------------------------------------------------------------------------\n\033[0m")
print(f"\nThe used Dataframe have {corpus_dataframe.shape[0]} rows and {corpus_dataframe.shape[1]} columns.\n")


# remove rows with empty value in complaint_narrative, duplicates & irrelevant columns -----------------------------------------

print("\033[32m\n> Phase 1: Prepare the Dataframe <")
print("----------------------------------\n\033[0m")
print("\t- Remove the unused Columns\n\t- Remove all rows with empty Fields in narrative\n\t- Remove duplicate Rows\n")


corpus_dataframe = corpus_dataframe[relevant_columns]   # remove all unuser columns
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


#corpus_dataframe["final_text"] = corpus_dataframe.progress_apply(build_final_text, axis=1)

corpus_dataframe.to_csv(
    "final_text.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    lineterminator="\n"
)


# vectorization -------------------------------------------------------------------------------------------------------------

print("\033[32m\n\n> Phase 4: Vectorization <")
print("--------------------------\n\033[0m")

lemma_texts = corpus_dataframe["lemma_text"]

bow = CountVectorizer(
    max_df=0.9, # ignore extermly frequent words that have frequency by x%
    min_df=5,   # ignore rare words, has to exist in x files 
    ngram_range=(1,1)   # define n-gram (1,2 = unigram & bigram)
)
bow_vector = bow.fit_transform(lemma_texts) # create the bow vector

tfidf = TfidfVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1,1)
)
tfidf_vector = tfidf.fit_transform(lemma_texts)     # create the tf-idf vector

#bow vectorization ----------------------------------------------------------------------------------------------------------

'''
bow_array = pd.DataFrame(
    bow_vector.toarray(),
    columns=bow.get_feature_names_out()
)

print("\nBoW Vector:\n")
print("\n",bow_array,"\n")
'''

# tf-idf vektorization -----------------------------------------------------------------------------------------------------

'''
tfidf_array = pd.DataFrame(
    tfidf_vector.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\nTF-IDF Vector:\n")
print("\n",tfidf_array,"\n")
'''

# compare the bow & tf-idf vectors -----------------------------------------------------------------------------------------------

print("\n############## BoW ##################\n")

word_counts = bow_vector.toarray().sum(axis=0)   # 
feature_names = bow.get_feature_names_out()

top_indices = np.argsort(word_counts)[::-1]   # sort highest first

for idx in top_indices[:20]:  # top 20 words
    print(feature_names[idx], word_counts[idx])


print("\n############## TF-IDF ##################\n")

tfidf_scores = tfidf_vector.toarray().sum(axis=0)
feature_names = tfidf.get_feature_names_out()

top_indices = np.argsort(tfidf_scores)[::-1]

for idx in top_indices[:20]:
    print(feature_names[idx], tfidf_scores[idx])
