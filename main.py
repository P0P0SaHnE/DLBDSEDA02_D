# MODULES ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import csv
import re
import nltk
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm   #
tqdm.pandas()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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

print("\n--------------------------------------------------------------------------------------------------------")
print("##         This is the Project for the IU Modul DLBDSEDA02_D with the topic of Data Analysis.         ##")
print("--------------------------------------------------------------------------------------------------------\n")
print(f"\nThe used Dataframe have {corpus_dataframe.shape[0]} rows and {corpus_dataframe.shape[1]} columns.\n")


# remove rows with empty value in complaint_narrative, duplicates & irrelevant columns -----------------------------------------

print("\n> Phase 1: Prepare the Dataframe <")
print("----------------------------------\n")

corpus_dataframe = corpus_dataframe[relevant_columns]   # remove all unuser columns
corpus_dataframe = corpus_dataframe.dropna(subset=[complaint_narrative])    # remove all rows with empty field in complaint_narrative
corpus_dataframe = corpus_dataframe.drop_duplicates()   # remove duplicate rows

print(f"\nThe Dataframe have {corpus_dataframe.shape[0]} rows and {corpus_dataframe.shape[1]} columns after the preparation.\n")


# text cleaning ----------------------------------------------------------------------------------------------------------------

print("\n> Phase 2: Text Cleaning <")
print("--------------------------\n")

corpus_dataframe["clean_text"] = corpus_dataframe[complaint_narrative].progress_apply(clean_text) 

print("\nExample of 10 rows after Text Cleaning:\n")
print(corpus_dataframe.sample(20, random_state=42),"\n")


# tokenizing, lemmatizing -------------------------------------------------------------------------------------------------------

print("\n> Phase 3: Tokenizing & Lemmatize <")
print("-----------------------------------\n")

# try to skip the lammatize process with existing lemma file
try:
    corpus_dataframe = pd.read_csv("lemma_text.csv")

except:
    corpus_dataframe["lemma_text"] = corpus_dataframe["clean_text"].progress_apply(lemmatize_text)

    corpus_dataframe = corpus_dataframe[corpus_dataframe["lemma_text"].str.len() > 0]   # remove with empty fields
    corpus_dataframe = corpus_dataframe[corpus_dataframe["lemma_text"].str.strip() != ""] # remove rows with blankspaces
    corpus_dataframe = corpus_dataframe.reset_index(drop=True)  # reindexing after removing

    corpus_dataframe.to_csv(
    "lemma_text.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    lineterminator="\n"
)

print(corpus_dataframe.sample(10, random_state=42),"\n")

#corpus_dataframe["final_text"] = corpus_dataframe.progress_apply(build_final_text, axis=1)

corpus_dataframe.to_csv(
    "final_text.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    lineterminator="\n"
)


# bow vectorization -------------------------------------------------------------------------------------------------------------

texts = corpus_dataframe["lemma_text"]

bow_vectorizer = CountVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1,1)
)
X_bow = bow_vectorizer.fit_transform(texts)

print("BoW\n")
print("\n",X_bow,"\n")

# tf-idf vektorization -----------------------------------------------------------------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1,1)
)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("TF-IDF\n")
print(X_tfidf,"\n")

#

