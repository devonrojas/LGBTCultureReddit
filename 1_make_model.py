from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from sklearn.feature_extraction import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import re, emoji, pickle

from util import combine_texts

nltk.download("punkt_tab")

DATA_DIR = "data"

def main():
        df = pd.read_csv("lgbt.csv")

        df['combined'] = df.apply(combine_texts, axis=1)
        df['length'] = df['combined'].apply(len)

        # Filter 1 and 2-length docs
        df = df[df['length']>2]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df = df.dropna(subset="year") # There is a na value for one of the years

        texts = df['combined'].to_list()
        dates = df['year'].to_list()

        assert len(texts) == len(dates)
        

        representation_model = {
                "KeyBERT": KeyBERTInspired(),
                "MMR": MaximalMarginalRelevance(diversity=0.2)
        }
        vectorizer_model = CountVectorizer(stop_words="english", min_df=0.05)
        hdbscan_model = HDBSCAN(min_cluster_size=150,
                        metric="euclidean",
                        cluster_selection_method="eom",
                        prediction_data=True)
        umap_model = UMAP(n_neighbors=15,
                        n_components=10,
                        min_dist=0.0,
                        metric="cosine",
                        random_state=1234)
        
        topic_model_sents = BERTopic(
                calculate_probabilities=True,
                representation_model=representation_model,
                vectorizer_model=vectorizer_model,
                hdbscan_model=hdbscan_model,
                umap_model=umap_model,
                top_n_words = 20,
                verbose=True
        )
        topics, probs = topic_model_sents.fit_transform(texts)

        topic_model_sents.save(f"{DATA_DIR}/lgbt_bertopic_unigram_v4_no_reduce.pickle", serialization="pickle")