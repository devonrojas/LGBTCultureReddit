import pandas as pd
import os

from util import flatten

INPUT_DATA_DIR = "data/raw"
OUTPUT_DATA_DIR = "data"

def main():
    dir = os.path.join(INPUT_DATA_DIR, "lgbt.corpus")    

    conversations = pd.read_json(os.path.join(dir, "conversations.json"), 
                                orient="index")
    conversations.index.name = "id"

    utterances = pd.read_json(os.path.join(dir, "utterances.jsonl"), lines=True)
    utterances = utterances.set_index("id")

    # combine utterances by root id (should match to conv id)
    res = utterances.groupby("root").apply(flatten).reset_index(name="text")

    merged = conversations.merge(res, left_on=["id"], right_on=["root"])
    merged['ym'] = merged['timestamp'].dt.strftime("%Y-%m")
    merged['type'] = "lgbtq" # Set type key for topic model

    # save to year folder
    merged.to_csv(os.path.join(OUTPUT_DATA_DIR, "lgbt.csv"))
    

if __name__ == "__main__":
    main()