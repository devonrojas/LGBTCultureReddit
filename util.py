import pandas as pd

def combine_texts(row):
    title = row['title']
    text = row['text']

    if type(text) is not float:
            if text.strip():
                    return title + "\n\n" + text

    return title

def flatten(df):
    df = df.sort_values("timestamp")
    texts = df['text']
    texts = "\n\n".join([t.strip() 
                         for t in texts 
                         if t.strip() and t != "[deleted]" and t != "[removed]"])
    return texts

def read_data(dir):
    df = pd.read_csv(dir, index_col=0)
    df['combined'] = df.apply(combine_texts, axis=1)
    df['length'] = df['combined'].apply(len)

    # Filter 1 and 2-length docs
    df = df[df['length']>2]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df = df.dropna(subset="year")
    return df