import pandas as pd
import numpy as np
import re
import string
import requests
import io
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text


def load_20newsgroups_data():
    """Load data from 20 Newsgroups dataset"""
    print("Loading 20 Newsgroups dataset")

    sport_categories = ['rec.sport.baseball', 'rec.sport.hockey']
    politics_categories = ['talk.politics.guns',
                           'talk.politics.mideast', 'talk.politics.misc']

    categories = sport_categories + politics_categories

    dataset = fetch_20newsgroups(
        subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    texts = []
    labels = []

    for text, target in zip(dataset.data, dataset.target):
        texts.append(text)
        category_name = dataset.target_names[target]
        if 'sport' in category_name:
            labels.append('Sport')
        else:
            labels.append('Politics')

    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"  20 Newsgroups: {len(df)} documents")
    print(
        f"  Sport: {len(df[df['label'] == 'Sport'])}, Politics: {len(df[df['label'] == 'Politics'])}")

    return df


def load_bbc_data():
    """Load data from BBC News dataset"""
    print("\nLoading BBC News dataset")

    url = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Try different encodings
        try:
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.StringIO(
                    response.content.decode('latin1')))
            except:
                df = pd.read_csv(io.StringIO(
                    response.content.decode('iso-8859-1')))

        # The BBC dataset has columns: text, category
        df.columns = ['text', 'label']

        # Normalize labels
        df['label'] = df['label'].str.lower()

        # Filter for sport and politics only
        df = df[df['label'].isin(['sport', 'politics'])].copy()

        # Capitalize labels to match 20newsgroups format
        df['label'] = df['label'].str.capitalize()

        print(f"  BBC News: {len(df)} documents")
        print(
            f"  Sport: {len(df[df['label'] == 'Sport'])}, Politics: {len(df[df['label'] == 'Politics'])}")

        return df

    except Exception as e:
        print(f"  Warning: Could not download BBC data: {e}")
        print("  Continuing with only 20 Newsgroups dataset")
        return pd.DataFrame(columns=['text', 'label'])


def prepare_data():
    """Load and prepare data with train/validation/test splits"""

    # Load both datasets
    df_20news = load_20newsgroups_data()
    df_bbc = load_bbc_data()

    # Combine datasets
    df = pd.concat([df_20news, df_bbc], ignore_index=True)
    print(f"\nCombined dataset: {len(df)} documents")
    print(
        f"Sport: {len(df[df['label'] == 'Sport'])}, Politics: {len(df[df['label'] == 'Politics'])}")

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean text
    print("\nCleaning text data")
    df['clean_text'] = df['text'].apply(clean_text)

    # Remove very short documents
    df = df[df['clean_text'].str.len() > 50].reset_index(drop=True)
    print(f"Documents after cleaning: {len(df)}")

    # Split data: 70% train, 15% validation, 15% test
    X = df['clean_text']
    y = df['label']

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Second split: split temp into validation and test (50/50)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nData splits:")
    print(f"Training: {len(X_train)} documents")
    print(f"Validation: {len(X_val)} documents")
    print(f"Test: {len(X_test)} documents")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print("\nData preparation complete")
