import os
import re
import string

import pandas as pd

from stemmer import stemmer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def load_stop_words_from_csv():
    file_path = BASE_DIR + f'/stop_words.csv'
    stop_words = pd.read_csv(file_path, names=["stop_words"])
    return stop_words["stop_words"].values


def remove_stop_words(df):
    stop_words = load_stop_words_from_csv()
    string_punch = string.punctuation + '।'
    punch = str.maketrans("", "", string_punch)
    df["text"] = df['text'].apply(lambda x: ' '.join([word for word in str(re.sub('\s+', " ", str(x).translate(punch)))
                                                     .split() if word not in stop_words and not (word.isalpha()
                                                                                                 or word.isalnum())]))
    return df


def stem_data(df):
    df['text'] = df.apply(stemmer, args=('text',), axis=1)
    return df


def clean_data():
    try:
        appended_data = []
        files = ['automobile.csv', 'crime,csv', 'entertainment.csv', 'health.csv', 'politics.csv', 'sport.csv']
        for file in files:
            file_path = BASE_DIR + f'/data/{file}'
            df = pd.read_csv(file_path)
            df['text'] = df['title'] + df['content']
            df = remove_stop_words(df)
            df = stem_data(df)
            appended_data.append(df)
            print(f'{file}')
        file_path = BASE_DIR + f'/data/data_output.csv'
        result = pd.concat(appended_data)
        result.to_csv(file_path, columns=['text', 'category'], index=False)
    except Exception as e:
        print(str(e))
