#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:34:48 2022

@author: akashchoudhuri
"""
import pandas as pd
import numpy as np
year=[2007,2008]
date=[[31],[31,28,31,30,31,30,31,31]]
month=[[12],[1,2,3,4,5,6,7,8]]
corpus=[]
p=0
whole_df=pd.DataFrame()
for i in range(len(year)):
    years=year[i]
    months=month[i]
    dates=date[i]
    if p==1:
        break
    for j in range(len(months)):
        month1=months[j]
        if years==2008 and month1==11:
            p=1
            break
        if month1<10:
            mon='0'+str(month1)
        else:
            mon=str(month1)
        if years==2008:
            if month1==4:
                dates[j]=11
        date1=1
        while date1<=dates[j]:
            if years==2008:
                if month1==5:
                    if date1==6:
                        date1=7
                        continue
            if years==2008:
                if month1==6:
                    if date1==13:
                        date1=14
                        continue
            if years==2008:
                if month1==9:
                    if date1==1:
                        date1=31
                        continue
            if years==2008:
                if month1==10:
                    if date1==1:
                        date1=14
                        continue
            path='v3/CS_Notes_'
            path=path+str(years)
            if date1<10:
                dat='0'+str(date1)
            else:
                dat=str(date1)
            path=path+mon+dat+'.txt'
            print(path)        
            df=pd.read_csv(path,encoding='unicode_escape',sep='\t',header=None,low_memory=False)
            col_heads=df.loc[0].tolist()
            df.columns=col_heads
            df = df.iloc[1: , :]
            df['NOTE_DATETIME']=df['NOTE_DATETIME'].astype('datetime64[ns]')
            #df['NOTE_DATETIME']=pd.to_datetime(df['NOTE_DATETIME']).dt.date
            #df=df[['NOTE_DATETIME','MEDREC_NUM','NOTE_TEXT']]
            df=df.dropna(subset=['NOTE_TEXT'])
            if len(whole_df)==0:
                whole_df=df
            else:
                whole_df=pd.concat([whole_df,df],axis=0)
            whole_df=whole_df.reset_index(drop=True)
            date1=date1+1
        if years==2008:
            if month1==4:
                dates[j]=30
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
from glob import glob
import json
import pandas as pd
from nltk.corpus import stopwords
print(stopwords.words('english'))
from nltk import word_tokenize
#from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
import string
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm

def load_stopwords():
    stoplist =set(stopwords.words("english"))
    #stop_words=list(stop_words)
    return stoplist

def preprocessing(data_df):
    # lowercase
    print('Lowercase...')
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].str.lower()
    # remove extra whitespaces
    print('Remove consecutive spaces...')
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(lambda x: " ".join(x.split()))
    # tokenization
    print('Tokenization...')
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(lambda x: word_tokenize(x))
    
    ## Normalization
    # remove punctuations
    print('Remove punctuation...')
    def remove_punct(text):
        tokenizer = RegexpTokenizer(r"\w+")
        lst = tokenizer.tokenize(' '.join(text))
        return lst

    def remove_punct(text):
        text = ' '.join(text)
        text = re.sub(f"[{string.punctuation}]", ' ', text)

        tokenizer = RegexpTokenizer(r"\w+")
        res = tokenizer.tokenize(text)
        return res

    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(remove_punct)
    # convert numbers to NUM
    print('Generalize numbers...')
    def convert_number(text):
        text = " ".join(text)
        text = re.compile(r'\d+[\.\,]*\d*').sub('', text)
        return text.split()
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(convert_number)
    
    ## Stemming
    print('Stemming...')
    def stemming(text):
        porter = WordNetLemmatizer()
        result=[]
        for word in text:
            result.append(porter.lemmatize(word))
        return result
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(stemming)

    # remove stopwords
    print('Remove stopwords...')
    stoplist = load_stopwords()
    print("stopwords: ", stoplist)
    data_df['NOTE_TEXT'] = data_df['NOTE_TEXT'].apply(lambda x: [xi for xi in x if xi not in stoplist])
    return data_df
whole_df=preprocessing(whole_df)
def med_tok(text):
    text = " ".join(text)
    doc = nlp(text)
    return doc
# Domain Tokenization
#df_notes['TEXT'] = df_notes['TEXT'].apply(med_tok)
def create_vocab(data_df):
    vocab = set()
    for doc in data_df['NOTE_TEXT']:
        vocab.update(doc)
    return vocab
vocab = create_vocab(whole_df)
print(len(vocab))
def save_vocab(vocab, filename):
    with open(filename, 'wt') as f:
        for w in sorted(vocab):
            f.write(w)
            f.write("\n")
save_vocab(vocab,'data/vocabulary_whole.txt')
def sent_conv(text):
    text = " ".join(text)
    return text
whole_df['NOTE_TEXT'] = whole_df['NOTE_TEXT'].apply(sent_conv)
df11=whole_df[(whole_df['NOTE_DATETIME']>='2008-05-04') & (whole_df['NOTE_DATETIME']<'2008-06-26')]
df12=whole_df[(whole_df['NOTE_DATETIME']>='2008-06-13') & (whole_df['NOTE_DATETIME']<'2008-08-05')]
df13=whole_df[(whole_df['NOTE_DATETIME']>='2008-07-10') & (whole_df['NOTE_DATETIME']<'2008-09-01')]
print(len(df11))
print(len(df12))
print(len(df13))
print(len(whole_df))
df11.to_csv("data/chunk1.csv", index=False)
df12.to_csv("data/chunk2.csv", index=False)
df13.to_csv("data/chunk3.csv", index=False)
whole_df.to_csv("data/WHOLE.csv", index=False)
#df_2011.to_csv("data/chunk5.csv", index=False)
