"""This file is the pre-processing pipeline.
It splits the fake news dataset into 70 train, 10 val, 20 test.
It creates a tokenizer object file that other models can call for consistency."""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

#Put True.csv and Fake.csv in the same directory as this script
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

#Assign real and fake labels
true_df['label'] = 0 #real=0
fake_df['label'] = 1 #fake=1

#Then shuffle all the true and false together in one frame
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle

#combine title and text into one piece of data
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

def clean_text(text):
    """Remove URLs, brackets, non alpha-numerics, and spaces"""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'\[.*?\]', '', text)                  
    text = re.sub(r'[^a-z\s]', '', text)                 
    text = re.sub(r'\s+', ' ', text).strip()             
    return text

#Tokenization, stop-word removal, and lemmatization
lemmatizer = WordNetLemmatizer() #reduces words to base forms, like "running" to "run"
stop_words = set(stopwords.words('english')) #make set of meaningless words like "the"

def tokenize_and_lemmatize(text):
    #split words into tokens, like "hello world" to ["hello", "world"]
    tokens = word_tokenize(text) 
    #loop through tokens and lemmatize them while skipping stop_words
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens) #TF-IDF expects string not list

print("Cleaning text (takes a few minutes)")
df['cleaned'] = df['content'].apply(clean_text)
df['processed'] = df['cleaned'].apply(tokenize_and_lemmatize)

#Train 70, Val 10, Test 20
X = df['processed'].replace('', pd.NA).dropna() #drop nans
y = df.loc[X.index, 'label'] #align y indices with whatever gets dropped

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp)
#stratify ensures ratio of real to fake is the same after split

#Fit TF-IDF on training data only, then transform all splits
#Converts text into numbers
#Ranks the top 50,000 words with unique words scoring higher
#ngram_range(1,2) captures single words and two-word pairs to keep some context.
#Don't fit on val or test because that's like cheating.
#Later call .transform() on all 3 splits using the same vectorizer here from training
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
vectorizer.fit(X_train)

#Save Outputs
os.makedirs('data', exist_ok=True)

#make dataframes for 3 splits
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df   = pd.DataFrame({'text': X_val,   'label': y_val})
test_df  = pd.DataFrame({'text': X_test,  'label': y_test})

#converts 3 splits to CSVs
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv',     index=False)
test_df.to_csv('data/test.csv',   index=False)

#save vectorizer from this training
joblib.dump(vectorizer, 'data/tfidf_vectorizer.joblib')

print(f"Done!")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"Files saved to: data/train.csv, data/val.csv, data/test.csv")
print(f"Vectorizer saved to: data/tfidf_vectorizer.joblib")