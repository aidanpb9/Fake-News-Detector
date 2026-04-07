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

# ── Load Data ──────────────────────────────────────────────────────────────────
# Place True.csv and Fake.csv in the same directory as this script
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

true_df['label'] = 0  # Real = 0
fake_df['label'] = 1  # Fake = 1

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# ── Combine title + text ───────────────────────────────────────────────────────
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# ── Cleaning ───────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'\[.*?\]', '', text)                  # remove brackets
    text = re.sub(r'[^a-z\s]', '', text)                 # remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()             # remove extra spaces
    return text

# ── Tokenization, Stop-word removal, Lemmatization ────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

print("Cleaning text... (this may take a few minutes)")
df['cleaned'] = df['content'].apply(clean_text)
df['processed'] = df['cleaned'].apply(tokenize_and_lemmatize)

# ── Train/Val/Test Split (70/10/20) ───────────────────────────────────────────
X = df['processed']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp
)

# ── Fit TF-IDF on training data only, then transform all splits ───────────────
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
vectorizer.fit(X_train)

# ── Save Outputs ───────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)

train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df   = pd.DataFrame({'text': X_val,   'label': y_val})
test_df  = pd.DataFrame({'text': X_test,  'label': y_test})

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv',     index=False)
test_df.to_csv('data/test.csv',   index=False)

joblib.dump(vectorizer, 'data/tfidf_vectorizer.joblib')

print(f"Done!")
print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Testing samples    : {len(test_df)}")
print(f"Files saved to     : data/train.csv, data/val.csv, data/test.csv")
print(f"Vectorizer saved to: data/tfidf_vectorizer.joblib")