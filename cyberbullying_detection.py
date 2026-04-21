
#!/usr/bin/env python

import re, copy, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import os

import nltk
# Only download if not already present
project_path = "./data/" # path of your datafiles

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib # Import joblib for model persistence

print('Libraries ready')

CLASSIFIERS = {
    'SVM':        LinearSVC(C=1.0, max_iter=2000, random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute'),
    'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3, metric='cosine', algorithm='brute'),
}

TFIDF_VARIANTS = {
    'Unigram (1,1)': TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, max_features=10000, min_df=2, max_df=0.95),
    'Bigram (2,2)': TfidfVectorizer(ngram_range=(2,2), sublinear_tf=True, max_features=10000, min_df=2, max_df=0.95),
    'Uni+Bigram (1,2)': TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=15000, min_df=2, max_df=0.95),
}
print('Classifiers:', list(CLASSIFIERS.keys()))
print('TF-IDF variants:', list(TFIDF_VARIANTS.keys()))

df = pd.read_csv(project_path + 'cyberbullying_tweets.csv')

print(f"\nMissing values: {df.isnull().sum().sum()}")
df = df.dropna(subset=['tweet_text', 'cyberbullying_type'])
print(f"After dropping nulls: {len(df)} rows")

print(f'Shape: {df.shape}')
print('\nClass distribution:')
print(df['cyberbullying_type'].value_counts())
print('First 5 rows of the dataset:')
print(df.head(5))

df['label'] = df['cyberbullying_type'].apply(
    lambda x: 0 if x == 'not_cyberbullying' else 1)
df['label_name'] = df['label'].map({0: 'Normal', 1: 'Bully'})


print(f"Normal: {(df['label']==0).sum()}  |  Bully: {(df['label']==1).sum()}")

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['rt','via','http','https','amp','get','got'])
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)            # remove @mentions
    text = re.sub(r'#\w+', '', text)            # remove #hashtags
    text = re.sub(r'&[a-z]+;', '', text)        # HTML entities
    text = re.sub(r'[^a-z\s]', '', text)        # letters only
    return re.sub(r'\s+', ' ', text).strip()

def preprocess(text):
    text   = clean_text(text)                                       # Clean
    tokens = word_tokenize(text)                                    # Tokenize
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]  # Stopwords
    tokens = [stemmer.stem(t) for t in tokens]                     # Stem
    return ' '.join(tokens)

print('Processing tweets (~1 min)...')
df['processed_text'] = df['tweet_text'].apply(preprocess)
df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)

for _, row in df.sample(3, random_state=1).iterrows():
    print(f'Label    : {row.label_name}')
    print(f'Raw      : {row.tweet_text[:80]}')
    print(f'Processed: {row.processed_text[:80]}')
    print()
print(f' {len(df)} samples ready')

print("\nPRE-PROCESSING EXAMPLES:")
print(f"{'Original Tweet':<50} {'Processed'}")
print("-" * 80)
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"{str(row['tweet_text'])[:48]:<50} {row['processed_text'][:30]}")

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)
print(f'Train: {len(X_train)}  |  Test: {len(X_test)}')

results    = []
fit_models = {}

for vec_name, vectorizer in TFIDF_VARIANTS.items():
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    print(f'\n[{vec_name}] — {X_tr.shape[1]} features')
    print(f"  {'Classifier':<15} {'Acc%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7}")
    print(f"  {'-'*15} {'-----':>7} {'-----':>7} {'----':>7} {'---':>7}")

    for clf_name, clf_template in CLASSIFIERS.items():
        clf = copy.deepcopy(clf_template)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='binary', zero_division=0)

        print(f"  {clf_name:<15} {acc*100:>6.2f}% {prec*100:>6.2f}% {rec*100:>6.2f}% {f1*100:>6.2f}%")

        results.append({
            'TF-IDF Variant': vec_name, 'Classifier': clf_name,
            'Accuracy': round(acc*100,2), 'Precision': round(prec*100,2),
            'Recall':   round(rec*100,2), 'F1-Score':  round(f1*100,2),
            # Removed '_cm' and '_y_pred' for standalone script
        })
        fit_models[f'{vec_name}|{clf_name}'] = (vectorizer, clf)

print("5-FOLD CROSS VALIDATION")
print("=" * 55)
print("(More reliable than a single train/test split)\n")

cv_results = []

for vec_name, vectorizer in TFIDF_VARIANTS.items():
    for clf_name, clf_template in CLASSIFIERS.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=vectorizer.ngram_range,
                sublinear_tf=True,
                max_features=vectorizer.max_features,
                min_df=2, max_df=0.95
            )),
            ('clf', copy.deepcopy(clf_template))
        ])

        scores = cross_val_score(
            pipe, df['processed_text'], df['label'],
            cv=5, scoring='f1', n_jobs=-1
        )

        mean_f1 = round(scores.mean() * 100, 2)
        std_f1  = round(scores.std()  * 100, 2)

        print(f"  {vec_name} + {clf_name}")
        print(f"    CV F1 = {mean_f1}% ± {std_f1}%")

        # Retrieve single-split F1-Score from results list directly
        single_split_f1 = next(r['F1-Score'] for r in results if r['TF-IDF Variant']==vec_name and r['Classifier']==clf_name)
        print(f"    Single-split F1 = {single_split_f1}%\n")

        cv_results.append({
            'TF-IDF Variant': vec_name,
            'Classifier':     clf_name,
            'CV F1 Mean':     mean_f1,
            'CV F1 Std':      std_f1
        })

cols = ['TF-IDF Variant','Classifier','Accuracy','Precision','Recall','F1-Score']
df_r = pd.DataFrame(results)[cols].sort_values('F1-Score', ascending=False)

# Removed display(df_r.style.background_gradient(subset=['Accuracy','F1-Score'], cmap='YlOrRd')) for standalone script
print('\nResults Table:')
print(df_r.to_string())

best = df_r.iloc[0]
print(f"\n★  Best: {best['TF-IDF Variant']} + {best['Classifier']}")
print(f"   Acc={best['Accuracy']}%  Prec={best['Precision']}%  Rec={best['Recall']}%  F1={best['F1-Score']}% ")

# Access the best model objects
best_vec_name = best['TF-IDF Variant']
best_clf_name = best['Classifier']
best_vectorizer, best_classifier = fit_models[f'{best_vec_name}|{best_clf_name}']

# Save the best vectorizer and classifier
joblib.dump(best_vectorizer, 'best_vectorizer.pkl')
joblib.dump(best_classifier, 'best_classifier.pkl')
print(f"\nBest model ({best_vec_name} + {best_clf_name}) saved as best_vectorizer.pkl and best_classifier.pkl")


# Panels 5 & 6: Confusion matrices (best SVM vs best k-NN)
# For standalone script, we'll just print the classification report
svm_res  = [r for r in results if r['Classifier']=='SVM']
knn_res  = [r for r in results if 'k-NN' in r['Classifier']]
best_svm = sorted(svm_res, key=lambda r: r['F1-Score'], reverse=True)[0]
best_knn = sorted(knn_res, key=lambda r: r['F1-Score'], reverse=True)[0]

# Added this for the standalone script to make confusion matrix accessible
for res in [best_svm, best_knn]:
    # Need to re-predict to get y_pred and confusion matrix for standalone script
    vec_name = res['TF-IDF Variant']
    clf_name = res['Classifier']
    vectorizer, clf = fit_models[f'{vec_name}|{clf_name}']
    X_te = vectorizer.transform(X_test)
    y_pred = clf.predict(X_te)
    res['_cm'] = confusion_matrix(y_test, y_pred)
    res['_y_pred'] = y_pred

for res, header in [(best_svm,'BEST SVM'), (best_knn,'BEST k-NN')]:
    print('='*55)
    print(f"{header} — {res['TF-IDF Variant']} + {res['Classifier']}")
    print('='*55)
    print(classification_report(y_test, res['_y_pred'],
                                 target_names=['Normal (0)','Bully (1)']))
    print('\nConfusion Matrix:')
    print(res['_cm'])

# Demo: Predict New Text (using the persisted best model)
label_map  = {0: '✅ Normal', 1: '🚨 Bully'}
demo_texts = [
    'You are so stupid and ugly, nobody likes you',
    'Have a great day! Hope you are doing well.',
    'I hate all people like you, go away loser',
    'Thanks for this informative video!',
    'Kill yourself you worthless piece of trash',
    'Great work everyone, proud of this team!',
    'Hi there!'
]

# Load the best model for demonstration
loaded_vectorizer = joblib.load('best_vectorizer.pkl')
loaded_classifier = joblib.load('best_classifier.pkl')

f1_val = next(r['F1-Score'] for r in results
                if r['TF-IDF Variant']==best_vec_name and r['Classifier']==best_clf_name)

out  = f"\n{'='*55}\n"
out += f"  DEMO using PERSISTED BEST MODEL: {best_vec_name} + {best_clf_name}\n"
out += f"  F1    : {f1_val}%\n"
out += f"{'='*55}\n"
for text in demo_texts:
    proc = preprocess(text)
    # Use the loaded models for prediction
    pred = loaded_classifier.predict(loaded_vectorizer.transform([proc]))[0]
    out += f"  Input : {text}\n"
    out += f"  Label : {label_map[pred]}\n\n"

print(out)

"""### Interactive Bullying Text Detector

Enter text below to see if the model classifies it as 'Normal' or 'Bully'. Type 'exit' or 'quit' to end the session.
"""

print('\n' + '='*55)
print('  START INTERACTIVE DETECTION (type "exit" or "quit" to stop)')
print('='*55)

while True:
    user_input = input("\nEnter your text (or 'exit'/'quit'): ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting interactive session.")
        break

    if not user_input.strip():
        print("Please enter some text.")
        continue

    processed_input = preprocess(user_input)
    if not processed_input:
        print("Couldn't process the input text effectively. Please try a different phrase.")
        continue

    # Use the loaded models for prediction
    prediction = loaded_classifier.predict(loaded_vectorizer.transform([processed_input]))[0]
    predicted_label = label_map[prediction]

    print(f"  Your Text: {user_input}")
    print(f"  Prediction: {predicted_label}")

    if prediction == 1: # If classified as Bully
        print("  Status: 🚫 Rejected (Bullying content detected).")
    else:
        print("  Status: ✅ Accepted (Normal content).")
