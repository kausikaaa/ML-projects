# Data handling
import pandas as pd
import numpy as np

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Load the dataset
df=pd.read_csv('data/spam.csv',encoding='latin-1')
print(df.head())
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename for clarity

# Check shape and info
print(df.shape)
print(df.info())

# Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any missing values (safety check)
df.dropna(inplace=True)

print(df.head())  # Show first few rows
print(df['label'].value_counts())  # Show how many ham/spam messages

# Separate features and labels
X = df['message']  # The messages (input)
y = df['label']    # The labels (0 or 1)

# Convert text to numeric features
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_transformed = tfidf.fit_transform(X)  # Now X is a matrix of TF-IDF features