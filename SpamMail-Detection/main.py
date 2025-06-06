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