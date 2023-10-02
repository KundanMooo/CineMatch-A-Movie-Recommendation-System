# Import necessary libraries
import numpy as np
import pandas as pd

# Read the movie credits and movies datasets
dfcred = pd.read_csv('tmdb_5000_credits.csv')
dfmov = pd.read_csv('tmdb_5000_movies.csv')

# Display the first two rows of each dataset
dfmov.head(2)
dfcred.head(2)

# Check for the number of unique values in the credits dataset
dfcred.nunique()

# Check for missing values in the credits dataset
dfcred.isna().sum()

# Check for the number of unique values in the movies dataset
dfmov.nunique()

# Check for missing values in the movies dataset
dfmov.isna().sum()

# Merge the credits and movies datasets using 'id' and 'movie_id' as keys
dfmov = pd.merge(dfmov, dfcred, left_on='id', right_on='movie_id', how='inner')

# Display the first row of the merged dataset and its columns
dfmov.head(1)
dfmov.columns

# Convert 'release_date' to a year format
dfmov['release_date'] = pd.to_datetime(dfmov['release_date']).dt.year

# Select specific columns for analysis
selected_columns = ['genres', 'id', 'keywords', 'title_x', 'overview', 'cast', 'crew']
dfmov1 = dfmov[selected_columns]

# Display the first three rows of the selected columns
dfmov1.head(3)

# Check for missing values in the selected columns
dfmov1.isna().sum()

# Drop rows with missing values
dfmov1 = dfmov1.dropna()

# Check for missing values again
dfmov1.isna().sum()

# Remove duplicates from the dataset
dfmov1.duplicated().sum()

# Define a function to extract names from a list of dictionaries
import ast

def extract_names(listdict):
    names = []
    for item in ast.literal_eval(listdict):
        names.append(item['name'])
    return names

# Apply the function to 'genres' and 'keywords' columns
dfmov1['genres'] = dfmov1['genres'].apply(extract_names)
dfmov1['keywords'] = dfmov1['keywords'].apply(extract_names)

# Define a function to extract the first 3 names from a list of dictionaries
def extract_top3_names(listdict):
    names = []
    count = 0
    for item in ast.literal_eval(listdict):
        if count < 3:
            names.append(item['name'])
            count += 1
        else:
            break
    return names

# Apply the function to 'cast' column
dfmov1['cast'] = dfmov1['cast'].apply(extract_top3_names)

# Define a function to extract writer and director names from a list of dictionaries
def extract_writers_directors(listdict):
    names = []
    for item in ast.literal_eval(listdict):
        if item['job'] == 'Writer' or item['job'] == 'Director':
            names.append(item['name'])
    return names

# Apply the function to 'crew' column
dfmov1['crew'] = dfmov1['crew'].apply(extract_writers_directors)

# Tokenize the 'overview' column
dfmov1['overview'] = dfmov1['overview'].apply(lambda x: x.split())

# Remove spaces from elements in lists
dfmov1['genres'] = dfmov1['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
dfmov1['keywords'] = dfmov1['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
dfmov1['cast'] = dfmov1['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
dfmov1['crew'] = dfmov1['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
dfmov1['overview'] = dfmov1['overview'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create a new column 'decision' by combining selected columns
dfmov1['decision'] = dfmov1['genres'] + dfmov1['keywords'] + dfmov1['overview'] + dfmov1['cast'] + dfmov1['crew']

# Select relevant columns
dfmov11 = dfmov1[['title_x', 'id', 'decision']]

# Join the elements in the 'decision' column into a single string
dfmov11['decision'] = dfmov11['decision'].apply(lambda x: " ".join(x))

# Convert 'decision' column to lowercase
dfmov11['decision'] = dfmov11['decision'].apply(lambda x: x.lower())

# Save the resulting DataFrame to a CSV file
dfmov11.to_csv('radkeee.csv', index=False)

# Import CountVectorizer from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer with max_features and stop_words
cv = CountVectorizer(max_features=4000, stop_words='english')

# Transform 'decision' column into a numerical vector
vec = cv.fit_transform(dfmov11['decision']).toarray()

# Import NLTK library for text processing
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Define a function to apply stemming to text
def stem_text(text):
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(ps.stem(word))
    return " ".join(stemmed_words)

# Apply stemming to the 'decision' column
dfmov11['decision'] = dfmov11['decision'].apply(stem_text)

# Save the resulting DataFrame to a new CSV file
dfmov11.to_csv('radkeee11.csv', index=False)

# Transform 'decision' column into a numerical vector again after stemming
vec = cv.fit_transform(dfmov11['decision']).toarray()

# Import cosine_similarity function from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity matrix
ss = cosine_similarity(vec)

# Define a function to recommend movies based on similarity
def recommend(movie):
    indexx = dfmov11[dfmov11['title_x'] == movie].index[0]
    movielist = sorted(list(enumerate(ss[indexx])), reverse=True, key=lambda x: x[1])[1:6]
    for i in movielist:
        print(dfmov11.iloc[i[0]].title_x)

# Recommend movies similar to 'Batman Begins'
recommend('Batman Begins')

# Recommend movies similar to 'Avatar'
recommend('Avatar')
