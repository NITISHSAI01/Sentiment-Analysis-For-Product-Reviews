import pandas as pd
import re
import string
import emoji  # Added for handling emojis
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('C:/Users/hi/Downloads/amazon_vfl_reviews.csv')

# Preprocess reviews to handle emojis
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert emojis to text descriptions
    text = emoji.demojize(text)  # Converts emoji to a format like :grinning_face:
    
    # Lowercase, remove punctuation, and remove numbers
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Target variable (0 for negative, 1 for neutral, 2 for positive)
def sentiment_label(rating):
    if rating < 3:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

X = df['cleaned_review']
y = df['rating'].apply(sentiment_label)

# Split the data
# 
#  into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)

# Test both models 
nb_pred = nb_model.predict(X_test_vec)
lr_pred = lr_model.predict(X_test_vec)

# Calculate accuracy for both models
nb_accuracy = accuracy_score(y_test, nb_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Naive Bayes Model Accuracy: {nb_accuracy}")
print(f"Logistic Regression Model Accuracy: {lr_accuracy}")

# Choose the best model
if lr_accuracy > nb_accuracy:
    best_model = lr_model
    print("Logistic Regression selected as the best model.")
else:
    best_model = nb_model
    print("Naive Bayes selected as the best model.")

# Save the best-performing model to a .pkl file
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Save the vectorizer to a .pkl file
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Best model and vectorizer saved as 'sentiment_model.pkl' and 'vectorizer.pkl'.")
