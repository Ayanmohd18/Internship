# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================================
# 2. Load Dataset
# ================================
# Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

# ================================
# 3. Encode Labels
# ================================
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ================================
# 4. Text Preprocessing
# ================================
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()
    return text

df['message'] = df['message'].apply(preprocess_text)

# ================================
# 5. Split Data
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# ================================
# 6. Vectorization (TF-IDF)
# ================================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================================
# 7. Model Training
# ================================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ================================
# 8. Predictions & Evaluation
# ================================
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================================
# 9. Test with Custom Input
# ================================
def predict_spam(text):
    text = preprocess_text(text)
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example tests
print("\nCustom Test 1:", predict_spam("Congratulations! You have won $1000. Click here to claim."))
print("Custom Test 2:", predict_spam("Hey, are we still on for lunch today?"))
