import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMS spam dataset (example format: first column contains the labels, second column contains the text)
# You can replace 'sms_spam_dataset.csv' with your dataset file name and appropriate columns
sms_data = pd.read_csv('sms_spam_dataset.csv', encoding='latin-1')

# Data preprocessing
# Convert labels to binary (1 for spam, 0 for non-spam)
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sms_data['text'], sms_data['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
