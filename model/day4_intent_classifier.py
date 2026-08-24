from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

texts = [
    "I want to book an appointment", "Can I schedule a visit for next week",
    "I'd like to make an appointment", "I need to see the dentist soon",
    "Book me in for a checkup", "I want an appointment this Friday",
    "Is it possible to schedule a consultation", "I'd like to set something up",
    "Can I reschedule my appointment", "I need to move my appointment to another day",
    "Can we change the time of my visit", "I can't make Tuesday, can we shift it",
    "Please move my booking to next week", "I need a different time slot",
    "I want to cancel my appointment", "Please cancel my booking",
    "I won't be able to make it, cancel please", "Cancel my visit for tomorrow",
    "I need to call off my appointment", "Remove my booking please",
    "What are your opening hours", "Do you accept walk-ins",
    "How much does a checkup cost", "Where are you located",
    "Do you take insurance", "What services do you offer",
]

labels = [
    "book","book","book","book","book","book","book","book",
    "reschedule","reschedule","reschedule","reschedule","reschedule","reschedule",
    "cancel","cancel","cancel","cancel","cancel","cancel",
    "other","other","other","other","other","other",
]

print(f"Total examples: {len(texts)}")

# Split into train/test
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
predictions = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nDetailed report:\n", classification_report(y_test, predictions))

# Test on new unseen sentences
test_sentences = [
    "I'd like to set up a visit",
    "Can you move my slot to Friday",
    "I have to call off my appointment",
]
test_X = vectorizer.transform(test_sentences)
test_predictions = clf.predict(test_X)

print("\nTesting on brand new sentences:")
for sentence, pred in zip(test_sentences, test_predictions):
    print(f"'{sentence}' -> predicted intent: {pred}")

# Save the model
os.makedirs("model/saved", exist_ok=True)
joblib.dump(clf, "model/saved/intent_classifier.pkl")
joblib.dump(vectorizer, "model/saved/vectorizer.pkl")
print("\nModel saved to model/saved/")