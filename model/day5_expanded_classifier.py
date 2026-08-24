from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Original Day 4 data
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

# New synthetic data (10 per category)
new_texts = [
    "Hi, um, so I was hoping to get an appointment set up",
    "Yeah I need to come in sometime this week if possible",
    "Could you fit me in for a cleaning",
    "I'm calling to arrange a visit for my son",
    "So I need an appointment, is Thursday open",
    "Hi there, I'd like to get booked in please",
    "Can you squeeze me in tomorrow morning",
    "I need to set up a first-time visit",
    "Hey, just wondering if I can get an appointment this month",
    "I'm trying to book a slot for a filling",

    "Um, so I actually can't make my appointment, can we push it",
    "I need to change my appointment to a later date",
    "Something came up, can we move my visit",
    "Is it possible to switch my appointment to next Monday",
    "I have to shift my booking, is that okay",
    "Can we bump my appointment to the afternoon instead",
    "I need a new time, the current one doesn't work anymore",
    "Would it be possible to push my visit back a week",
    "Hi, I need to rearrange my upcoming appointment",
    "Can we swap my Thursday slot for Friday",

    "So I actually need to cancel, sorry about that",
    "I won't be needing my appointment anymore",
    "Please take me off the schedule for tomorrow",
    "I have to skip my appointment this week",
    "Can you cancel my booking, something came up",
    "I need to drop my appointment, sorry for the short notice",
    "Yeah I can't make it in, just cancel it please",
    "Take me off the list for Tuesday's appointment",
    "I won't be coming in, please cancel that",
    "Sorry, I need to call off my visit today",

    "Do you guys take walk-in patients",
    "What time do you close today",
    "Is parking available near your clinic",
    "Can you tell me more about your services",
    "Do you have a pediatric dentist on staff",
    "What's the address of your clinic exactly",
    "Do you offer emergency appointments",
    "How long does a typical checkup take",
    "Can I get a price list for your treatments",
    "Do you accept new patients right now",
]

new_labels = (
    ["book"] * 10 +
    ["reschedule"] * 10 +
    ["cancel"] * 10 +
    ["other"] * 10
)

texts = texts + new_texts
labels = labels + new_labels

print(f"Total examples now: {len(texts)}")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

clf = LogisticRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nDetailed report:\n", classification_report(y_test, predictions))

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

os.makedirs("model/saved", exist_ok=True)
joblib.dump(clf, "model/saved/intent_classifier.pkl")
joblib.dump(vectorizer, "model/saved/vectorizer.pkl")
print("\nModel saved to model/saved/")