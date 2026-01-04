import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('punkt_tab')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

with open("dataset.json") as f:
    data = json.load(f)

sentences = []
labels = []

for intent, patterns in data.items():
    for p in patterns:
        sentences.append(preprocess(p))
        labels.append(intent)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# === EVALUASI ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi Model:", accuracy * 100, "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model berhasil di-train dan disimpan!")
