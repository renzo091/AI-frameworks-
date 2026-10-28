import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./train_tickets.csv")

X = data["description"]
y = data["about"]

train_X, test_X, train_y, test_y = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    TfidfVectorizer(sublinear_tf=True),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(train_X, train_y)

predictions = model.predict(test_X)

print("Accuracy:", accuracy_score(test_y, predictions))
print(classification_report(test_y, predictions))