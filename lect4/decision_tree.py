import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./train_tickets.csv")

X = data["description"]
Y = data["about"]

train_X, test_X,train_Y,test_Y = train_test_split(
    X,Y,
    test_size = 0.2,
    random_state=42
      )

model = make_pipeline(
    TfidfVectorizer(max_features=5000, min_df=2),
    DecisionTreeClassifier(random_state=42)
)

model.fit(train_X, train_Y)

predictions = model.predict(test_X)

print("Accuracy:", accuracy_score(test_Y, predictions))
print(classification_report(test_Y, predictions))



