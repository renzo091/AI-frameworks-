import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("./train_tickets.csv")

X = data["description"]
y = data["about"]

train_X, test_X, train_y, test_y = train_test_split(
    X,y,
    test_size= 0.2,
    random_state= 42
)

model = make_pipeline(
    TfidfVectorizer(
        binary=True,       
        use_idf=False,
        norm=None,
        stop_words=None,   
        min_df=2,          
        ngram_range=(1, 2),
    ),
    BernoulliNB(alpha=0.1),
)

model.fit(train_X,train_y)
pred = model.predict(test_y)

print("Accuracy:", accuracy_score(test_y, pred))
print(classification_report(test_y, pred))