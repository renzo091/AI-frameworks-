from sklearn.linear_model import LogisticRegression
from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV as random

digits = datasets.load_digits()
x = digits.data
y = digits.target

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=2000)
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
acc = accuracy_score(test_y, y_pred)
print("accuracy:", acc)

print(x)
print(y)
"""

lr = LogisticRegression(max_iter=2000)

param_grid = [
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10, 100]
    },
    {
        'solver': ['saga'],
        'penalty': ['l1', 'elasticnet'],
        'C': [0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0.2, 0.5, 0.8]  
    }
]

grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy' 
)

grid_search.fit(train_x, train_y)
print(grid_search.cv_results_)
print(grid_search.best_params_)
"""