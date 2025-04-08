import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from ucimlrepo import fetch_ucirepo

bank_marketing = fetch_ucirepo(id = 222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

print(X.head())
print(y.head())

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dt_classifier = DecisionTreeClassifier(random_state = 42, max_depth = 5)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

plt.figure(figsize = (20, 15))
plot_tree(dt_classifier,
          filled = True,
          feature_names = X.columns,
          class_names = ['No', 'Yes'],
          fontsize = 10,
          proportion = True,
          rounded = True,
          max_depth = 5)
plt.show()
