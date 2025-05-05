# Učitavanje potrebnih biblioteka
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Učitavanje podataka
data = pd.read_csv("Social_Network_Ads.csv")

# Odabir ulaznih i izlaznih veličina
X = data[["Age", "EstimatedSalary"]].values
y = data["Purchased"].values

# Podjela podataka na skup za učenje i testiranje (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definicija KNN modela
knn = KNeighborsClassifier()

# Definicija raspona hiperparametra K za pretragu
param_grid = {"n_neighbors": np.arange(1, 51)}

# Unakrsna validacija za odabir optimalnog K
grid_search = GridSearchCV(
    estimator=knn, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Prikaz optimalne vrijednosti K i pripadajuće točnosti
optimal_k = grid_search.best_params_["n_neighbors"]
best_score = grid_search.best_score_

print(f"Optimalna vrijednost K: {optimal_k}")
print(f"Najbolja točnost tijekom unakrsne validacije: {best_score:.2f}")

# Evaluacija modela s optimalnim K na testnom skupu
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
test_accuracy = knn_optimal.score(X_test, y_test)

print(f"Točnost na testnom skupu s optimalnim K={optimal_k}: {test_accuracy:.2f}")

# Train KNN with optimal K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train[:, :2], y_train)

# Calculate accuracy
train_accuracy_optimal = accuracy_score(y_train, knn_optimal.predict(X_train[:, :2]))
test_accuracy_optimal = accuracy_score(y_test, knn_optimal.predict(X_test[:, :2]))

print(f"Training Accuracy (Optimal K={optimal_k}): {train_accuracy_optimal:.2f}")
print(f"Testing Accuracy (Optimal K={optimal_k}): {test_accuracy_optimal:.2f}")
