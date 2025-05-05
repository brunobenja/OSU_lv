import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("titanic.csv")

num_women=data[data["Sex"]=="female"].shape[0] #shape[0] daje broj redova
print(f"a) Broj zena u skupu podataka: {num_women}")

percentage_death=data[data["Survived"]==0].shape[0]/data.shape[0]*100
print(f"b) Postotak umrlih: {percentage_death:.2f}%")

survival_by_gender=data.groupby("Sex")["Survived"].mean()*100
print(f"c) Postotak prezivjelih po spolu:\n{survival_by_gender}")
# Plotting the survival rates
plt.bar(["Žene","Muškarci"],survival_by_gender,color=["yellow","green"])
plt.title("Postotak prezivjelih po spolu")
plt.xlabel("Spol")
plt.ylabel("Postotak prezivjelih (%)")
plt.ylim(0, 100)
plt.show()

print(f"d)")
# Calculate the average age of surviving women
avg_age_surviving_women = data[(data["Survived"] == 1) & (data["Sex"] == "female")]["Age"].mean()
# Calculate the average age of surviving men
avg_age_surviving_men = data[(data["Survived"] == 1) & (data["Sex"] == "male")]["Age"].mean()
print(f"Prosječna dob preživjelih žena: {avg_age_surviving_women:.2f}")
print(f"Prosječna dob preživjelih muškaraca: {avg_age_surviving_men:.2f}")

print(f"e)")

oldest_survivor = data[data["Survived"] == 1].groupby("Pclass")["Age"].max()
print(f"Najstariji prezivjeli putnik po klasi: {oldest_survivor}")


# ZAD 005

import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
        )

# Drop rows with missing values
data = data.dropna()

# Encode categorical variables
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Select input (X) and output (y) variables
X = data[["Pclass", "Sex", "Fare", "Embarked"]]
y = data["Survived"]

# Split the data into training and testing sets (70:30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Train KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:, :2], y_train)  # Use only the first two features for visualization

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(["red", "blue"]))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=ListedColormap(["red", "blue"]))
    plt.xlabel("Pclass (scaled)")
    plt.ylabel("Sex (scaled)")
    plt.title("Decision Boundary (K=5)")
    plt.show()

# b) točnost klasifikacije
plot_decision_boundary(X_train[:, :2], y_train, knn)

train_accuracy = accuracy_score(y_train, knn.predict(X_train[:, :2]))
test_accuracy = accuracy_score(y_test, knn.predict(X_test[:, :2]))

print(f"Training Accuracy (K=5): {train_accuracy:.2f}")
print(f"Testing Accuracy (K=5): {test_accuracy:.2f}")

#c) unakrsna validacija
param_grid = {"n_neighbors": np.arange(1, 21)}
grid_search = GridSearchCV(
    estimator=knn, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

optimal_k = grid_search.best_params_["n_neighbors"]
best_score = grid_search.best_score_

print(f"Optimalna vrijednost K: {optimal_k}")
print(f"Najbolja točnost tijekom unakrsne validacije: {best_score:.2f}")

# Train KNN with the optimal K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train[:, :2], y_train)

# Calculate accuracy for the optimal K
train_accuracy_optimal = accuracy_score(y_train, knn_optimal.predict(X_train[:, :2]))
test_accuracy_optimal = accuracy_score(y_test, knn_optimal.predict(X_test[:, :2]))

# Print results for optimal K
print(f"Training Accuracy (Optimal K={optimal_k}): {train_accuracy_optimal:.2f}")
print(f"Testing Accuracy (Optimal K={optimal_k}): {test_accuracy_optimal:.2f}")

# Compare with K=5
print("\nComparison with K=5:")
print(f"Training Accuracy (K=5): {train_accuracy:.2f}")
print(f"Testing Accuracy (K=5): {test_accuracy:.2f}")

# 006

from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#a) 
# Build the neural network
model = Sequential([
    Dense(12, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(8, activation="relu"),  # Second hidden layer
    Dense(4, activation="relu"),  # Third hidden layer
    Dense(1, activation="sigmoid")  # Output layer
])

# Print model summary
model.summary()

#b) Podesite proces treniranja
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

#c) Učenje mreze sa 100 epoha i batch size 5
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=5,
    validation_split=0.1,
    verbose=1
)

#d)  Save the model
model.save("titanic_model.h5")
print("Model saved to disk.")

#e) Evaluacija na testnom skupu
loaded_model = tf.keras.models.load_model("titanic_model.h5")

# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


#f) Predikcija na skupu za testiranje
# matrca zabune

from sklearn.metrics import ConfusionMatrixDisplay
# Make predictions
y_pred = (loaded_model.predict(X_test) > 0.5).astype("int32")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
ConfusionMatrixDisplay(conf_matrix, display_labels=["Not Survived", "Survived"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()