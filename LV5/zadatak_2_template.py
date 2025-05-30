import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

labels = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}


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
            edgecolor="w",
            label=labels[cl],
        )


# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=["sex"])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df["species"].replace({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}, inplace=True)

print(df.info())

# izlazna velicina: species
output_variable = ["species"]

# ulazne velicine: bill length, flipper_length
input_variables = ["bill_length_mm", "flipper_length_mm"]

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# a) stupcasti dijagram broja primjera za svaku klasu
train_counts = np.unique(y_train, return_counts=True)
test_counts = np.unique(y_test, return_counts=True)
plt.figure(figsize=(8, 6))
plt.bar(
    train_counts[0], train_counts[1], color="blue", alpha=0.7, label="Training Data"
)
plt.bar(test_counts[0], test_counts[1], color="orange", alpha=0.7, label="Testing Data")
plt.xticks(train_counts[0], labels.values())
plt.xlabel("Penguin Species")
plt.ylabel("Number of Examples")
plt.title("Class Distribution in Training and Testing Sets")
plt.legend()
plt.show()

# b) model logisticke regresije na podacima za ucenje
from sklearn.linear_model import LogisticRegression

# Train logistic regression model
logistic_model = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=200
)
logistic_model.fit(X_train, y_train.ravel())

# C)u atributima izgradenog modela pronaci parametre modela
# razlika u odnosu na binarni klasif problem iz 1zad
print("Intercepts (θ₀ for each class):", logistic_model.intercept_)
print("Coefficients (θ₁, θ₂ for each class):", logistic_model.coef_)

# d) funkcijom plot_decision_regions i predati joj podatke za učenje i izgradeni
# model log regresije. komentiraj dobivene rezultate
plot_decision_regions(X_train, y_train.ravel(), classifier=logistic_model)
plt.xlabel("Bill Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.title("Decision Regions for Training Data")
plt.legend()
plt.show()

# E)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# Predict on test data
y_pred = logistic_model.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=labels.values()
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels.values()))


# f)
# Add more features to the model
input_variables = [
    "bill_length_mm",
    "flipper_length_mm",
    "bill_depth_mm",
    "body_mass_g",
]
X = df[input_variables].to_numpy()

# Split the data again
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Train the logistic regression model with additional features
logistic_model.fit(X_train, y_train.ravel())

# Predict and evaluate
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Additional Features: {accuracy:.2f}")
print("\nClassification Report with Additional Features:")
print(classification_report(y_test, y_pred, target_names=labels.values()))
