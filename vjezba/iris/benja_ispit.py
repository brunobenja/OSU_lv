from sklearn import datasets
iris = datasets.load_iris()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 007
#a) odnos duljine latica i sacisa virginice sa scatter zelenom bojom
virginica = iris.data[iris.target == 0]
duljina_latica_virginica = virginica[:, 2]
duljina_casice_virginica = virginica[:, 3]
""" 
plt.scatter(duljina_latica_virginica, duljina_casice_virginica, color="green",label="virginica")
plt.xlabel("Duljina latice (cm)")
plt.ylabel("Duljina casice (cm)")
plt.title("Odnos duljine casica i latica")

setosa = iris.data[iris.target == 0]
duljina_latica_setosa = setosa[:, 2]
duljina_casice_setosa = setosa[:, 0]
plt.scatter(duljina_latica_setosa, duljina_casice_setosa, color="gray",label="setosa")
plt.legend()
plt.grid(True)
plt.show()

#b) stupcasti dijagram najvece sirine casice za sve tri klase, naziv dijagrama i osi
max_sepal_width_setosa = iris.data[iris.target == 0][:, 1].max()
max_sepal_width_versicolor = iris.data[iris.target == 1][:, 1].max()
max_sepal_width_virginica = iris.data[iris.target == 2][:, 1].max()

# Create a bar chart
classes = ["Setosa", "Versicolor", "Virginica"]
max_values = [max_sepal_width_setosa, max_sepal_width_versicolor, max_sepal_width_virginica]

plt.bar(classes, max_values, color=["gray", "blue", "green"])

# Add labels, title, and grid
plt.xlabel("Klase cvijeta")
plt.ylabel("Najveća širina čašice (cm)")
plt.title("Najveća širina čašice za sve tri klase cvijeta")
plt.show()

#c) 
sepal_width_setosa = iris.data[iris.target == 0][:, 1]

# Calculate the average sepal width for Setosa
average_sepal_width_setosa = sepal_width_setosa.mean()

# Count how many Setosa samples have a sepal width greater than the average
count_greater_than_average = np.sum(sepal_width_setosa > average_sepal_width_setosa)

# Print the results
print(f"Prosječna širina čašice za Setosa klasu: {average_sepal_width_setosa:.2f}")
print(f"Broj jedinki klase Setosa s većom širinom čašice od prosječne: {count_greater_than_average}")
 """

# 008
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

#a) optimalni broj klastera K za K-means algoritam

X = iris.data  # Use all features for clustering
y_true = iris.target  # True class labels for comparison

inertia = []
K_range = range(1, 11)  # Test K values from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

#b) graficki lakat metoda
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Broj klastera (K)")
plt.ylabel("Inercija")
plt.title("Lakat metoda za određivanje optimalnog K")
plt.grid(True)
plt.show()

#c) K-means klasifikacija s K=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

colors = ['green', 'yellow', 'orange']
for i in range(optimal_k):
    cluster_names = ["Setosa", "Versicolor", "Virginica"]
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=50, color=colors[i], label=cluster_names[i])

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, color='red', marker='x', label="Centroidi")

# Add labels, title, and legend
plt.xlabel("Duljina čašice (cm)")
plt.ylabel("Širina čašice (cm)")
plt.title("K-Means klasteri za Iris dataset")
plt.legend()
plt.show()

#e) Usporedite dobivene klase sa stvarnim vrijednostima. tocnost klasifikacije
from scipy.stats import mode

labels = np.zeros_like(y_kmeans)
for i in range(optimal_k):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask])[0]

# Calculate accuracy
accuracy = accuracy_score(y_true, labels)
print(f"Točnost klasifikacije: {accuracy:.2f}")
