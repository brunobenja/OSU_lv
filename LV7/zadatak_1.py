import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


# generiranje umjetnih podatkovnih primjera
def generate_data(n_samples, flagc):
    # n_samples = zeljeni br primjera, flagc=nacin generiranja
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
            random_state=random_state,
        )
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    # 2 grupe
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=0.05)

    else:
        X = []

    return X


# generiranje podatkovnih primjera
X = generate_data(500, 2)  # 1=defaultni, 2=rotirani, 3=4 grupe, 4=circle, 5=moon

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("podatkovni primjeri")
plt.show()


#########################
#########################
#########################
# 1) Postoji 3 grupe u generiranim podacima
# u line 47 mijenjam način generiranja podataka


# 2) Primijenite metodu K srednjih vrijednosti te
# ponovo prikažite primjere
def apply_kmeans(X, n_clusters):
    # Primjena K-Means algoritma
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Prikaz rezultata grupiranja
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", edgecolor="k", alpha=0.7)
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c="red",
        marker="X",
        label="Centroids",
    )
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"K-Means (K={n_clusters})")
    plt.legend()
    plt.show()


# Testiranje s različitim brojem grupa K
for k in [2, 3, 4, 5]:
    print(f"Primjena K-Means s K={k}")
    apply_kmeans(X, n_clusters=k)


# mijenjanjem broja K dobijamo različit broj grupa,
# ako je k premalen necemo dobiti dovoljnu podjelu,
# ako je prevelik dobit cemo prevelik broj grupa


# 3) Mijenjajte nacin definiranja umjetnih primjera te
# promatrajte rezultate grupiranja podataka
# (koristite optimalni broj grupa).
# Kako komentirate dobivene rezultate?

# flagc=1/2 => dobro radi kada je K=broj grupa
# flagc=3 => teze radi sa outliersima,
# pretpostavlja da su sve grupe jednake velicine
# flagc=4/5 => ne radi dobro jer grupe nisu sferične
