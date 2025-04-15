import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")  # ucitavanje slike

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

###################################
###################################
###################################
# 1) Broj različitih boja
broj_boja = np.unique(img_array, axis=0)
print(f"Broj različitih boja: {len(broj_boja)}")

# 2)Primijenite algoritam K srednjih vrijednosti
# koji ce pronaci grupe u RGB vrijednostima
# elemenata originalne slike
K = 2  # Broj grupa (boja)
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(img_array)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 3)  Vrijednost svakog elementa slike
# originalne slike zamijeni s njemu
# pripadajucim centrom
# Zamjena vrijednosti svakog piksela s pripadajućim centrom
img_array_aprox = centroids[labels]
img_aprox = np.reshape(img_array_aprox, (w, h, d))

# 4) Usporedite dobivenu sliku s originalnom.
# Mijenjate broj grupa K .
# Komentirajte dobivene rezultate

plt.figure()
plt.title(f"Kvantizirana slika (K={K})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()


# 6) Grafički prikažite ovisnost J o broju grupa K
inertia_values = []
K_values = range(1, 11)  # Testiramo za K od 1 do 10

for K in K_values:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(img_array)
    inertia_values.append(kmeans.inertia_)

# Prikaz grafa ovisnosti J o K
plt.figure()
plt.plot(K_values, inertia_values, marker="o")
plt.xlabel("Broj grupa K")
plt.ylabel("Vrijednost J (inertia)")
plt.title("Ovisnost J o broju grupa K")
plt.grid()
plt.show()

# 7) Prikaz elemenata slike koji pripadaju jednoj grupi kao binarnu sliku
for group in range(K):
    # Stvaranje binarne maske za trenutnu grupu
    binary_mask = (labels == group).astype(np.float64)
    binary_image = np.reshape(binary_mask, (w, h))

    # Prikaz binarne slike
    plt.figure()
    plt.title(f"Grupa {group + 1} - Binarna slika")
    plt.imshow(binary_image, cmap="gray")
    plt.tight_layout()
    plt.show()
