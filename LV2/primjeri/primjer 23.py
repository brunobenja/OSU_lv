import numpy as np

np.random.seed(56)  # postavi seed generatora brojeva
rNumbers = np.random.rand(10)  # generiraj 10 slucajnih brojeva
print(rNumbers)  # ispis generiranih brojeva
print(round(rNumbers.mean(), 2))  # ispis srednje vrijednosti zaokruzene na 2 decimale
