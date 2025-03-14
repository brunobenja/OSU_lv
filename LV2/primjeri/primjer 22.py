import numpy as np

a = np.array([3, 1, 5], float)
b = np.array([2, 4, 8], float)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a.min())
print(a.argmin())  # index najmanjeg elementa
print(a.max())
print(a.argmax())
print(a.sum())  # suma elemenata
print(a.mean())  # srednja vrijednost
print(np.mean(a))  # srednja vrijednost
print(np.max(a))  # najveći element
print(np.sum(a))  # suma elemenata
a.sort()  # sortira niz
print(a)
