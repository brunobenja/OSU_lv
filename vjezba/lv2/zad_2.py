import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
spol = data[:, 0]
visina = data[:, 1]
masa = data[:, 2]
numberOfPeople = len(data)
print(f"Broj mjerenja:{numberOfPeople}")

plt.figure()
plt.scatter(visina, masa, marker="o", color="b")
plt.xlabel("visina")
plt.ylabel("masa")
plt.title("b) Odnos visine i mase")


visina_50 = visina[::50]
masa_50 = masa[::50]
plt.figure()
plt.title("c) Odnos visine i mase svake 50e osobe")
plt.scatter(visina[::50], masa[::50], marker="o", color="g")
plt.show()


min_visina = np.min(visina)
max_visina = np.max(visina)
srednja_visina = np.mean(visina)
print(
    f"Minimalna visina: {min_visina}, maksimalna visina: {max_visina}, srednja visina: {srednja_visina}"
)


ind_m = spol == 1
visina_50m = visina[ind_m][:]
print(
    f"minimalna visina_50m: {np.min(visina_50m)},max visina: {np.max(visina_50m)}, srednja visina: {np.mean(visina_50m)}"
)
ind_z = spol == 0
visina_50z = visina[ind_z][:]
print(
    f"minimalna visina_50z: {np.min(visina_50z)},max visina: {np.max(visina_50z)}, srednja visina: {np.mean(visina_50z)}"
)
