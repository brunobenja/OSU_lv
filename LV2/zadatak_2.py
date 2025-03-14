import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
spol = data[:, 0]
visina = data[:, 1]
masa = data[:, 2]

number = len(data)
print(f"a) Mjerenja su izvršena na: {number} osoba.")


plt.scatter(visina, masa, marker="o", color="red")
plt.xlabel("visina")
plt.ylabel("masa")
plt.title("b) odnos visine i mase")
plt.show()

plt.title("c) Odnos visine i mase svake 50e osobe")
visina_50 = visina[::50]
masa_50 = masa[::50]
plt.scatter(visina_50, masa_50, marker="o", color="green")
plt.show()

print(
    f"d) Minimalna visina: {np.min(visina_50)}, maksimalna visina: {np.max(visina_50)}, srednja visina: {np.mean(visina_50)}"
)

print(f"e)")
ind_m = spol == 1
visina_50m = visina[ind_m][::50]
masa_50m = masa[ind_m][::50]
print(
    f"Minimalna visina muškaraca: {np.min(visina_50m)}, maksimalna visina muškaraca: {np.max(visina_50m)}, srednja visina muškaraca: {np.mean(visina_50m)}"
)

ind_z = spol == 0
visina_50z = visina[ind_z][::50]
masa_50z = masa[ind_z][::50]
print(
    f"Minimalna visina žena: {np.min(visina_50z)}, maksimalna visina žena: {np.max(visina_50z)}, srednja visina žena: {np.mean(visina_50z)}"
)
