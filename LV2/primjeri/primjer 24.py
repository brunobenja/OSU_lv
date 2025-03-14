import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 6, num=30)  # generira niz brojeva od 0 do 6 sa 30 elemenata
y = np.sin(x)  # generira niz vrijednosti funkcije sinusa
plt.plot(x, y, "b", linewidth=1, marker=".", markersize=5)  # crta graf funkcije sinusa
plt.axis([0, 6, -2, 2])  # postavlja granice osi x 0-6, y -2-2
plt.xlabel("x")  # postavlja oznaku x osi
plt.ylabel("vrijednosti funkcije")  # postavlja oznaku y osi
plt.title("sinus funkcija")  # postavlja naslov grafu
plt.show()  # prikazuje graf
