import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50))
top = np.hstack((black, white))  # horizontal stack
bottom = np.hstack((white, black))  # horizontal stack
pattern = np.vstack((top, bottom))  # vertical stack dva reda
plt.imshow(pattern, cmap="grey", vmin=0, vmax=1)
# cmap gray mijenja u crno bijelo, default je ljubicasto zuto
plt.show()
