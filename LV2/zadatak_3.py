import numpy as np
import matplotlib.pyplot as plt

img_org = plt.imread("road.jpg")
img_org = img_org[:, :, 0].copy()
print(img_org.shape)
print(img_org.dtype)
plt.figure()
plt.imshow(img_org, cmap="gray")
plt.show()

# a) Posvijetljena slika
brightness = 100
img_bright = img_org + brightness
img_bright[img_bright < brightness] = 255
img_bright = np.clip(img_bright, 0, 255)
plt.figure()
plt.imshow(img_bright, cmap="gray")
plt.title("a) Posvijetljena slika")
plt.show()

# b) druga četvrtina slike po širini
img_cropped = img_org[:, img_org.shape[1] // 4 : img_org.shape[1] // 2]
plt.figure()
plt.imshow(img_cropped, cmap="gray")
plt.title("b) druga četvrtina slike po širini")
plt.show()

# c) zarotirana slika za 90 stupnjeva
img_rotated = np.rot90(img_org, k=3)
plt.figure()
plt.imshow(img_rotated, cmap="gray")
plt.title("c) zarotirana slika za 90 stupnjeva")
plt.show()

# d) Zrcaljena slika
img_flipped = np.fliplr(img_org)
plt.figure()
plt.imshow(img_flipped, cmap="gray")
plt.title("d) Zrcaljena slika")
plt.show()
