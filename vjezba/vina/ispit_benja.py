import pandas as pd
import matplotlib.pyplot as plt

# 0010
data=pd.read_csv("winequality-red.csv", sep=";")

#a) broj mjerenja
print(f"Broj mjerenja: {len(data)}")

#b) histogram distribucije alkoholne jakosti.
plt.hist(data["alcohol"], bins=20, color="blue")
plt.xlabel("Alkoholna jakost (%)")
plt.ylabel("Broj uzoraka")
plt.title("Distribucija alkoholne jakosti")
plt.grid(axis="y",linestyle="--")
plt.show()
# najvise rezultata ima oko 9-10% alkoholne jakosti

#c) broj vina sa kvalitetom manjom od 6 i broj sa vecom od 6
print(f"Broj vina sa kvalitetom manjom od 6: {len(data[data['quality'] < 6])}")
print(f"Broj vina sa kvalitetom vecom od 6: {len(data[data['quality'] > 6])}")

#d) Korelacija svih veličina u datasetu. Interpretirajte dobivene rezultate.
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Korelacija")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Korelacija između veličina u datasetu")
plt.show()
# Crvenije vrijednosti su pozitivne korelacije (povecavanjem jedne velicine povecava se i druga), 
# a plavije negativne (povecavanjem jedne velicine druga se smanjuje).


