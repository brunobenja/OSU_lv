import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv")
""" print(len(data))  # ispisuje broj redaka
print(data)  # ispisuje DataFrame data """

""" print("prvih 5 redaka:")
print(data.head(5))  # ispisuje prvih 5 redaka
print("zadnjih 5 redaka:")
print(data.tail(5))  # ispisuje zadnjih 5 redaka
print("info")
print(data.info())  # ispisuje informacije o DataFrameu
print("describe")
print(data.describe())  # ispisuje statističke podatke o DataFrameu
print("max")
print(data.max())  # ispisuje maksimalne vrijednosti
print("min")
print(data.min())  # ispisuje minimalne vrijednosti """
""" 
print(data["Cylinders"])  # ispisuje stupac Cylinders
print(data.Cylinders)  # ispisuje stupac Cylinders

print(data[["Model", "Cylinders"]])  # ispisuje stupce Model i Cylinders

print(data.iloc[2:6, 2:7])  # ispisuje redove od 2 do 6 i stupce od 2 do 7
print(data.iloc[:, 2:5])  # ispisuje sve retke i stupce od 2 do 5
print(data.iloc[:, [0, 4, 7]])  # ispisuje sve retke i stupce 0,4,7 """

print(data.Cylinders > 6)  # ispisuje True ili False za svaki redak
print(data[data.Cylinders > 6])  # ispisuje retke za koje je Cylinders > 6
# ispisuje modele za koje je Cylinders = 4 i Engine Size (L) > 2.4
print(data[(data["Cylinders"] == 4) & (data["Engine Size (L)"] > 2.4)].Model)

# dodavanje novih stupaca
data["jedinice"] = np.ones(len(data))
data["large"] = data["Cylinders"] > 10
