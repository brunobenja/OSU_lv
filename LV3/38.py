import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")
plt.figure()
data["Fuel Consumption City (L/100km)"].plot(kind="hist", bins=20)
plt.figure()
data["Fuel Consumption City (L/100km)"].plot(kind="box")
plt.show()
