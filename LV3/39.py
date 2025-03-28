import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")
grouped = data.groupby("Cylinders")  # grupira podatke prema stupcu "Cylinders"
grouped.boxplot(
    column=["CO2 Emissions (g/km)"]
)  # crta boxplot za stupac "CO2 Emissions (g/km)" za svaku grupu
data.boxplot(
    column=["CO2 Emissions (g/km)"], by="Cylinders"
)  # crta boxplot za stupac "CO2 Emissions (g/km)" grupirano po "Cylinders"
plt.show()
