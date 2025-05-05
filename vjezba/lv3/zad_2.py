import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data= pd.read_csv("data_C02_emission.csv")
plt.figure()
data["CO2 Emissions (g/km)"].plot.hist(bins=50)
plt.title("Histogram emisije CO2")
plt.xlabel("Emisija CO2 (g/km)")
plt.ylabel("Broj vozila")

plt.figure()
for fuel_type, group in data.groupby("Fuel Type"):
    plt.scatter(
        group["Fuel Consumption City (L/100km)"],
        group["CO2 Emissions (g/km)"],
        label=fuel_type,
    )
plt.legend(title="Fuel Type")
plt.title("Odnos gradske potrošnje goriva i emisije CO2")

plt.figure()
data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")


plt.figure()
data.groupby("Fuel Type").size().plot.bar()
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")


plt.figure()
data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot.bar()
plt.title("Prosječna emisija CO2 po broju cilindara")
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna emisija CO2")
plt.show()