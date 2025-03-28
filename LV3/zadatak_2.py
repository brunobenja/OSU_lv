import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")
# a) histogram emisije CO2
plt.figure()
data["CO2 Emissions (g/km)"].plot.hist(bins=50)
plt.title("Histogram emisije CO2")
plt.xlabel("Emisija CO2 (g/km)")
plt.ylabel("Broj vozila")


# b) dijagram raspršenja između gradske potrošnje goriva i emisije CO2
plt.figure()
for fuel_type, group in data.groupby("Fuel Type"):
    plt.scatter(
        group["Fuel Consumption City (L/100km)"],
        group["CO2 Emissions (g/km)"],
        label=fuel_type,
    )
plt.legend(title="Fuel Type")
plt.title("Odnos gradske potrošnje goriva i emisije CO2")

# c) kutijasti dijagram razdiobe izvangradske potrošnje s obzirom na tip goriva
plt.figure()
data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")

# d) stupičasti dijagram broja vozila po tipu goriva
plt.figure()
data.groupby("Fuel Type").size().plot.bar()
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")

# e) stupčasti graf prosječne emisije CO2 po broju cilindara
plt.figure()
data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot.bar()
plt.title("Prosječna emisija CO2 po broju cilindara")
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna emisija CO2")


plt.show()
