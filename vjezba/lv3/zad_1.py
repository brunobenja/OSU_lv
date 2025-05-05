import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv")
print(f"a)\nBroj mjerenja: {len(data)}")  # broj mjerenja,-1 jer je prvi redak header
print(f"Tipovi podataka:")
print(data.dtypes)
print(
    f"Izostale vrijednosti:\n",
    data.isnull(),
    "\nDuplicirane vrijednosti:\n",
    data.duplicated(),
)
data = data.dropna()  # izbacuje retke s nedostajućim vrijednostima
data = data.drop_duplicates()  # izbacuje duplikate
categorical_columns = data.select_dtypes(include=["object"]).columns
# ⬆️ odabire stupce s kategoričkim podacima, dtypes su stupci, include object znači da uzima sve stupce koji sadrze stringove
for column in categorical_columns:  # za svaki stupac s kategoričkim podacima
    data[column] = data[column].astype("category")  # pretvara ih u kategoričke podatke

print(f"\nb)")
print(f"Tri najveća potrosaca:")
print(
    data.nlargest(3, "Fuel Consumption City (L/100km)")[
        ["Make", "Model", "Fuel Consumption City (L/100km)"]
    ]
)
print(f"\nTri najmanja potrosaca:")
print(
    data.nsmallest(3, "Fuel Consumption City (L/100km)")[
        ["Make", "Model", "Fuel Consumption City (L/100km)"]
    ]
)

print(f"\nc)")
print(f"Broj vozila sa veličinom motora između 2.5 i 3.5L:")
print(len(data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]))
print(f"\nProsječna CO2 emisija:")
print(data["CO2 Emissions (g/km)"].mean().__round__(2))

print(f"\nd)")
print(f"Broj mjerenja marke Audi:")
print(len(data[data["Make"] == "Audi"]))
print(f"\nProsječna emisija CO2 vozila marke Audi sa 4 cilindra:")
print(
    data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)]["CO2 Emissions (g/km)"]
    .mean()
    .round(2)
)

print(f"\ne)")
print(f"Broj vozila sa parnim brojem cilindara većim od 2:")
print(len(data[data["Cylinders"] % 2 == 0 & (data["Cylinders"] > 2)]))
print(f"\nProsječna emisija CO2 s obzirom na broj cilindara:")
print(data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().round(2))

print(f"\nf)")
print(f"Prosječna gradska potrosnja vozila na dizel:")
print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean().round(2))
print(f"\nProsječna gradska potrosnja vozila na benzin:")
print(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean().round(2))
print(f"\nMedijalne vrijednosti potrošnje:")
print(f"\nDizel:")
print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median())
print(f"Benzin:")
print(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median())

print(f"\ng)")
print(f"Vozilo sa dizelskim motorom sa 4 cilindra i najvećom potrošnjom:")
print(
    data[(data["Fuel Type"] == "D") & (data["Cylinders"] == 4)].nlargest(
        1, "Fuel Consumption City (L/100km)"
    )["Model"]
)

print(f"\nh)")
print(f"Broj vozila sa ručnim mjenjačem:")
print(len(data[data["Transmission"] == "M"]))

print(f"\ni)")
print(f"Korelacija između numeričkih veličina:")
print(data.select_dtypes(include=[np.number]).corr())
# select_dtypes uzima stupce s određenim tipom podataka, np.number uzima numeričke podatke
# 1 => kako se jedna veličina povećava, povećava se i druga -> savršena pozitivna korelacija
# 0 => nema povezanosti
# -1 => kako se jedna veličina povećava, druga se smanjuje  -> savršena negativna korelacija
