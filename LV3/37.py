import pandas as pd

data = pd.read_csv("data_C02_emission.csv")
# Provjera koliko je izostalih vrijednosti po svakom stupcu DataFrame-a
print(data.isnull().sum())
# Brisanje redova gdje barem vrijednost jedne veličine nedostaje
data = data.dropna(axis=0)
# Brisanje stupaca gdje barem jedna vrijednost nedostaje
data = data.dropna(axis=1)
# Brisanje dupliciranih redova
data = data.drop_duplicates()
# Kada se obrišu pojedini redovi potrebno je resetirati indekse retka
data = data.reset_index(drop=True)
