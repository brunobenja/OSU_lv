import pandas as pd
import numpy as np

data = pd.read_csv("data_C02_emission.csv")  # učitava podatke iz datoteke
new_data = data.groupby("Cylinders")
print(new_data.count())  # ispisuje broj redaka za svaku grupu
print(new_data.size())  # ispisuje veličinu svake grupe
print(new_data.sum())  # ispisuje zbroj svake grupe
""" print(new_data.mean())  # ispisuje prosjek svake grupe """  # izbacuje grešku jer stupac "Model" nije numerički
