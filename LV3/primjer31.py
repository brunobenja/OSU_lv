import pandas as pd
import numpy as np

s1 = pd.Series(["crvenkapica", "vuk", "baka", "lovac"])
print(s1)
s2 = pd.Series(
    5.0, index=["a", "b", "c", "d", "e"], name="ime_objekta"
)  # stavlja 5.0 u svaki element, index je a,b,c,d,e, name je ime_objekta/ime serije
print(s2)
print(s2["b"])  # ispisuje element na indexu b

s3 = pd.Series(np.random.randn(5))  # generira 5 slučajnih brojeva
print(s3)  # ispisuje seriju
print(s3[3])  # ispisuje element na indexu 3
