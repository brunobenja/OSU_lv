import pandas as pd
import numpy as np

data = {
    "country": ["Belgium", "France", "Germany", "Netherlands", "United Kingdom"],
    "population": [11.3, 64.3, 81.3, 16.9, 64.9],
    "code": ["BE", "FR", "DE", "NL", "GB"],
}
countries = pd.DataFrame(
    data, columns=["country", "population", "code"]
)  # stvara DataFrame, data su podaci, columns je redosljed stupaca,
print(countries)  # ispisuje DataFrame countries
