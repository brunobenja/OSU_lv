import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, r2_score

# a)
data = pd.read_csv("data_C02_emission.csv")
X = data[
    [
        "Engine Size (L)",
        "Cylinders",
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
        "Fuel Consumption Comb (L/100km)",
        "Fuel Consumption Comb (mpg)",
    ]
].values  # odabir numerickih velicina
y = data["CO2 Emissions (g/km)"].values  # odabir ciljne varijable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)  # podjela na test i trening skupove

""" 
# b)
plt.figure()
plt.scatter(
    X_train[:, 1], y_train, color="blue", label="Trening skup"
)  # prikaz trening skupa
plt.scatter(
    X_test[:, 1], y_test, color="red", label="Testni skup"
)  # prikaz testnog skupa
plt.xlabel("Cyliders")  # označavanje x osi
plt.ylabel("Emisija CO2 (g/km)")  # označavanje y osi
plt.legend()
"""
# c)
scaler = StandardScaler()  # standardizacija podataka
X_train_scaled = scaler.fit_transform(X_train)  # standardizacija trening skupa
X_test_scaled = scaler.transform(X_test)  # standardizacija testnog skupa
""" 
plt.figure()
plt.hist(X_train[:, 0], color="blue", label="Prije skaliranja")
plt.xlabel("Engine Size (L)")
plt.ylabel("Frequency")
plt.title("Histogram Engine Size (L) (Prije skaliranja)")
plt.legend()

plt.figure()
plt.hist(X_train_scaled[:, 0], color="red", label="Nakon skaliranja")
plt.xlabel("Engine Size (L) (Skalirano)")
plt.ylabel("Frequency")
plt.title("Histogram Engine Size (L) (Nakon skaliranja)")
plt.legend()
plt.show() 
"""

# d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_scaled, y_train)
print("Intercept (β₀):", linearModel.intercept_)
print("Coefficients (β₁, β₂, ...):", linearModel.coef_)

print("\nLinear Regression Equation:")
print(
    f"y = {linearModel.intercept_:.2f} + {linearModel.coef_[0]:.2f} * x1 + {linearModel.coef_[1]:.2f} * x2 + {linearModel.coef_[2]:.2f} * x3+ {linearModel.coef_[3]:.2f} * x4 + {linearModel.coef_[4]:.2f} * x5"
)


# e)
y_pred = linearModel.predict(X_test_scaled)  # predikcija na testnom skupu
plt.figure()
plt.scatter(y_test, y_pred, color="blue")  # prikaz stvarnih i predikcija
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
)
# crvena linija prikazuje predikciju
plt.xlabel("Stvarna vrijednost CO2 emisija (g/km)")
plt.ylabel("Predikcija CO2 emisija (g/km)")
plt.title("Stvarna vs Predikcija CO2 emisija")
plt.legend()
plt.show()

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# f)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# g)
# vise
X = data[
    [
        "Engine Size (L)",
        "Cylinders",
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
    ]
].values
# manje
X = data[["Engine Size (L)", "Cylinders"]].values
