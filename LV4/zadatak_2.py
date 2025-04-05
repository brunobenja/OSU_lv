import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, max_error

data = pd.read_csv("data_C02_emission.csv")

encoder = OneHotEncoder(
    sparse_output=False, drop="first"
)  # drop="first" uklanja prvu kategoriju da bi se izbjegla multikolinearnost
fuel_type_encoded = encoder.fit_transform(data[["Fuel Type"]])  # kodiranje tipa goriva

X_numericalna = data[
    [
        "Engine Size (L)",
        "Cylinders",
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
        "Fuel Consumption Comb (L/100km)",
    ]
].values  # odabir numerickih velicina
X = np.hstack(
    (X_numericalna, fuel_type_encoded)
)  # spajanje numerickih i kategorickih varijabli
y = data["CO2 Emissions (g/km)"].values  # odabir ciljne varijable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)  # podjela na test i trening skupove
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)  # treniranje modela

print("Intercept (β₀):", linearModel.intercept_)
print("Coefficients (β₁, β₂, ...):", linearModel.coef_)

y_pred = linearModel.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Maximum Error:", max_error(y_test, y_pred))

# Identify the vehicle with the maximum error
max_error_index = np.argmax(np.abs(y_test - y_pred))
print("\nVehicle with Maximum Error:")
print("Actual CO2 Emissions:", y_test[max_error_index])
print("Predicted CO2 Emissions:", y_pred[max_error_index])
print("Features of the Vehicle:", X_test[max_error_index])

# Scatter plot: Actual vs Predicted values
plt.figure()
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Ideal Fit",
)
plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Actual vs Predicted CO2 Emissions")
plt.legend()
plt.show()
