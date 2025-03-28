from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Učitaj ugrađeni podatkovni skup
X, y = datasets.load_diabetes(return_X_y=True)
# Podijeli skup na podatkovni skup za učenje i podatkovni skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)


ohe = OneHotEncoder()
# Define a sample DataFrame with a "Fuel Type" column
data = pd.DataFrame({"Fuel Type": ["Petrol", "Diesel", "Electric", "Petrol"]})

# Perform one-hot encoding on the "Fuel Type" column
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
