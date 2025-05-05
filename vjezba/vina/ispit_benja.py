import pandas as pd
import matplotlib.pyplot as plt

# 0010
data=pd.read_csv("winequality-red.csv", sep=";")
""" 
#a) broj mjerenja
print(f"Broj mjerenja: {len(data)}")

#b) histogram distribucije alkoholne jakosti.
plt.hist(data["alcohol"], bins=20, color="blue")
plt.xlabel("Alkoholna jakost (%)")
plt.ylabel("Broj uzoraka")
plt.title("Distribucija alkoholne jakosti")
plt.grid(axis="y",linestyle="--")
plt.show()
# najvise rezultata ima oko 9-10% alkoholne jakosti

#c) broj vina sa kvalitetom manjom od 6 i broj sa vecom od 6
print(f"Broj vina sa kvalitetom manjom od 6: {len(data[data['quality'] < 6])}")
print(f"Broj vina sa kvalitetom vecom od 6: {len(data[data['quality'] > 6])}")

#d) Korelacija svih veličina u datasetu. Interpretirajte dobivene rezultate.
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Korelacija")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Korelacija između veličina u datasetu")
plt.show()
# Crvenije vrijednosti su pozitivne korelacije (povecavanjem jedne velicine povecava se i druga), 
# a plavije negativne (povecavanjem jedne velicine druga se smanjuje).
 """
# 0011
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Podijeliti na ulazne (X) i izlazne (y) podatke (kvaliteta)
X = data.drop(columns=["quality"]).values  # svi stupci osim "quality"
y = data["quality"].values      # samo stupac "quality"
y = (y >= 6).astype(int)        # ako je y veci od 6, onda je 1, inace 0

#podjela na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

""" 
#a) model logisticke regresije
model = LinearRegression()
model.fit(X_train, y_train) # ucenje modela
#parametri
print("a) Parametri linearnog regresijskog modela:")
print(f"Koeficijenti: {model.coef_}")
print(f"Presjek (intercept): {model.intercept_}")

#b) predikcija na testnom skupu, dijagram rasprsenja
y_pred = model.predict(X_test)
#dijagram rasprsenja
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Stvarne vrijednosti (y_test)")
plt.ylabel("Predviđene vrijednosti (y_pred)")
plt.title("Odnos stvarnih i predviđenih vrijednosti")
plt.grid(True)
plt.show()
#linearna regresija nije pogodna za binarnu klasifikaciju 
# jer daje vrijednosti izmedju 0 i 1, a ne samo 0 ili 1.

#c) vrednovanje modela RMSE, MAE, MAPE, R2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f"RMSE (Root Mean Squared Error): {rmse:.2f}") #koliko se u prosjeku predviđene vrijednosti razlikuju od stvarnih
print(f"MAE (Mean Absolute Error): {mae:.2f}") #prosječna apsolutna greška
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%") #prosječna apsolutna greška u postotcima
print(f"R² (R-squared): {r2:.2f}") #postotak varijacije u stvarnim vrijednostima koji model moze objasniti, 1 ukazuje na bolji model
 """
# 0012

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#a) izgradi neuronsku mrezu
model = Sequential([
    Dense(22, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    Dense(12, activation='relu'),                                  # Second hidden layer
    Dense(4, activation='relu'),                                   # Third hidden layer
    Dense(1, activation='sigmoid')                                 # Output layer
])

# Print model summary
model.summary()

#b) podesite proces treniranja
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary classification loss
    metrics=['accuracy']
)

#c) pokretanje ucenja
history = model.fit(
    X_train, y_train,
    epochs=800,
    batch_size=50,
    validation_split=0.1,  # Use 10% of the training data for validation
    verbose=1
)

#d) pohraniti na disk
model.save("wine_quality_model.h5")
print("Model saved to disk as 'wine_quality_model.h5'")

#e) evaluacija modela
# Load the model
loaded_model = load_model("wine_quality_model.h5")

# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#f) predikcija na testnom skupu
y_pred = (loaded_model.predict(X_test) > 0.5).astype("int32")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=["Quality < 6", "Quality >= 6"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)
