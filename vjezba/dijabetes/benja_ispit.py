import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pima-indians-diabetes.csv')

#a) broj osoba
print(f'Mjerenja su izvršena na {len(data)} osoba.')

#b) izostale (nema izostalih pa sam stavio koje su 0) i duplicirane vrijednosti u stupcima dobi i bmi
#za bas izostale: data_cleaned = data.dropna(subset=["Age", "BMI"])
data_cleaned = data[(data["Age"] != 0) & (data["BMI"] != 0)]
data_cleaned=data_cleaned.drop_duplicates(subset=["Age","BMI"])
""" 
print(f"Broj preostalih mjerenja :{len(data_cleaned)}")

#c) scatter dijagram omjer dobi i bmi
plt.scatter(data_cleaned['Age'], data_cleaned['BMI'])
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Scatter plot of Age vs BMI')
plt.show()

#d) min,max, mean bmi
print(f"Min BMI: {data_cleaned['BMI'].min()}")
print(f"Max BMI: {data_cleaned['BMI'].max()}")
print(f"Mean BMI: {data_cleaned['BMI'].mean()}")

#e) min, max, mean za osobe s dijabetesom i bez dijabetesa
diabetes_positive = data_cleaned[data_cleaned['Outcome'] == 1]
diabetes_negative = data_cleaned[data_cleaned['Outcome'] == 0]
print(f"Min BMI (diabetes positive): {diabetes_positive['BMI'].min()}")
print(f"Max BMI (diabetes positive): {diabetes_positive['BMI'].max()}")
print(f"Mean BMI (diabetes positive): {diabetes_positive['BMI'].mean():.2f}")
print(f"Min BMI (diabetes negative): {diabetes_negative['BMI'].min()}")
print(f"Max BMI (diabetes negative): {diabetes_negative['BMI'].max()}")
print(f"Mean BMI (diabetes negative): {diabetes_negative['BMI'].mean():.2f}") 
"""

# 002
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Split into input (X) and output (y) variables
# Define input and output variables
input_variables = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
output_variable = ["Outcome"]

X = data[input_variables].to_numpy()  # Input features
y = data[output_variable].to_numpy().ravel()  # Output variable (flattened)

# Split the data into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
#a) Izgradite model log. regresije
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Logistic regression model has been successfully trained.")

#b) klasifikacija skupine podataka za testiranje izgrađenog modela
y_pred = model.predict(X_test)

#c) matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#d) tocnost, preciznost i odziv
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
"""
# 003

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


#a) izgradite neuronsku mrezu
model = Sequential([
    Dense(12, activation='relu', input_shape=(8,)),  # First hidden layer
    Dense(8, activation='relu'),                    # Second hidden layer
    Dense(1, activation='sigmoid')                  # Output layer
])

# Print model summary
model.summary()

#b)podesite proces treniranja mreze
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Cross-entropy loss for binary classification
    metrics=['accuracy']
)

#c) trenirajte model
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=10,
    validation_split=0.1,  # Use 10% of the training data for validation
    verbose=1
)

#d) pohranite model na hdd
model.save('diabetes_model.h5')
print("Model saved as 'diabetes_model.h5'")

#e) evaluacija na testnom skupu
loaded_model = load_model("diabetes_model.h5")
print("Model loaded from 'diabetes_model.h5'")
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#f) predikcija na skupu za testiranje, matrica zabune za testni skup
y_pred = (loaded_model.predict(X_test) > 0.5).astype("int32")

conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=["No Diabetes", "Diabetes"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("Confusion Matrix:")
print(conf_matrix)






