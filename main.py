import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = pd.read_csv('iris.csv')
X = iris.iloc[:, :-1].values
y = iris['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Pakai decision tree
# Melatih model Decision Tree
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# Cek akurasi model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Akurasi: {accuracy * 100:.2f}%")

def custom_logic_model(data):
    predictions = []
    for row in data:
        if row[2] < 2.5 and row[3] < 1.0:
            predictions.append('Iris-setosa')
        elif row[0] < 5.8:
            predictions.append('Iris-versicolor')
        else:
            predictions.append('Iris-virginica')
    return predictions

y_pred = custom_logic_model(X_test)

# Input data dari pengguna
sepal_length = float(input("Masukkan sepal length (cm): "))
sepal_width = float(input("Masukkan sepal width (cm): "))
petal_length = float(input("Masukkan petal length (cm): "))
petal_width = float(input("Masukkan petal width (cm): "))

# Membuat array numpy dari input pengguna
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Memprediksi kelas
predicted_class = custom_logic_model(input_data)
print(f"Prediksi kelas: {predicted_class[0]}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy * 100:.2f}%")

