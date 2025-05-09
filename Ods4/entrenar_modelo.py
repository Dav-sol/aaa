import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Cargar el dataset
df = pd.read_csv('datos_desercion.csv')

# Separar variables predictoras (X) y la variable objetivo (y)
X = df.drop('riesgo', axis=1)  # Las características (sin la columna 'riesgo')
y = df['riesgo']  # La columna 'riesgo' es la etiqueta que queremos predecir

# Crear el clasificador (árbol de decisión)
modelo = DecisionTreeClassifier()

# Entrenar el modelo
modelo.fit(X, y)

# Guardar el modelo entrenado en un archivo
with open('modelo_arbol.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo entrenado y guardado como modelo_arbol.pkl")
