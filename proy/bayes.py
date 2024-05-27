# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:52:47 2024

@author: nanor
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Cargar los datos preprocesados y las etiquetas
tfidf_df = pd.read_csv('tfidf_features.csv')
labels_df = pd.read_csv('labels.csv')

# Verificar que el número de muestras y etiquetas coincida
assert len(tfidf_df) == len(labels_df), "El número de muestras y etiquetas no coincide"

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(tfidf_df, labels_df, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train.values.ravel())

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

