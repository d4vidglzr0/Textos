# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:51:49 2024

@author: nanor
"""

import pandas as pd

# Ejemplo de etiquetas para los posts (0: no tóxico, 1: tóxico)
# Deberías reemplazar estas etiquetas con las correctas para tus datos
labels = [0, 1, 0, 0, 1]  # Asegúrate de que este array tenga tantas etiquetas como posts

# Crear un DataFrame de pandas
labels_df = pd.DataFrame(labels, columns=['label'])

# Guardar el DataFrame en un archivo CSV
labels_df.to_csv('labels.csv', index=False)

print("Archivo labels.csv creado.")
