# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:09:20 2024

@author: nanor
"""

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Descargar recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')

def remove_special_characters(text):
    # Eliminar caracteres especiales y signos de puntuación
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

def convert_to_lowercase(text):
    # Convertir texto a minúsculas
    lowercase_text = text.lower()
    return lowercase_text

def tokenize_text(text):
    # Tokenizar el texto en palabras individuales
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    # Obtener lista de stopwords en español
    spanish_stopwords = set(stopwords.words('spanish'))
    # Eliminar stopwords del texto tokenizado
    filtered_tokens = [word for word in tokens if word not in spanish_stopwords]
    return filtered_tokens

def preprocess_text(text):
    # Preprocesar texto
    clean_text = remove_special_characters(text)
    clean_text = convert_to_lowercase(clean_text)
    tokens = tokenize_text(clean_text)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)

# Leer el texto desde un archivo .txt
file_path = 'reddit_posts.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Dividir el archivo en posts individuales
posts = text.split('\n\n')

# Aplicar preprocesamiento a cada post
preprocessed_posts = [preprocess_text(post) for post in posts]

# Guardar los posts preprocesados en un nuevo archivo
preprocessed_file_path = 'preprocessed_reddit_posts.txt'
with open(preprocessed_file_path, 'w', encoding='utf-8') as file:
    for post in preprocessed_posts:
        file.write("%s\n\n" % post)

# Vectorizar los posts preprocesados utilizando TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Puedes ajustar el número de características
tfidf_matrix = vectorizer.fit_transform(preprocessed_posts)

# Convertir la matriz TF-IDF a un DataFrame de pandas para su análisis
# Usa get_feature_names en lugar de get_feature_names_out
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names())

# Mostrar las primeras filas del DataFrame
print(tfidf_df.head())

# Guardar la matriz TF-IDF en un archivo CSV para su posterior análisis
tfidf_df.to_csv('conteo.csv', index=False)





