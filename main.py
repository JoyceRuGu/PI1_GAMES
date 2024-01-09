from fastapi import FastAPI
import pandas as pd
import uvicorn
import numpy as np 
from typing import Dict
from sklearn.utils.extmath           import randomized_svd
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

Df_muestra = pd.read_parquet('Df_muestra.parquet')
# Se crea una muestra de 5000 datos para poder realizar las consultas
Df_muestra = Df_muestra.head(2000)
app = FastAPI()

#Inicio
@app.get('/')
def start():
    return {'Mensaje': 'Proyecto N° 1 - Modelo de recomendación'}

# Función N°1. Proporciona el año en el que se registró la mayor cantidad de horas de juego para un género específico.
@app.get('/genero/{genre}')
def PlayTimeGenre(genre: str):
    # Filtrar los datos por el género proporcionado
    df_genres = Df_muestra[Df_muestra[genre] == 1]
    # Agrupar por año y calcular la suma de las horas jugadas
    sum_year_playtime = df_genres.groupby('release_year')['playtime_forever'].sum()
    # Encontrar el año con la mayor suma de horas jugadas
    max_year = sum_year_playtime.idxmax()
    # Retornar el resultado en un diccionario
    result = {
        "Año de lanzamiento con más horas jugadas para género": genre,
        "Año": int(max_year),
        "Horas Jugadas": int(sum_year_playtime[max_year])
    }
    return result


# # Función N°2. Devuelve el usuario que tiene la mayor cantidad de horas de juego en un género específico
@app.get('/usuario_por_genero/{genero}')
def UserForGenre(genero: str) -> dict:
    df_genres = Df_muestra[Df_muestra[genero] == 1]
    agg_df = df_genres.groupby('release_year').agg({'user_id': 'max', 'playtime_forever': 'sum'}).reset_index()
    playtime_list = agg_df.to_dict(orient='records')

    result = {
        "Usuario con más horas jugadas para Género " + genero: 
        df_genres.loc[df_genres['playtime_forever'].idxmax(), 'user_id'],
        "Horas jugadas": playtime_list
    }
    return result

# Función N°3 para obtener el top 3 de juegos más recomendados por usuarios por año.
@app.get("/Juegos más recomendados por usuarios/{year}")
def UsersRecommend(year : int):

    if year >= 2010 and year <= 2015:
        filtrado = Df_muestra[Df_muestra['year'] == year]
        filtrado = filtrado[filtrado['sentiment'].isin([1, 2])]
        conteo = filtrado.groupby('app_name')['recommend'].sum().reset_index()
        top_3 = conteo.nlargest(3, 'recommend')
        return {
            f"Top 3 de los juegos MÁS recomendados por el año {year}":
            [
                {'1': f"Name = {str(top_3['app_name'].iloc[0])}"},
                {'2': f"Name = {str(top_3['app_name'].iloc[1])}"},
                {'3': f"Name = {str(top_3['app_name'].iloc[2])}"}
            ]
        }
    else:
        return {f'Año no encontrado'}

# # Función N°4 Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado
@app.get('/Desarrolladoras con juegos MENOS recomendados por usuarios/{anio}')
def UsersWorstDeveloper(year: int):
    # Filtra el DataFrame por el año dado
    filtrado = Df_muestra[Df_muestra['year'] == year]

    # Agrupa por desarrolladora y calcula la media de las recomendaciones
    df_agrupado = filtrado.groupby('developer')['recommend'].mean().reset_index()

    # Ordena en orden ascendente (menor recomendación primero) y toma las primeras 3 desarrolladoras
    top3_desarrolladoras = df_agrupado.sort_values(by='sentiment').head(3)

    return {"top3_desarrolladoras_menos_recomendadas": top3_desarrolladoras.to_dict(orient='records')}



# Función N°5 Analisis de sentimiento
    
@app.get('/Analisis de sentimiento/{anio}')

def sentiment_analysis(año_de_lanzamiento:int) -> Dict[int, Dict[str, int]]:
        df = Df_muestra[Df_muestra['release_year'] == año_de_lanzamiento]
        sentiment_counts = df['sentiment'].value_counts()
        result = {año_de_lanzamiento: {
            'Negative': sentiment_counts.get(0, 0),
            'Neutral' : sentiment_counts.get(1, 0),
            'Positive': sentiment_counts.get(2, 0)
        }}

        return result

# Se crea el modelo de machine learning con Scikit-Learn
tfidf = TfidfVectorizer(stop_words='english')
Df_muestra=Df_muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(Df_muestra['review'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

# Se crea la funcion de recomendación de 5 juegos recomendados para el usuario ingresado.
@app.get('/Recomendación Usuario/{id}')
def recomendacion_usuario(id: int):
    if id not in Df_muestra['id'].values:
        return {'mensaje': 'No existe el id del usuario.'}
    titulo = Df_muestra.loc[Df_muestra['id'] == id, 'app_name'].iloc[0]
    sim_juegos = obtener_juegos_similares(titulo)
    
    return {'juegos recomendados': sim_juegos}

def obtener_juegos_similares(titulo: str, num_recomendaciones: int = 5):
    idx = Df_muestra[Df_muestra['app_name'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:num_recomendaciones + 1]]
    sim_juegos = Df_General['app_name'].iloc[sim_ind].values.tolist()
    
    return sim_juegos


