import numpy as np
import pandas as pd
import sqlite3 as sql
import openpyxl
import a_funciones as fn ## para procesamiento
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors


def preprocesar():

    #### conectar_base_de_Datos#################
    conn=sql.connect('C:/Users/cesar/Documents/GitHub/T2_Marketing/MARKETING_E2_A_III/db_moviesF')
    cur=conn.cursor()
    

    # Ruta al archivo .ipynb que deseas ejecutar
    ruta_notebook = 'C:\\Users\\cesar\\Documents\\GitHub\\T2_Marketing\\MARKETING_E2_A_III\\Preprocesamiento.ipynb'

    # Lee el notebook
    with open(ruta_notebook, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Ejecuta el notebook
    executepreprocessor = ExecutePreprocessor(timeout=600, kernel_name='python3')
    executepreprocessor.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

    # Exporta el notebook ejecutado a HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb)

    ##### llevar datos que cambian constantemente a python ######
    movies=pd.read_sql('select * from full_ratings', conn )
    ratings=pd.read_sql('SELECT * FROM ratings', conn)
    usuarios=pd.read_sql('select distinct (user_id) as user_id from ratings_final',conn)

    
    #### transformación de datos crudos - Preprocesamiento ################
    movies['year']= movies.year.astype('int')

    ##### escalar para que año esté en el mismo rango ###
    sc=MinMaxScaler()
    movies[["year"]]=sc.fit_transform(movies[['year']])

    # Convertir la lista de géneros en múltiples columnas de género binario
    movies_dum1 = movies['genre_list'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0)

    # Concatenar las nuevas columnas de género con el DataFrame original
    movies_dum2 = pd.concat([movies.drop(columns=['movieId', 'year', 'title', 'genres', 'genre_list']), movies_dum1], axis=1)
    return movies_dum2,movies, conn, cur

##########################################################################
###############Función para entrenar modelo por cada usuario ##########
###############Basado en contenido todo lo visto por el usuario Knn#############################
def recomendar(user_id):
    
    books_dum2, books, conn, cur= preprocesar()
    
    ratings=pd.read_sql('select *from ratings_final where user_id=:user',conn, params={'user':user_id})
    l_books_r=ratings['isbn'].to_numpy()
    books_dum2[['isbn','book_title']]=books[['isbn','book_title']]
    books_r=books_dum2[books_dum2['isbn'].isin(l_books_r)]
    books_r=books_r.drop(columns=['isbn','book_title'])
    books_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    centroide=books_r.groupby("indice").mean()
    
    
    books_nr=books_dum2[~books_dum2['isbn'].isin(l_books_r)]
    books_nr=books_nr.drop(columns=['isbn','book_title'])
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(books_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0]
    recomend_b=books.loc[ids][['book_title','isbn']]
    
    
    return recomend_b


##### Generar recomendaciones para usuario lista de usuarios ####
##### No se hace para todos porque es muy pesado #############
def main(list_user):
    
    recomendaciones_todos=pd.DataFrame()
    for user_id in list_user:
            
        recomendaciones=recomendar(user_id)
        recomendaciones["user_id"]=user_id
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

    recomendaciones_todos.to_excel('C:\\cod\\LEA3_RecSys\\salidas\\reco\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv('C:\\cod\\LEA3_RecSys\\salidas\\reco\\recomendaciones.csv')


if __name__=="__main__":
    list_user=[52853,31226,167471,8066 ]
    main(list_user)
    

import sys
sys.executable