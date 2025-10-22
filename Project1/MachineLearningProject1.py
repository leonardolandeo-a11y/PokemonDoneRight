"""
Objetivo: 
    El objetivo de este proyecto es realizar un agrupamiento (clustering) sobre la columna "moves" de nuestra base de datos.  Con la ayuda de la herramienta TfidfVectorizer que nos proporciona sklearn, convertiremos nuestra columna "moves", originalmente con elementos de texto, a datos numéricos con los que alimentaremos un modelo de Machine Learning que realizará el clustering de nuestros datos.

Líneas resaltantes:
    (36): 
        Utilizaremos TfidfVectorizer, herramienta esencial para realizar un buen procesamiento de lenguaje natural.

    (38):
        Utilizaremos KMeans, algoritmo esencial para realizar el agrupamiento.
    
    (50-51):
        Realizamos la ejecución de TfidfVectorizer indicando que usaremos 1-grama y 2-grama, es decir, solo una palabra por elemento y luego 2 palabras por elemento. Además, guardamos lo que nos devuelve el método de la instancia para poder usarlo posteriormente.

    (56-57):
        Realizamos la ejecución de KMeans indicando 3 clusters, ya que dividiremos los clusters representaran que tan poderosa es una habilidad. Además, guardamos los clusters para usarlos posteriormente.
        
    (62-63):
        Impresion de numero de columnas.
    
    (67-68):
        Impresion de tokens.
    
    (74-76):
        Creacion de un DataFrame que contendra la matrix retornada de TfidfVectorizer con sus respectivas columnas.
    
    (87-91):
        Simplemente remplazaremos el numero de los clusters con la interpretacion que les habiamos dado. El cluster 0 representara a los pokemones que tengan movimientos genericos. El cluster 1 representara a los pokemones que tengan movimientos mas especiales. Por ultimo, el cluster 3 representara a los pokemones con movimientos ofensivos.

    (96-97):
        En esta última sección realizaremos la impresion de la base de datos final para posteriormente generar un archivo .csv con los datos.
"""


######## Importacion ########
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd   
from sklearn.cluster import KMeans

#############################

######## Abrir CSV ########
Pokemon = pd.read_csv("Project1/smogon.csv")
MovimientosPokemon = Pokemon["moves"].copy()
NombresPokemon = Pokemon["Pokemon"].copy()

###########################

######## TfidfVectorizer ########
TFIDF = TfidfVectorizer(ngram_range=(1,2))
MatrixNumericaTFIDF = TFIDF.fit_transform(MovimientosPokemon).toarray()

#################################

######## KMeans ########
Cluster = KMeans(n_clusters=3,n_init=10, random_state= 42) 
PokemonesAgrupados = Cluster.fit(MatrixNumericaTFIDF).labels_

########################

######## Numero de Columnas ########
NumeroColumnas = len(MatrixNumericaTFIDF[0])
print(f"Numero de columnas:\n{NumeroColumnas}")
####################################

######## Tokens ########
Tokens = TFIDF.vocabulary_
print(f"Tokens:\n{Tokens}")
print("\n")

########################

######## DataFrame con la matrix TFIDF ########
Vocabulario = TFIDF.get_feature_names_out()
TFIDF_DataFrame  = pd.DataFrame(MatrixNumericaTFIDF, columns = Vocabulario)
print(f"Dataframe con la matrix TF-IDF y con el vocabulario:\n{TFIDF_DataFrame }")

##################################

######## DataFrame final ########
SeriePokemonesAgrupados = pd.Series(PokemonesAgrupados,name = "ClusterPokemones")
PokemonDataSet = pd.concat([NombresPokemon, SeriePokemonesAgrupados],axis = 1)

#########################################################

######## Interpretacion de clusters ########
PokemonDataSet["ClusterPokemones"] = PokemonDataSet["ClusterPokemones"].replace({
    0:"Movimientos Genericos",
    1:"Movimientos Especiales",
    2:"Movimiento Ofensivo",
})

############################################

######### Muestra y conversion a CSV ########
print(PokemonDataSet)
PokemonDataSet.to_csv("Project1/PokemonDataset(Ejercicio1).csv")

#############################################
