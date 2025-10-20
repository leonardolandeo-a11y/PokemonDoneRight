"""
Objetivo: 
    El objetivo de este proyecto es realizar un agrupamiento (clustering) sobre la columna "moves" de nuestra base de datos. Haremos uso de un vocabulario controlado y, con la ayuda de la herramienta TfidfVectorizer que nos proporciona sklearn, convertiremos nuestra columna "moves", originalmente con elementos de texto, a datos numéricos con los que alimentaremos un modelo de Machine Learning que realizará el clustering de nuestros datos.

Líneas resaltantes:
    (30): 
        Utilizaremos TfidfVectorizer, herramienta esencial para realizar un buen procesamiento de lenguaje natural.

    (32):
        Utilizaremos KMeans, algoritmo esencial para realizar el agrupamiento.

    (42): 
        Se define nuestro vocabulario controlado a partir de las características de los Pokémon.
        
    (52-54):
        En este apartado del código realizaremos una limpieza de los datos. Como podemos observar, el texto dentro de la columna "moves" está junto (por ejemplo: "HolaMundo"). Para poder utilizar TfidfVectorizer, necesitamos que los datos estén separados por espacios (por ejemplo: "Hola Mundo"). El objetivo de esta parte del código es, a partir de nuestro vocabulario controlado, buscar en cada fila de la columna "moves" y, al momento de encontrar una coincidencia, agregarle un espacio a la izquierda y a la derecha de la palabra (por ejemplo, si el vocabulario contiene "Poison": "EjemploEjemploPoisonEjemplo" → "EjemploEjemplo Poison Ejemplo").
    
    (59-60):
        Realizamos la ejecución de TfidfVectorizer indicando que usaremos 1-grama, es decir, solo una palabra por elemento. Además, guardamos lo que nos devuelve el método de la instancia para poder usarlo posteriormente.

    (65-66):
        Realizamos la ejecución de KMeans indicando 17 clusters, ya que son la cantidad de tipos de Pokémon que existen dentro de nuestra base de datos. Además, guardamos los clusters para usarlos posteriormente.
    
    (71-74):
        En esta última sección realizaremos la creación de la base de datos final para posteriormente generar un archivo .csv con los datos.
"""


######## Importacion ########
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans

#############################

######## Abrir CSV ########
Pokemon = pd.read_csv("Project2/smogon.csv")
NombresPokemon = Pokemon['Pokemon'].copy()
###########################

######## Vocabulario Controlado ########
VocabularioControlado =  [
    'bug', 'dark', 'dragon', 'electric', 'fairy', 
    'fighting', 'fire', 'flying', 'ghost', 'grass',
    'ground', 'ice', 'poison', 'psychic', 'rock', 
    'steel','water'
]

########################################

######## Limpieza de datos ########
for TipoPokemon in VocabularioControlado:
    Pokemon["moves"] = Pokemon["moves"].str.replace(TipoPokemon, f" {TipoPokemon} ", regex=False)
MovimientoPokemon = Pokemon["moves"].str.replace(r"\s+", " ", regex=True).str.strip()

###################################

######## TfidfVectorizer ########
TFIDF = TfidfVectorizer(vocabulary=VocabularioControlado,ngram_range=(1,1))
MatrixNumericaTFIDF = TFIDF.fit_transform(MovimientoPokemon).toarray()

#################################

######## KMeans ########
Clusters = KMeans(n_clusters=17)
PokemonesAgrupados = Clusters.fit(MatrixNumericaTFIDF).labels_

########################

######## Base de datos (Resultado) ########
PokemonesAgrupadosSerie = pd.Series(PokemonesAgrupados,name="ClusterPokemon")

PokemonDataSet = pd.concat([NombresPokemon,PokemonesAgrupadosSerie],axis = 1)
PokemonDataSet.to_csv("Project2/PokemonDataSet(Ejercicio2).csv")

###########################################