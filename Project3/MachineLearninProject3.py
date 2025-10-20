"""
Objetivo: 
    El objetivo de este proyecto es obtener la habilidad dominante de cada pokemon. Para ello, partiremos de un vocabulario controlado a partir del cual se generará una matriz numérica mediante herramienta CountVectorizer. Contaremos la aparición de los elementos del vocabulario controlado en cada elemento de la columna "moves". Luego, la habilidad mas frecuente de cada pokemon sera su habilidad dominante.

Líneas resaltantes:
    (30): 
        Utilizaremos CountVectorizer para contar las apariciones del vocabulario controlado.
    
    (43): 
        Se define nuestro vocabulario controlado a partir de las características de los pokémon.
    
    (54-56): 
        En este apartado del código realizaremos una limpieza de los datos. 
        Como podemos observar, el texto dentro de la columna "moves" está junto (por ejemplo: "HolaMundo"). 
        Para poder utilizar CountVectorizer, necesitamos que los datos estén separados por espacios (por ejemplo: "Hola Mundo"). El objetivo de esta parte del código es, a partir de nuestro vocabulario controlado, buscar en cada fila de la columna "moves" y, al momento de encontrar una coincidencia, agregarle un espacio a la izquierda y a la derecha de la palabra (por ejemplo, si el vocabulario contiene "Poison": "EjemploEjemploPoisonEjemplo" → "EjemploEjemplo Poison Ejemplo").
    
    (62-63): 
        Realizamos la ejecución de CountVectorizer y convertimos lo que nos devuelve el método fit_transform de la instancia en una matriz, para poder visualizar de mejor manera los datos.
    
    (69-77):
        Creamos un DataFrame con los datos obtenidos previamente en la ejecución de CountVectorizer. Además, a este DataFrame le agregamos nombres para cada columna y una columna que almacena los nombres de los pokémon.
    
    (72):
        En esta parte, por cada pokémon recorreremos todas sus habilidades, y aquella que tenga el valor más alto será su habilidad dominante. Crearemos una nueva columna para ello y la agregaremos a la base de datos posteriormente. Además, crearemos un documento CSV.
"""



######## Importacion ########
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 

#############################


######## Abrir CSV ########
Pokemon = pd.read_csv("Project3/smogon.csv")
NombresPokemon = Pokemon["Pokemon"].copy()

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
    Pokemon["moves"] = Pokemon["moves"].str.replace(TipoPokemon, f" {TipoPokemon} ",regex= False) 
MovimientoPokemon = Pokemon["moves"].str.replace(r"\s+", " ", regex=True).str.strip()

###################################


######## CountVectorizer ########
Count = CountVectorizer(vocabulary=VocabularioControlado)
RepeticionesMatrixNumerica = Count.fit_transform(MovimientoPokemon).toarray()

#################################

######## Base de datos (Resultado) ########
#creacion:
PokemonDataSet = pd.DataFrame(RepeticionesMatrixNumerica,columns=VocabularioControlado)

#Obtencion de la habilidad dominante:
HabilidadMasFrecuente= PokemonDataSet[VocabularioControlado].idxmax(axis = 1)

#Agregacion final de datos:
PokemonDataSet.insert(0,"NombresPokemon",NombresPokemon)
PokemonDataSet.insert(1,"TipoDominante",HabilidadMasFrecuente)

print(PokemonDataSet)
PokemonDataSet.to_csv("Project3/PokemonDataSet(3).csv")